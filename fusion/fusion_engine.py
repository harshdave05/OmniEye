"""
fusion/fusion_engine.py
========================
Multi-model decision fusion combining:
  1. YOLOv8 classification score
  2. MediaPipe pose rule score
  3. Optional temporal motion score

Uses a weighted soft fusion strategy with configurable weights.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ─────────────────────────────────────────────
# ALERT LEVELS
# ─────────────────────────────────────────────

class AlertLevel:
    NORMAL = "normal"
    POSSIBLE = "possibly_suspicious"
    SUSPICIOUS = "suspicious"

    @staticmethod
    def color_bgr(level: str) -> tuple[int, int, int]:
        return {
            AlertLevel.NORMAL: (0, 200, 0),        # Green
            AlertLevel.POSSIBLE: (0, 165, 255),     # Orange
            AlertLevel.SUSPICIOUS: (0, 0, 220),     # Red
        }.get(level, (200, 200, 200))

    @staticmethod
    def color_hex(level: str) -> str:
        return {
            AlertLevel.NORMAL: "#00C800",
            AlertLevel.POSSIBLE: "#FFA500",
            AlertLevel.SUSPICIOUS: "#DC0000",
        }.get(level, "#CCCCCC")


# ─────────────────────────────────────────────
# FUSION RESULT
# ─────────────────────────────────────────────

@dataclass
class FusionResult:
    yolo_score: float               # Raw YOLO suspicious probability
    pose_score: float               # Raw pose rule score
    temporal_score: float           # Motion/temporal score
    fused_score: float              # Final fused score (0–1)
    alert_level: str                # AlertLevel constant
    triggered_rules: list[str] = field(default_factory=list)
    is_low_confidence: bool = False # Flag for active learning

    @property
    def color_bgr(self) -> tuple:
        return AlertLevel.color_bgr(self.alert_level)

    @property
    def color_hex(self) -> str:
        return AlertLevel.color_hex(self.alert_level)

    def to_dict(self) -> dict:
        return {
            "yolo_score": round(self.yolo_score, 4),
            "pose_score": round(self.pose_score, 4),
            "temporal_score": round(self.temporal_score, 4),
            "fused_score": round(self.fused_score, 4),
            "alert_level": self.alert_level,
            "triggered_rules": self.triggered_rules,
            "is_low_confidence": self.is_low_confidence,
        }


# ─────────────────────────────────────────────
# FUSION ENGINE
# ─────────────────────────────────────────────

class FusionEngine:
    """
    Combines multiple model outputs into a final suspicious activity score.

    Fusion Strategies:
        'weighted_avg'   : Weighted average of all scores
        'max_vote'       : Take the max score across models
        'any_trigger'    : Suspicious if EITHER model triggers above threshold
        'all_confirm'    : Suspicious only if ALL models agree
    """

    # Thresholds
    SUSPICIOUS_THRESHOLD = 0.65
    POSSIBLE_THRESHOLD = 0.40
    LOW_CONFIDENCE_BAND = (0.40, 0.65)   # flag for active learning

    def __init__(
        self,
        strategy: str = "weighted_avg",
        yolo_weight: float = 0.55,
        pose_weight: float = 0.30,
        temporal_weight: float = 0.15,
        sensitivity: float = 1.0,
    ):
        assert strategy in ("weighted_avg", "max_vote", "any_trigger", "all_confirm")
        assert abs(yolo_weight + pose_weight + temporal_weight - 1.0) < 1e-4

        self.strategy = strategy
        self.weights = np.array([yolo_weight, pose_weight, temporal_weight])
        self.sensitivity = sensitivity   # multiplier on final score

    def fuse(
        self,
        yolo_score: float,
        pose_score: float,
        temporal_score: float = 0.0,
        triggered_rules: Optional[list] = None,
    ) -> FusionResult:
        """
        Produce a fused decision from individual model scores.

        Args:
            yolo_score: Probability of 'suspicious' from YOLOv8 (0–1)
            pose_score: Rule-based suspicious score from pose (0–1)
            temporal_score: Motion-based temporal score (0–1)
            triggered_rules: List of triggered pose rule names

        Returns:
            FusionResult with final decision
        """
        scores = np.array([yolo_score, pose_score, temporal_score])

        if self.strategy == "weighted_avg":
            fused = float(np.dot(scores, self.weights))

        elif self.strategy == "max_vote":
            fused = float(np.max(scores))

        elif self.strategy == "any_trigger":
            # Suspicious if any model passes threshold
            any_suspicious = any(s >= self.SUSPICIOUS_THRESHOLD for s in scores[:2])
            fused = max(scores) if any_suspicious else float(np.dot(scores, self.weights))

        elif self.strategy == "all_confirm":
            # Both YOLO and pose must agree
            if yolo_score >= self.SUSPICIOUS_THRESHOLD and pose_score >= 0.3:
                fused = float(np.dot(scores, self.weights))
            else:
                fused = min(float(np.dot(scores, self.weights)), self.POSSIBLE_THRESHOLD - 0.01)

        # Apply sensitivity multiplier
        fused = min(fused * self.sensitivity, 1.0)

        # Determine alert level
        if fused >= self.SUSPICIOUS_THRESHOLD:
            alert_level = AlertLevel.SUSPICIOUS
        elif fused >= self.POSSIBLE_THRESHOLD:
            alert_level = AlertLevel.POSSIBLE
        else:
            alert_level = AlertLevel.NORMAL

        # Low confidence flag for active learning
        is_low_conf = self.LOW_CONFIDENCE_BAND[0] <= fused <= self.LOW_CONFIDENCE_BAND[1]

        return FusionResult(
            yolo_score=yolo_score,
            pose_score=pose_score,
            temporal_score=temporal_score,
            fused_score=fused,
            alert_level=alert_level,
            triggered_rules=triggered_rules or [],
            is_low_confidence=is_low_conf,
        )


# ─────────────────────────────────────────────
# TEMPORAL BUFFER (smoothing over N frames)
# ─────────────────────────────────────────────

class TemporalSmoother:
    """
    Prevents flickering by smoothing the fused score over a window of frames.
    Uses exponential moving average (EMA).
    """

    def __init__(self, alpha: float = 0.3, window: int = 8):
        """
        alpha: EMA weight for new score (lower = more smoothing)
        window: Max history for fallback mean
        """
        self.alpha = alpha
        self.window = window
        self._ema = None
        self._history = []

    def smooth(self, score: float) -> float:
        if self._ema is None:
            self._ema = score
        else:
            self._ema = self.alpha * score + (1 - self.alpha) * self._ema

        self._history.append(score)
        if len(self._history) > self.window:
            self._history.pop(0)

        return self._ema

    def reset(self):
        self._ema = None
        self._history.clear()
