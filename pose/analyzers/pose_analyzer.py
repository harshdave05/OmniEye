"""
pose/analyzers/pose_analyzer.py
================================
MediaPipe-based pose analysis with rule-based suspicious gesture detection.
Outputs confidence scores, not just binary labels.
"""

import mediapipe as mp
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class PoseAnalysisResult:
    pose_detected: bool
    suspicious_score: float           # 0.0 → 1.0
    triggered_rules: list[str] = field(default_factory=list)
    landmarks: Optional[object] = None
    annotated_frame: Optional[np.ndarray] = None

    @property
    def label(self) -> str:
        if self.suspicious_score >= 0.7:
            return "suspicious"
        elif self.suspicious_score >= 0.4:
            return "possibly_suspicious"
        return "normal"


# ─────────────────────────────────────────────
# LANDMARK HELPERS
# ─────────────────────────────────────────────

def _lm(landmarks, idx: int) -> tuple[float, float, float]:
    """Extract normalized (x, y, z) for a landmark index."""
    lm = landmarks.landmark[idx]
    return lm.x, lm.y, lm.z


def _distance(p1, p2) -> float:
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def _angle_degrees(a, b, c) -> float:
    """Angle at point B formed by A-B-C."""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


# ─────────────────────────────────────────────
# POSE RULES ENGINE
# ─────────────────────────────────────────────

class PoseRulesEngine:
    """
    Evaluates a set of pose rules against MediaPipe landmarks.
    Each rule returns a score contribution (0.0–1.0) and label.
    Final score is a weighted sum capped at 1.0.
    """

    RULES_WEIGHTS = {
        "hands_raised_above_head": 0.3,
        "arm_extended_forward": 0.4,      # possible weapon pointing
        "both_arms_extended": 0.5,
        "elbow_bent_aggressive": 0.3,
        "torso_leaning_forward": 0.2,
        "asymmetric_arm_raise": 0.25,
        "hands_at_chest_level": 0.15,
    }

    def __init__(self, sensitivity: float = 1.0):
        """
        sensitivity: 1.0 = default. 0.5 = less sensitive (fewer false positives).
        """
        self.sensitivity = sensitivity

    def evaluate(self, landmarks) -> tuple[float, list[str]]:
        """
        Returns (suspicious_score 0–1.0, list of triggered rule names)
        """
        triggered = []
        total_score = 0.0

        lm = landmarks.landmark

        # Convenience: extract key points
        nose = _lm(landmarks, mp_pose.PoseLandmark.NOSE)
        left_shoulder = _lm(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER)
        right_shoulder = _lm(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER)
        left_elbow = _lm(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW)
        right_elbow = _lm(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW)
        left_wrist = _lm(landmarks, mp_pose.PoseLandmark.LEFT_WRIST)
        right_wrist = _lm(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST)
        left_hip = _lm(landmarks, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = _lm(landmarks, mp_pose.PoseLandmark.RIGHT_HIP)

        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2

        # ── Rule 1: Hands raised above head ──
        if left_wrist[1] < nose[1] and right_wrist[1] < nose[1]:
            triggered.append("hands_raised_above_head")
            total_score += self.RULES_WEIGHTS["hands_raised_above_head"] * self.sensitivity

        # ── Rule 2: One arm extended forward (gun pointing pose) ──
        left_arm_vec = np.array([left_wrist[0] - left_shoulder[0], left_wrist[1] - left_shoulder[1]])
        right_arm_vec = np.array([right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1]])

        for arm_vec, side in [(left_arm_vec, "L"), (right_arm_vec, "R")]:
            arm_len = np.linalg.norm(arm_vec)
            if arm_len > 0.25:  # extended arm in normalized coords
                arm_angle = abs(np.degrees(np.arctan2(arm_vec[1], arm_vec[0])))
                if arm_angle < 30:  # nearly horizontal = forward point
                    triggered.append(f"arm_extended_forward_{side}")
                    total_score += self.RULES_WEIGHTS["arm_extended_forward"] * self.sensitivity

        # ── Rule 3: Both arms extended (threatening) ──
        if np.linalg.norm(left_arm_vec) > 0.22 and np.linalg.norm(right_arm_vec) > 0.22:
            triggered.append("both_arms_extended")
            total_score += self.RULES_WEIGHTS["both_arms_extended"] * self.sensitivity

        # ── Rule 4: Elbow bent aggressively (boxing/punching) ──
        left_elbow_angle = _angle_degrees(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = _angle_degrees(right_shoulder, right_elbow, right_wrist)

        if left_elbow_angle < 90 and left_wrist[1] < left_shoulder[1]:
            triggered.append("elbow_bent_aggressive_L")
            total_score += self.RULES_WEIGHTS["elbow_bent_aggressive"] * self.sensitivity

        if right_elbow_angle < 90 and right_wrist[1] < right_shoulder[1]:
            triggered.append("elbow_bent_aggressive_R")
            total_score += self.RULES_WEIGHTS["elbow_bent_aggressive"] * self.sensitivity

        # ── Rule 5: Torso leaning aggressively forward ──
        torso_height = hip_y - shoulder_y
        if torso_height < 0.15:  # compressed torso = leaning forward
            triggered.append("torso_leaning_forward")
            total_score += self.RULES_WEIGHTS["torso_leaning_forward"] * self.sensitivity

        # ── Rule 6: Asymmetric arm raise (one arm up) ──
        arm_height_diff = abs(left_wrist[1] - right_wrist[1])
        if arm_height_diff > 0.3:
            triggered.append("asymmetric_arm_raise")
            total_score += self.RULES_WEIGHTS["asymmetric_arm_raise"] * self.sensitivity

        return min(total_score, 1.0), triggered


# ─────────────────────────────────────────────
# MAIN POSE ANALYZER
# ─────────────────────────────────────────────

class PoseAnalyzer:
    """
    Wraps MediaPipe Pose + PoseRulesEngine.
    Thread-safe: create one instance per thread/stream.
    """

    def __init__(
        self,
        sensitivity: float = 1.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        draw_skeleton: bool = True,
    ):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.rules_engine = PoseRulesEngine(sensitivity=sensitivity)
        self.draw_skeleton = draw_skeleton

    def analyze(self, frame: np.ndarray) -> PoseAnalysisResult:
        """
        Analyze a single BGR frame.
        Returns PoseAnalysisResult with suspicious score and triggered rules.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return PoseAnalysisResult(pose_detected=False, suspicious_score=0.0)

        score, rules = self.rules_engine.evaluate(results.pose_landmarks)

        annotated = frame.copy()
        if self.draw_skeleton:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

        return PoseAnalysisResult(
            pose_detected=True,
            suspicious_score=score,
            triggered_rules=rules,
            landmarks=results.pose_landmarks,
            annotated_frame=annotated,
        )

    def close(self):
        self.pose.close()


# ─────────────────────────────────────────────
# TEMPORAL POSE ANALYZER (tracks motion across frames)
# ─────────────────────────────────────────────

class TemporalPoseAnalyzer:
    """
    Tracks wrist positions over time to detect sudden motion spikes.
    Spike = high-velocity movement = aggressive gesture.
    """

    def __init__(self, window_size: int = 10, spike_threshold: float = 0.05):
        self.analyzer = PoseAnalyzer()
        self.window_size = window_size
        self.spike_threshold = spike_threshold
        self._wrist_history = []     # list of (lx, ly, rx, ry)

    def analyze(self, frame: np.ndarray) -> PoseAnalysisResult:
        result = self.analyzer.analyze(frame)

        if not result.pose_detected:
            self._wrist_history.clear()
            return result

        lm = result.landmarks.landmark
        lw = (lm[mp_pose.PoseLandmark.LEFT_WRIST].x, lm[mp_pose.PoseLandmark.LEFT_WRIST].y)
        rw = (lm[mp_pose.PoseLandmark.RIGHT_WRIST].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y)
        self._wrist_history.append((*lw, *rw))

        if len(self._wrist_history) > self.window_size:
            self._wrist_history.pop(0)

        # Detect sudden velocity spike
        motion_score = 0.0
        if len(self._wrist_history) >= 3:
            positions = np.array(self._wrist_history)
            velocities = np.diff(positions, axis=0)
            max_velocity = np.max(np.abs(velocities))

            if max_velocity > self.spike_threshold:
                motion_score = min(max_velocity / (self.spike_threshold * 3), 1.0)
                result.triggered_rules.append(f"sudden_motion_spike={max_velocity:.3f}")

        # Blend motion score with pose score
        result.suspicious_score = min(result.suspicious_score + motion_score * 0.4, 1.0)
        return result
