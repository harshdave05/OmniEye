"""
active_learning/feedback/feedback_collector.py
===============================================
Saves flagged low-confidence frames and user corrections.
Builds a corrected dataset for fine-tuning.
"""

import cv2
import json
import shutil
from pathlib import Path
from datetime import datetime
from loguru import logger


class FeedbackCollector:
    """
    Manages the active learning feedback loop:
    1. Flags low-confidence frames automatically
    2. Accepts manual corrections from user
    3. Exports corrected dataset for fine-tuning
    """

    def __init__(self, base_dir: str = "active_learning/feedback"):
        self.base_dir = Path(base_dir)
        self.flagged_dir = self.base_dir / "flagged_frames"
        self.corrections_file = self.base_dir / "corrected_labels.json"
        self.corrected_dataset = self.base_dir / "corrected_dataset"

        self.flagged_dir.mkdir(parents=True, exist_ok=True)
        self.corrected_dataset.mkdir(parents=True, exist_ok=True)
        for cls in ["normal", "suspicious"]:
            (self.corrected_dataset / cls).mkdir(exist_ok=True)

        self._corrections = self._load_corrections()
        self._flag_count = 0

    def _load_corrections(self) -> list:
        if self.corrections_file.exists():
            with open(self.corrections_file) as f:
                return json.load(f)
        return []

    def _save_corrections(self):
        with open(self.corrections_file, "w") as f:
            json.dump(self._corrections, f, indent=2)

    def flag_frame(self, frame, fusion_result, timestamp: float):
        """
        Auto-flag a low-confidence frame for later human review.
        """
        self._flag_count += 1
        fname = f"flagged_{self._flag_count:05d}_t{timestamp:.2f}.jpg"
        fpath = self.flagged_dir / fname
        cv2.imwrite(str(fpath), frame)

        logger.debug(f"Flagged low-confidence frame: {fname} (fused={fusion_result.fused_score:.3f})")

    def add_correction(self, frame, fusion_result_dict: dict, correct_label: str):
        """
        Record a manual correction and save the frame to the corrected dataset.
        """
        assert correct_label in ("normal", "suspicious"), "Label must be 'normal' or 'suspicious'"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fname = f"corrected_{ts}.jpg"
        dest = self.corrected_dataset / correct_label / fname
        cv2.imwrite(str(dest), frame)

        record = {
            "timestamp": ts,
            "file": str(dest),
            "correct_label": correct_label,
            "predicted": fusion_result_dict,
        }
        self._corrections.append(record)
        self._save_corrections()
        logger.info(f"Correction saved: predicted={fusion_result_dict.get('alert_level')} → correct={correct_label}")

    def export_retraining_stats(self) -> dict:
        """Return stats on how many corrections have been collected."""
        normal_count = len(list((self.corrected_dataset / "normal").glob("*.jpg")))
        suspicious_count = len(list((self.corrected_dataset / "suspicious").glob("*.jpg")))
        flagged_count = len(list(self.flagged_dir.glob("*.jpg")))

        return {
            "total_corrections": len(self._corrections),
            "normal_corrections": normal_count,
            "suspicious_corrections": suspicious_count,
            "flagged_for_review": flagged_count,
            "dataset_path": str(self.corrected_dataset),
        }

    def ready_for_retraining(self, min_samples: int = 50) -> bool:
        """Returns True if enough corrections exist to trigger fine-tuning."""
        stats = self.export_retraining_stats()
        return (
            stats["normal_corrections"] >= min_samples
            and stats["suspicious_corrections"] >= min_samples
        )
