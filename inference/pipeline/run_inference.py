"""
inference/pipeline/run_inference.py
=====================================
Main real-time inference pipeline.

Supports:
  --source webcam
  --source video path/to/video.mp4
  --source image path/to/image.jpg
  --source rtsp rtsp://user:pass@192.168.1.100:554/stream
  --source ipwebcam http://192.168.1.105:8080/video

Outputs:
  - Live annotated feed
  - Suspicious event timestamps (video)
  - Corrected labels (active learning)
"""

import cv2
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pose.analyzers.pose_analyzer import TemporalPoseAnalyzer
from fusion.fusion_engine import FusionEngine, TemporalSmoother, AlertLevel
from active_learning.feedback.feedback_collector import FeedbackCollector
from utils.visualization.overlay import draw_overlay


# ─────────────────────────────────────────────
# INFERENCE ENGINE
# ─────────────────────────────────────────────

class SuspiciousActivityDetector:
    """
    Full real-time detection pipeline.
    Combines YOLOv8 + Temporal Pose Analysis + Fusion.
    """

    def __init__(
        self,
        model_path: str = "models/checkpoints/best.pt",
        fusion_strategy: str = "weighted_avg",
        save_output: bool = False,
        output_dir: str = "outputs",
        enable_active_learning: bool = True,
        show_ui: bool = True,
    ):
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.pose_analyzer = TemporalPoseAnalyzer()
        self.fusion = FusionEngine(strategy=fusion_strategy)
        self.smoother = TemporalSmoother(alpha=0.35)
        self.feedback = FeedbackCollector() if enable_active_learning else None
        self.save_output = save_output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.show_ui = show_ui
        self.suspicious_timestamps = []

    def _get_yolo_score(self, frame: np.ndarray) -> tuple[float, str]:
        """Run YOLO classification on frame. Returns (suspicious_prob, class_name)."""
        results = self.model.predict(frame, verbose=False, imgsz=224)
        probs = results[0].probs
        suspicious_idx = self.model.names.index("suspicious") if "suspicious" in self.model.names else 1
        suspicious_prob = float(probs.data[suspicious_idx])
        top_class = self.model.names[int(probs.top1)]
        return suspicious_prob, top_class

    def process_frame(self, frame: np.ndarray, timestamp: float = 0.0) -> dict:
        """
        Process a single frame through the full pipeline.
        Returns a result dict.
        """
        # 1. YOLO classification
        yolo_score, yolo_class = self._get_yolo_score(frame)

        # 2. Pose analysis
        pose_result = self.pose_analyzer.analyze(frame)

        # 3. Temporal smoothing on pose score
        smoothed_pose = self.smoother.smooth(pose_result.suspicious_score)

        # 4. Fusion
        fusion_result = self.fusion.fuse(
            yolo_score=yolo_score,
            pose_score=smoothed_pose,
            temporal_score=0.0,
            triggered_rules=pose_result.triggered_rules,
        )

        # 5. Track suspicious timestamps
        if fusion_result.alert_level == AlertLevel.SUSPICIOUS:
            self.suspicious_timestamps.append({
                "timestamp": round(timestamp, 2),
                "fused_score": round(fusion_result.fused_score, 3),
                "yolo_score": round(yolo_score, 3),
                "pose_rules": pose_result.triggered_rules,
            })

        # 6. Annotate frame
        annotated = draw_overlay(
            frame=pose_result.annotated_frame if pose_result.annotated_frame is not None else frame,
            fusion_result=fusion_result,
            yolo_score=yolo_score,
            pose_score=smoothed_pose,
            timestamp=timestamp,
        )

        # 7. Active learning: flag low confidence frames
        if self.feedback and fusion_result.is_low_confidence:
            self.feedback.flag_frame(frame, fusion_result, timestamp)

        return {
            "annotated_frame": annotated,
            "fusion_result": fusion_result.to_dict(),
            "yolo_class": yolo_class,
            "timestamp": timestamp,
        }

    def process_image(self, image_path: str) -> dict:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        result = self.process_frame(frame)
        if self.show_ui:
            cv2.imshow("Detection Result", result["annotated_frame"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if self.save_output:
            out_path = self.output_dir / f"result_{Path(image_path).stem}.jpg"
            cv2.imwrite(str(out_path), result["annotated_frame"])
            logger.info(f"Saved result to {out_path}")
        return result

    def process_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = 0
        writer = None

        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = self.output_dir / f"result_{Path(video_path).stem}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp = frame_count / fps

            result = self.process_frame(frame, timestamp=timestamp)
            annotated = result["annotated_frame"]

            if writer:
                writer.write(annotated)

            if self.show_ui:
                cv2.imshow("Suspicious Activity Detection", annotated)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
                # Active learning: press 'c' to correct label
                if key == ord("c") and self.feedback:
                    correct = input("Correct label (normal/suspicious): ").strip()
                    if correct in ("normal", "suspicious"):
                        self.feedback.add_correction(frame, result["fusion_result"], correct)

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Save suspicious timestamps
        if self.suspicious_timestamps:
            ts_path = self.output_dir / f"timestamps_{Path(video_path).stem}.json"
            with open(ts_path, "w") as f:
                json.dump(self.suspicious_timestamps, f, indent=2)
            logger.info(f"Saved {len(self.suspicious_timestamps)} suspicious timestamps to {ts_path}")

        return {"suspicious_timestamps": self.suspicious_timestamps}

    def process_stream(self, source: str):
        """Process live stream: webcam, RTSP, or IP webcam URL."""
        if source == "webcam":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ConnectionError(f"Cannot open stream: {source}")

        logger.info(f"Streaming from: {source}")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            timestamp = time.time() - start_time

            result = self.process_frame(frame, timestamp=timestamp)
            annotated = result["annotated_frame"]

            # FPS counter
            fps = frame_count / max(timestamp, 0.001)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if self.show_ui:
                cv2.imshow("Live Detection", annotated)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Suspicious Activity Detection")
    parser.add_argument("--source", required=True,
                        help="Source: 'webcam', video path, image path, RTSP URL, or IP webcam URL")
    parser.add_argument("--model", default="models/checkpoints/best.pt")
    parser.add_argument("--strategy", default="weighted_avg",
                        choices=["weighted_avg", "max_vote", "any_trigger", "all_confirm"])
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()

    detector = SuspiciousActivityDetector(
        model_path=args.model,
        fusion_strategy=args.strategy,
        save_output=args.save,
        show_ui=not args.no_ui,
    )

    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        detector.process_image(args.source)
    elif args.source.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        detector.process_video(args.source)
    else:
        detector.process_stream(args.source)


if __name__ == "__main__":
    main()
