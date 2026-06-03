"""
OmniEye AI — Backend v6.1
──────────────────────────────────────────────────────────────
CHANGES vs v6.0
  1. WEAPON_EMOJI removed — weapon output is clean (no emoji)
  2. Adaptive detection:
       - frame_count % FRAME_SKIP (default 5)
       - AND time elapsed >= MIN_DETECT_INTERVAL (default 1.0 s)
       - After suspicious detection: FRAME_SKIP drops to 2 (faster)
       - After returning normal:     FRAME_SKIP resets to 6 (save CPU)
  3. Blur detection: skip frames where
       cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD
  4. Description text cleaned (no emoji inside description strings)
  5. All other v6.0 features retained (MJPEG, DirectShow, threads)

ARCHITECTURE (unchanged from v6.0)
  Thread-1 per feed: capture_loop  — 15 fps, JPEG encode
  Thread-2 per feed: detection_loop — adaptive frame-skip + blur check
  FastAPI: /stream/{id} MJPEG, /ws/alerts events only
──────────────────────────────────────────────────────────────
pip install fastapi "uvicorn[standard]" opencv-python ultralytics
            loguru python-multipart websockets reportlab twilio
"""

from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import (
    BackgroundTasks, FastAPI, File, Form,
    HTTPException, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from loguru import logger
from pydantic import BaseModel

# Silence MSMF before any CV2 call
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ── Optional libs ─────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True

    # ── PyTorch 2.6 fix: allowlist ultralytics globals ──────────
    try:
        import torch.serialization as _ts
        from ultralytics.nn.tasks import (
            ClassificationModel, DetectionModel,
            SegmentationModel, PoseModel,
        )
        from ultralytics.nn.modules.head import Classify, Detect, Segment, Pose
        if hasattr(_ts, "add_safe_globals"):
            _ts.add_safe_globals([
                ClassificationModel, DetectionModel,
                SegmentationModel, PoseModel,
                Classify, Detect, Segment, Pose,
            ])
            logger.info("PyTorch 2.6 safe_globals patch applied for ultralytics models")
    except Exception as _e:
        logger.warning(f"PyTorch safe_globals patch skipped: {_e}")

except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("ultralytics not installed — demo mode active")

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("twilio not installed — pip install twilio")


# ══════════════════════════════════════════════════════════════
# CONFIGURATION  (edit these at the top — nowhere else)
# ══════════════════════════════════════════════════════════════

# ── Models ────────────────────────────────────────────────────
ACTIVITY_MODEL_PATH = "models/checkpoints/best.pt"
WEAPON_MODEL_PATH   = "models/checkpoints/weapon_best.pt"
WEAPON_CONFIDENCE   = 0.45
WEAPON_CLASSES      = {0: "Gun", 1: "Knife", 2: "Person with Mask"}
# NOTE: no WEAPON_EMOJI — output is clean text only

# ── Camera / streaming ────────────────────────────────────────
CAM_WIDTH           = 640
CAM_HEIGHT          = 360
CAM_FPS_TARGET      = 15
STREAM_JPEG_QUALITY = 70
DETECT_FRAME_W      = 640       # resize width before YOLO
MAX_RECONNECT_TRIES = 10
RECONNECT_DELAY_SEC = 2.0

# ── Adaptive detection ────────────────────────────────────────
FRAME_SKIP_NORMAL     = 6       # skip N-1 frames when calm
FRAME_SKIP_ALERT      = 2       # skip N-1 frames when threat found
MIN_DETECT_INTERVAL   = 1.0     # minimum seconds between detections
BLUR_THRESHOLD        = 80.0    # skip frames below this Laplacian var

# ── Video analysis ────────────────────────────────────────────
VIDEO_FRAME_SKIP  = 5
VIDEO_RESIZE_W    = 640

# ── Twilio ────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID   = ""
TWILIO_AUTH_TOKEN    = ""
TWILIO_FROM_NUMBER   = ""
TWILIO_WHATSAPP_FROM = ""       # e.g. "whatsapp:+14155238886"
WHATSAPP_TO_NUMBERS: list[str] = []

# ── System ────────────────────────────────────────────────────
MAX_LIVE_FEEDS         = 5
ALERT_HISTORY_LIMIT    = 500
ALERT_THROTTLE_SECONDS = 30

# ── Threat thresholds ─────────────────────────────────────────
THREAT_THRESHOLDS = {
    "normal":              (0.00, 0.40),
    "possibly_suspicious": (0.40, 0.65),
    "suspicious":          (0.65, 1.01),
}
THREAT_COLORS = {
    "normal":              "GREEN",
    "possibly_suspicious": "YELLOW",
    "suspicious":          "RED",
}

# ── Global state ──────────────────────────────────────────────
live_feeds:           dict[str, "CameraManager"] = {}
alert_history:        list[dict]                 = []
connected_ws_clients: list[WebSocket]            = []
analysis_sessions:    dict[str, dict]            = {}
_phone_streams:       dict[str, dict]            = {}
_last_notif_time:     dict[str, float]           = {}
_feedback_store:      list[dict]                 = []

_alert_config: dict = {
    "levels": {
        "normal":              {"enabled": False, "channels": {"notification": False, "call": False}},
        "possibly_suspicious": {"enabled": True,  "channels": {"notification": True,  "call": False}},
        "suspicious":          {"enabled": True,  "channels": {"notification": True,  "call": True}},
    },
    "whatsapp_numbers": [],
    "phone_numbers":    [],
    "call_enabled":     True,
    "whatsapp_enabled": True,
}

_executor = ThreadPoolExecutor(max_workers=4)


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════
class OmniEyeDetector:
    """
    Thread-safe YOLO wrapper.
    - Resizes frame to DETECT_FRAME_W before inference
    - Returns clean dict with no emojis
    - Checks blur before running weapon model
    """
    _lock = threading.Lock()

    def __init__(self):
        self.activity_model: Optional[object] = None
        self.weapon_model:   Optional[object] = None
        self._load_models()

    @staticmethod
    def _load_yolo(path: str):
        """Load YOLO with PyTorch 2.6 compatibility (3-attempt strategy)."""
        import torch
        import torch.serialization as _ts

        # Attempt 1: safe_globals context manager (PyTorch >= 2.4)
        if hasattr(_ts, "safe_globals"):
            try:
                from ultralytics.nn.tasks import (
                    ClassificationModel, DetectionModel,
                    SegmentationModel, PoseModel,
                )
                from ultralytics.nn.modules.head import Classify, Detect, Segment, Pose
                with _ts.safe_globals([
                    ClassificationModel, DetectionModel,
                    SegmentationModel, PoseModel,
                    Classify, Detect, Segment, Pose,
                ]):
                    return YOLO(path)
            except Exception:
                pass

        # Attempt 2: monkey-patch torch.load → weights_only=False
        _orig = torch.load
        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig(*a, **kw)
        try:
            torch.load = _patched
            return YOLO(path)
        finally:
            torch.load = _orig  # always restore

    def _load_models(self):
        if YOLO_AVAILABLE and Path(ACTIVITY_MODEL_PATH).exists():
            try:
                self.activity_model = self._load_yolo(ACTIVITY_MODEL_PATH)
                logger.success(f"Activity model loaded: {ACTIVITY_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Activity model failed: {e}")
        else:
            logger.warning("Activity model not found — using demo scores")

        if YOLO_AVAILABLE and Path(WEAPON_MODEL_PATH).exists():
            try:
                self.weapon_model = self._load_yolo(WEAPON_MODEL_PATH)
                logger.success(f"Weapon model loaded: {WEAPON_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Weapon model failed: {e}")
        else:
            logger.warning("Weapon model not found")

    def score_to_level(self, score: float) -> str:
        for level, (lo, hi) in THREAT_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "normal"

    @staticmethod
    def is_blurry(frame: np.ndarray) -> bool:
        """Return True if frame is too blurry to analyse reliably."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < BLUR_THRESHOLD

    def _run_activity(self, frame: np.ndarray, sensitivity: float) -> tuple[float, float]:
        if self.activity_model is not None:
            try:
                results = self.activity_model(frame, verbose=False)
                yolo_score = 0.0
                for r in results:
                    if hasattr(r, "probs") and r.probs is not None:
                        probs = r.probs.data.cpu().numpy()
                        yolo_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    elif r.boxes is not None and len(r.boxes):
                        yolo_score = float(max(r.boxes.conf.cpu().numpy().tolist()) or 0.0)
            except Exception as e:
                logger.error(f"Activity inference error: {e}")
                yolo_score = 0.0
        else:
            yolo_score = float(np.random.beta(2, 5))

        fused = min(1.0, yolo_score * (0.5 + sensitivity))
        return round(yolo_score, 4), round(fused, 4)

    def _run_weapon(self, frame: np.ndarray) -> tuple[bool, list[dict]]:
        if self.weapon_model is None:
            return False, []
        try:
            results = self.weapon_model(frame, verbose=False, conf=WEAPON_CONFIDENCE)
            boxes = []
            h, w = frame.shape[:2]
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    conf   = float(box.conf.item())
                    xyxy   = box.xyxy[0].cpu().numpy()
                    boxes.append({
                        "class_id":       cls_id,
                        "class_name":     WEAPON_CLASSES.get(cls_id, f"class_{cls_id}"),
                        "confidence":     round(conf, 3),
                        "confidence_pct": round(conf * 100, 1),
                        "bbox": [
                            round(float(xyxy[0]) / w, 3),
                            round(float(xyxy[1]) / h, 3),
                            round(float(xyxy[2]) / w, 3),
                            round(float(xyxy[3]) / h, 3),
                        ],
                    })
            return bool(boxes), boxes
        except Exception as e:
            logger.error(f"Weapon inference error: {e}")
            return False, []

    def _build_description(
        self,
        fused:   float,
        level:   str,
        weapon:  bool,
        boxes:   list[dict],
        yolo:    float,
    ) -> tuple[str, str, str]:
        """Returns (detection_type, description_long, description_short). No emojis."""
        pct = round(fused * 100)

        if weapon and boxes:
            has_mask = any(b["class_name"] == "Person with Mask" for b in boxes)
            parts    = [f"{b['class_name']} ({b['confidence_pct']}%)" for b in boxes]

            if has_mask and len(boxes) == 1:
                return (
                    "masked_person",
                    f"Masked individual detected — potential threat. Confidence: {pct}%. "
                    f"Immediate attention required.",
                    "MASKED PERSON",
                )
            elif has_mask:
                return (
                    "weapon",
                    f"CRITICAL: Armed masked person. {', '.join(parts)}. "
                    f"Confidence: {pct}%. Immediate action required.",
                    "ARMED + MASKED",
                )
            else:
                return (
                    "weapon",
                    f"Weapon identified: {', '.join(parts)}. "
                    f"Threat score: {pct}%. Do not approach — contact authorities.",
                    "WEAPON DETECTED",
                )

        if level == "suspicious":
            if yolo > 0.85:
                desc = (f"High-confidence suspicious behavior detected ({pct}%). "
                        f"Pattern consistent with robbery or assault. Alert security.")
            else:
                desc = (f"Suspicious activity confirmed ({pct}%). "
                        f"Aggressive body language detected. Monitor closely.")
            return "activity", desc, "HIGH THREAT"

        if level == "possibly_suspicious":
            return (
                "activity",
                f"Potentially suspicious activity ({pct}%). "
                f"Unusual posture or movement detected. Continue monitoring.",
                "SUSPICIOUS",
            )

        return (
            "normal",
            f"Normal activity ({pct}% threat score). No immediate concerns.",
            "NORMAL",
        )

    def detect(self, frame: np.ndarray, sensitivity: float = 0.5,
               skip_blur: bool = True) -> dict:
        """
        Thread-safe detection. Returns plain dict — no emojis anywhere.
        skip_blur: if True, blurry frames return fast with normal result.
        """
        # Resize before inference
        h, w = frame.shape[:2]
        if w > DETECT_FRAME_W:
            frame = cv2.resize(frame, (DETECT_FRAME_W, int(h * DETECT_FRAME_W / w)))

        # Blur check
        if skip_blur and self.is_blurry(frame):
            return {
                "yolo_score": 0.0, "fused_score": 0.0, "confidence_pct": 0.0,
                "alert_level": "normal", "weapon_detected": False,
                "weapon_boxes": [], "weapon_names": [],
                "detection_type": "normal",
                "description": "Frame skipped — insufficient sharpness.",
                "description_short": "NORMAL",
                "skipped_blur": True,
            }

        with self._lock:
            yolo_score, fused_score = self._run_activity(frame, sensitivity)
            weapon_detected, weapon_boxes = self._run_weapon(frame)

        if weapon_detected:
            fused_score  = max(fused_score, 0.92)
            alert_level  = "suspicious"
            weapon_names = [b["class_name"] for b in weapon_boxes]
        else:
            alert_level  = self.score_to_level(fused_score)
            weapon_names = []

        det_type, description, desc_short = self._build_description(
            fused_score, alert_level, weapon_detected, weapon_boxes, yolo_score)

        return {
            "yolo_score":        yolo_score,
            "fused_score":       round(fused_score, 4),
            "confidence_pct":    round(fused_score * 100, 1),
            "alert_level":       alert_level,
            "weapon_detected":   weapon_detected,
            "weapon_boxes":      weapon_boxes,
            "weapon_names":      weapon_names,
            "detection_type":    det_type,
            "description":       description,
            "description_short": desc_short,
            "skipped_blur":      False,
        }


detector = OmniEyeDetector()


# ══════════════════════════════════════════════════════════════
# CAMERA MANAGER — threaded capture + adaptive detection
# ══════════════════════════════════════════════════════════════
class CameraManager:
    """
    Thread-1 (capture_loop):
        Reads frames at CAM_FPS_TARGET. Stores latest raw frame
        and latest JPEG bytes. Never runs YOLO here.

    Thread-2 (detection_loop):
        Adaptive frame-skip + time throttle.
        When threat found: skip fewer frames (FRAME_SKIP_ALERT).
        When calm: skip more frames (FRAME_SKIP_NORMAL).
        Blurry frames are discarded before YOLO call.
    """

    def __init__(self, feed_id: str, source, name: str,
                 sensitivity: float, loop: asyncio.AbstractEventLoop):
        self.feed_id     = feed_id
        self.source      = source
        self.name        = name
        self.sensitivity = sensitivity
        self.loop        = loop
        self.status      = "connecting"

        self._stop_event  = threading.Event()
        self._frame_lock  = threading.Lock()
        self._jpeg_lock   = threading.Lock()
        self._result_lock = threading.Lock()

        self._latest_frame:  Optional[np.ndarray] = None
        self._latest_jpeg:   Optional[bytes]       = None
        self._detection_result: Optional[dict]     = None

        self._capture_thread   = threading.Thread(target=self._capture_loop,   daemon=True)
        self._detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._capture_thread.start()
        self._detection_thread.start()
        logger.info(f"CameraManager started: {feed_id}")

    # ── Public API ────────────────────────────────────────────
    def get_jpeg(self)   -> Optional[bytes]: 
        with self._jpeg_lock:   return self._latest_jpeg
    def get_result(self) -> Optional[dict]:  
        with self._result_lock: return self._detection_result
    def stop(self):
        self._stop_event.set()
        logger.info(f"CameraManager stopping: {self.feed_id}")
    def to_info(self) -> dict:
        return {
            "id": self.feed_id, "name": self.name,
            "source_type": "webcam" if isinstance(self.source, int) else "rtsp",
            "status": self.status, "sensitivity": self.sensitivity,
            "latest_detection": self.get_result() or {},
        }

    # ── Thread 1: capture ─────────────────────────────────────
    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        src = self.source
        cap = (cv2.VideoCapture(src, cv2.CAP_DSHOW)
               if isinstance(src, int)
               else cv2.VideoCapture(src))
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          CAM_FPS_TARGET)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        return cap

    def _capture_loop(self):
        reconnect_count = 0
        cap = None
        pace = 1.0 / CAM_FPS_TARGET

        while not self._stop_event.is_set():
            if cap is None:
                cap = self._open_camera()
                if cap is None:
                    reconnect_count += 1
                    if reconnect_count > MAX_RECONNECT_TRIES:
                        self.status = "error"
                        logger.error(f"Camera {self.feed_id}: max reconnects reached")
                        break
                    time.sleep(RECONNECT_DELAY_SEC)
                    continue
                self.status = "live"
                reconnect_count = 0
                logger.success(f"Camera {self.feed_id}: opened")

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Camera {self.feed_id}: read failed — reconnecting")
                cap.release()
                cap = None
                time.sleep(RECONNECT_DELAY_SEC)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()

            time.sleep(pace)

        if cap:
            cap.release()
        self.status = "stopped"
        logger.info(f"Camera {self.feed_id}: capture thread exited")

    # ── Thread 2: adaptive detection ──────────────────────────
    def _detection_loop(self):
        """
        Hybrid strategy:
          - run if frame_count % frame_skip == 0
          - AND time since last run >= MIN_DETECT_INTERVAL
        After suspicious detection: frame_skip = FRAME_SKIP_ALERT
        After returning normal:     frame_skip = FRAME_SKIP_NORMAL
        """
        frame_count  = 0
        frame_skip   = FRAME_SKIP_NORMAL
        last_run     = 0.0

        while not self._stop_event.is_set():
            time.sleep(1.0 / CAM_FPS_TARGET)   # pace with capture thread
            if self._stop_event.is_set():
                break

            frame_count += 1

            # Frame-skip gate
            if frame_count % frame_skip != 0:
                continue

            # Time-throttle gate
            now = time.monotonic()
            if now - last_run < MIN_DETECT_INTERVAL:
                continue
            last_run = now

            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                continue

            try:
                result = detector.detect(frame, self.sensitivity, skip_blur=True)

                with self._result_lock:
                    self._detection_result = result

                # Adapt frame_skip based on threat level
                if result["alert_level"] == "suspicious":
                    frame_skip = FRAME_SKIP_ALERT
                elif result["alert_level"] == "normal":
                    frame_skip = FRAME_SKIP_NORMAL
                # possibly_suspicious keeps current skip unchanged

                if result["alert_level"] != "normal" and not result.get("skipped_blur"):
                    alert = {
                        "id":        str(uuid.uuid4()),
                        "source":    f"live:{self.feed_id}",
                        "feed_name": self.name,
                        "timestamp": datetime.utcnow().isoformat(),
                        "detection": result,
                        "color":     THREAT_COLORS[result["alert_level"]],
                    }
                    asyncio.run_coroutine_threadsafe(broadcast_alert(alert), self.loop)

            except Exception as e:
                logger.error(f"Detection error on {self.feed_id}: {e}")

        logger.info(f"Camera {self.feed_id}: detection thread exited")


# ══════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app_instance):
    logger.info("OmniEye v6.1 starting — adaptive detection + clean output")
    yield
    for cam in list(live_feeds.values()):
        cam.stop()
    _executor.shutdown(wait=False)
    logger.info("OmniEye v6.1 shutdown complete")


app = FastAPI(
    title="OmniEye Surveillance API v6.1",
    description="Adaptive Detection | MJPEG Stream | Clean Output | WhatsApp + Call Alerts",
    version="6.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════
class LiveFeedConfig(BaseModel):
    source_type: str
    url:         Optional[str] = None
    name:        str           = "Camera"
    sensitivity: float         = 0.5


# ══════════════════════════════════════════════════════════════
# NOTIFICATION SYSTEM
# ══════════════════════════════════════════════════════════════
def _should_call(det_type: str, pct: float, level: str) -> bool:
    """Voice call only for weapon/mask or very high confidence."""
    return det_type in ("weapon", "masked_person") or (
        level == "suspicious" and pct > 90.0)


def _build_whatsapp_msg(alert: dict) -> str:
    """Clean WhatsApp message — no emojis in alert body (WhatsApp emojis kept for readability)."""
    det   = alert.get("detection", {})
    dtype = det.get("detection_type", "activity")
    level = det.get("alert_level", "normal")
    pct   = round(det.get("confidence_pct", 0))
    desc  = det.get("description", "Suspicious activity detected")
    src   = alert.get("feed_name", "OmniEye")
    ts    = alert.get("timestamp", datetime.utcnow().isoformat())[:19].replace("T", " ")
    wnames = det.get("weapon_names", [])

    # Header label (text-only for professionalism)
    if dtype == "weapon":
        header = "[WEAPON DETECTED] OmniEye Alert"
    elif dtype == "masked_person":
        header = "[MASKED PERSON] OmniEye Alert"
    elif level == "suspicious":
        header = "[HIGH THREAT] OmniEye Alert"
    else:
        header = "[SUSPICIOUS] OmniEye Alert"

    lines = [
        f"*{header}*", "",
        f"Source: {src}",
        f"Confidence: {pct}%",
        f"Time: {ts}", "",
        f"Details: {desc}",
    ]
    if wnames:
        lines.append(f"Weapons: {', '.join(wnames)}")
    lines += ["", "_OmniEye AI Surveillance System_"]
    return "\n".join(lines)


async def _send_whatsapp(alert: dict, to_numbers: list[str]):
    if not (TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_WHATSAPP_FROM and to_numbers):
        return
    # Check global toggle
    if not _alert_config.get("whatsapp_enabled", True):
        logger.debug("WhatsApp disabled globally — skipped")
        return

    msg = _build_whatsapp_msg(alert)

    def _send():
        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for to in to_numbers:
            to_fmt = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
            try:
                m = client.messages.create(body=msg, from_=TWILIO_WHATSAPP_FROM, to=to_fmt)
                logger.info(f"WhatsApp sent to {to_fmt} — SID: {m.sid}")
            except Exception as e:
                logger.error(f"WhatsApp failed {to_fmt}: {e}")

    await asyncio.get_event_loop().run_in_executor(_executor, _send)


async def _twilio_call(to: str, level: str, pct: float, weapons: list, src: str):
    if not (TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_FROM_NUMBER):
        return
    # Check global toggle
    if not _alert_config.get("call_enabled", True):
        logger.debug("Voice calls disabled globally — skipped")
        return

    if weapons:
        say = (f"OmniEye Alert. Weapon detected: {' and '.join(weapons)}. "
               f"Confidence {int(pct)} percent. Source: {src}. Immediate action required.")
    else:
        say = (f"OmniEye Alert. High threat activity at {src}. "
               f"Score {int(pct)} percent. Check surveillance immediately.")

    twiml = (f"<Response>"
             f"<Say voice='alice' language='en-IN'>{say}</Say>"
             f"<Pause length='2'/>"
             f"<Say voice='alice' language='en-IN'>{say}</Say>"
             f"</Response>")

    def _call():
        try:
            c    = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            call = c.calls.create(twiml=twiml, from_=TWILIO_FROM_NUMBER, to=to)
            logger.info(f"Voice call initiated to {to} — SID: {call.sid}")
        except Exception as e:
            logger.error(f"Call failed {to}: {e}")

    await asyncio.get_event_loop().run_in_executor(_executor, _call)


async def _send_notifications(alert: dict):
    det   = alert.get("detection", {})
    level = det.get("alert_level", "normal")
    cfg   = _alert_config.get("levels", {}).get(level, {})

    if not cfg.get("enabled"):
        return

    now = time.time()
    if now - _last_notif_time.get(level, 0) < ALERT_THROTTLE_SECONDS:
        return
    _last_notif_time[level] = now

    chans   = cfg.get("channels", {})
    pct     = det.get("confidence_pct", 0.0)
    dtype   = det.get("detection_type", "activity")
    weapons = det.get("weapon_names", [])
    src     = alert.get("feed_name", "OmniEye")

    wa_nums = _alert_config.get("whatsapp_numbers") or WHATSAPP_TO_NUMBERS
    phones  = _alert_config.get("phone_numbers") or []
    tasks   = []

    if chans.get("notification") and wa_nums:
        tasks.append(_send_whatsapp(alert, wa_nums))
    if chans.get("call") and phones and _should_call(dtype, float(pct), level):
        for ph in phones:
            tasks.append(_twilio_call(ph, level, float(pct), weapons, src))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Notification error: {r}")


# ══════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════
async def broadcast_alert(alert: dict):
    alert_history.append(alert)
    if len(alert_history) > ALERT_HISTORY_LIMIT:
        alert_history.pop(0)
    asyncio.create_task(_send_notifications(alert))
    dead = []
    for ws in connected_ws_clients:
        try:
            await ws.send_json(alert)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            connected_ws_clients.remove(ws)
        except ValueError:
            pass


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    connected_ws_clients.append(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=25)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text("ping")
    except WebSocketDisconnect:
        pass
    finally:
        try:
            connected_ws_clients.remove(websocket)
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    wa_ok   = bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_WHATSAPP_FROM)
    call_ok = bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_FROM_NUMBER)
    return {
        "status": "operational", "version": "6.1.0",
        "model_loaded":        detector.activity_model is not None,
        "weapon_model_loaded": detector.weapon_model is not None,
        "live_feeds":          len(live_feeds),
        "phone_streams":       len(_phone_streams),
        "ws_clients":          len(connected_ws_clients),
        "timestamp":           datetime.utcnow().isoformat(),
        "detection": {
            "frame_skip_normal": FRAME_SKIP_NORMAL,
            "frame_skip_alert":  FRAME_SKIP_ALERT,
            "min_interval_sec":  MIN_DETECT_INTERVAL,
            "blur_threshold":    BLUR_THRESHOLD,
        },
        "notifications": {
            "whatsapp":    {"configured": wa_ok,
                            "enabled":    _alert_config.get("whatsapp_enabled", True)},
            "twilio_call": {"configured": call_ok,
                            "enabled":    _alert_config.get("call_enabled", True)},
        },
    }


# ── Image detection ───────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(
    file:                 UploadFile = File(...),
    sensitivity:          float      = Form(0.5),
    confidence_threshold: float      = Form(0.4),
):
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    result = await asyncio.get_event_loop().run_in_executor(
        _executor, lambda: detector.detect(frame, sensitivity, skip_blur=False))

    alert = {
        "id":        str(uuid.uuid4()),
        "source":    "image_upload",
        "feed_name": "Image Upload",
        "timestamp": datetime.utcnow().isoformat(),
        "detection": result,
        "color":     THREAT_COLORS[result["alert_level"]],
    }
    if result["alert_level"] != "normal" and result["fused_score"] >= confidence_threshold:
        await broadcast_alert(alert)
    else:
        alert_history.append(alert)
        if len(alert_history) > ALERT_HISTORY_LIMIT:
            alert_history.pop(0)
    return alert


# ── Video analysis ────────────────────────────────────────────
@app.post("/detect/video/upload")
async def detect_video_upload(
    background_tasks:     BackgroundTasks,
    file:                 UploadFile = File(...),
    group_seconds:        float      = Form(2.0),
    frame_skip:           int        = Form(VIDEO_FRAME_SKIP),
    confidence_threshold: float      = Form(0.4),
    sensitivity:          float      = Form(0.5),
    start_time:           float      = Form(0.0),
    end_time:             Optional[float] = Form(None),
):
    session_id  = str(uuid.uuid4())
    video_bytes = await file.read()
    analysis_sessions[session_id] = {
        "status": "queued", "progress": 0, "events": [], "stats": {},
        "filename": file.filename, "created_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(
        _run_video_analysis_async, session_id, video_bytes,
        group_seconds, frame_skip, confidence_threshold,
        sensitivity, start_time, end_time,
    )
    return {"session_id": session_id, "status": "queued"}


async def _run_video_analysis_async(session_id, video_bytes, group_seconds,
                                     frame_skip, confidence_threshold,
                                     sensitivity, start_time, end_time):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        _executor, _analyse_video_blocking,
        session_id, video_bytes, group_seconds,
        frame_skip, confidence_threshold, sensitivity, start_time, end_time,
    )


def _analyse_video_blocking(session_id, video_bytes, group_seconds,
                              frame_skip, confidence_threshold,
                              sensitivity, start_time, end_time):
    sess = analysis_sessions[session_id]
    sess["status"] = "processing"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            sess["status"] = "error"
            sess["error"]  = "Could not open video"
            return

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame  = int(start_time * fps)
        end_frame    = int(end_time * fps) if end_time else total_frames
        fpg          = max(1, int(group_seconds * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        groups, stats, weapon_total = {}, defaultdict(int), 0
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % frame_skip == 0:
                h, w = frame.shape[:2]
                if w > VIDEO_RESIZE_W:
                    frame = cv2.resize(frame, (VIDEO_RESIZE_W, int(h * VIDEO_RESIZE_W / w)))
                result    = detector.detect(frame, sensitivity, skip_blur=True)
                group_idx = (frame_idx - start_frame) // fpg
                groups.setdefault(group_idx, []).append(result)
                stats[result["alert_level"]] += 1
                if result["weapon_detected"]:
                    weapon_total += 1

            frame_idx += 1
            sess["progress"] = round(
                (frame_idx - start_frame) / max(end_frame - start_frame, 1) * 100)

        cap.release()

        events = []
        for gi in sorted(groups.keys()):
            best = max(groups[gi], key=lambda r: r["fused_score"])
            ts   = start_time + gi * group_seconds
            if best["fused_score"] >= confidence_threshold:
                events.append({
                    "timestamp":         round(ts, 2),
                    "timestamp_fmt":     _fmt_time(ts),
                    "frame":             start_frame + gi * fpg,
                    "alert_level":       best["alert_level"],
                    "color":             THREAT_COLORS[best["alert_level"]],
                    "fused_score":       best["fused_score"],
                    "yolo_score":        best["yolo_score"],
                    "confidence_pct":    round(best["fused_score"] * 100, 1),
                    "weapon_detected":   best["weapon_detected"],
                    "weapon_names":      best["weapon_names"],
                    "detection_type":    best["detection_type"],
                    "description":       best["description"],
                    "description_short": best["description_short"],
                })

        total = sum(stats.values()) or 1
        sess.update({
            "status": "done", "progress": 100, "events": events,
            "stats": {
                "total_frames_analysed":     frame_idx - start_frame,
                "total_groups":              len(events),
                "normal_count":              stats["normal"],
                "possibly_suspicious_count": stats["possibly_suspicious"],
                "suspicious_count":          stats["suspicious"],
                "weapon_detections":         weapon_total,
                "threat_percentage":         round(
                    (stats["suspicious"] + stats["possibly_suspicious"]) / total * 100, 1),
                "duration_seconds":          round((end_frame - start_frame) / fps, 2),
            },
        })
        logger.success(f"Video done: {session_id} — {len(events)} events")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/detect/video/status/{session_id}")
def video_status(session_id: str):
    s = analysis_sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return {"session_id": session_id, "status": s["status"], "progress": s["progress"]}


@app.get("/detect/video/results/{session_id}")
def video_results(session_id: str):
    s = analysis_sessions.get(session_id)
    if not s:
        raise HTTPException(404, "Not found")
    if s["status"] != "done":
        raise HTTPException(400, f"Not complete: {s['status']}")
    return s


# ── Live feeds ────────────────────────────────────────────────
@app.post("/feeds/add")
async def add_feed(config: LiveFeedConfig):
    if len(live_feeds) >= MAX_LIVE_FEEDS:
        raise HTTPException(400, f"Max {MAX_LIVE_FEEDS} feeds")
    feed_id = str(uuid.uuid4())[:8]
    src = 0 if config.source_type == "webcam" else config.url
    if src is None:
        raise HTTPException(400, "URL required for non-webcam source")
    loop = asyncio.get_event_loop()
    cam  = CameraManager(feed_id, src, config.name, config.sensitivity, loop)
    live_feeds[feed_id] = cam
    return {"feed_id": feed_id, "status": "connecting"}


@app.get("/feeds")
def list_feeds():
    return {fid: cam.to_info() for fid, cam in live_feeds.items()}


@app.get("/feeds/{feed_id}")
def get_feed(feed_id: str):
    cam = live_feeds.get(feed_id)
    if not cam:
        raise HTTPException(404, "Feed not found")
    return cam.to_info()


@app.delete("/feeds/{feed_id}")
def remove_feed(feed_id: str):
    cam = live_feeds.pop(feed_id, None)
    if not cam:
        raise HTTPException(404, "Feed not found")
    cam.stop()
    return {"status": "removed"}


@app.get("/feeds/{feed_id}/snapshot")
def feed_snapshot(feed_id: str):
    cam = live_feeds.get(feed_id)
    if not cam:
        raise HTTPException(404, "Feed not found")
    jpeg = cam.get_jpeg()
    if not jpeg:
        raise HTTPException(503, "No frame yet")
    return {
        "feed_id":          feed_id,
        "frame_b64":        base64.b64encode(jpeg).decode(),
        "latest_detection": cam.get_result(),
    }


# ── MJPEG streaming ───────────────────────────────────────────
def _mjpeg_generator(feed_id: str) -> Generator[bytes, None, None]:
    boundary   = b"frame"
    target_fps = 1.0 / CAM_FPS_TARGET
    last_jpeg: Optional[bytes] = None

    while True:
        cam = live_feeds.get(feed_id)
        if cam is None:
            break
        jpeg = cam.get_jpeg()
        if jpeg and jpeg is not last_jpeg:
            last_jpeg = jpeg
            yield (
                b"--" + boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                b"\r\n" + jpeg + b"\r\n"
            )
        time.sleep(target_fps)


@app.get("/stream/{feed_id}")
def stream_feed(feed_id: str):
    if feed_id not in live_feeds:
        raise HTTPException(404, "Feed not found")
    return StreamingResponse(
        _mjpeg_generator(feed_id),
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/stream/webcam/direct")
def stream_webcam_direct():
    """Direct webcam MJPEG — no feed registration required."""
    def _gen():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
                if ok:
                    jpeg = buf.tobytes()
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpeg)).encode() +
                           b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(1.0 / CAM_FPS_TARGET)
        finally:
            cap.release()
    return StreamingResponse(
        _gen(),
        media_type="multipart/x-mixed-replace;boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


# ── Phone as camera ───────────────────────────────────────────
@app.get("/mobile/camera/page")
def mobile_camera_page():
    html = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>OmniEye Mobile Camera</title>
<style>
* { margin:0; padding:0; box-sizing:border-box }
body { background:#000; color:#0f0; font-family:monospace }
#v { width:100vw; height:100vh; object-fit:cover }
#status { position:fixed; top:10px; left:50%; transform:translateX(-50%);
  background:rgba(0,0,0,.85); color:#0f0; padding:8px 18px;
  border:1px solid #0f0; border-radius:4px; font-size:13px; z-index:100; }
#btn { position:fixed; bottom:30px; left:50%; transform:translateX(-50%);
  background:#0f0; color:#000; padding:14px 32px; border:none;
  border-radius:8px; font-size:16px; font-weight:bold; cursor:pointer; }
</style></head><body>
<video id="v" autoplay muted playsinline></video>
<div id="status">Tap START to stream</div>
<button id="btn" onclick="startStream()">START</button>
<script>
const WS = window.location.origin.replace('http','ws') + '/ws/mobile/camera';
let ws, interval, camId = 'phone_' + Math.random().toString(36).slice(2,8);
async function startStream() {
  document.getElementById('btn').style.display='none';
  document.getElementById('status').textContent='Requesting camera...';
  try {
    const stream = await navigator.mediaDevices.getUserMedia(
      {video:{facingMode:'environment',width:640,height:480},audio:false});
    document.getElementById('v').srcObject = stream;
    ws = new WebSocket(WS);
    ws.onopen = () => {
      ws.send(JSON.stringify({type:'register',camera_id:camId,name:'Phone Camera'}));
      document.getElementById('status').textContent='LIVE — ID: '+camId;
      const canvas=document.createElement('canvas');
      canvas.width=640; canvas.height=480;
      const ctx=canvas.getContext('2d'), vid=document.getElementById('v');
      interval=setInterval(()=>{
        if(ws.readyState!==1) return;
        ctx.drawImage(vid,0,0,640,480);
        canvas.toBlob(blob=>{
          const r=new FileReader();
          r.onload=()=>ws.send(JSON.stringify(
            {type:'frame',camera_id:camId,frame:r.result.split(',')[1]}));
          r.readAsDataURL(blob);
        },'image/jpeg',0.6);
      },1000);
    };
    ws.onclose=()=>{clearInterval(interval);setTimeout(startStream,3000);};
  } catch(e){
    document.getElementById('status').textContent='Error: '+e.message;
    document.getElementById('btn').style.display='block';
  }
}
</script></body></html>"""
    return HTMLResponse(html)


@app.websocket("/ws/mobile/camera")
async def ws_mobile_camera(websocket: WebSocket):
    await websocket.accept()
    camera_id = None
    result    = None
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=35)
                msg  = json.loads(data)
                if msg.get("type") == "register":
                    camera_id = msg.get("camera_id", "unknown")
                    _phone_streams[camera_id] = {
                        "camera_id": camera_id, "name": msg.get("name", "Phone"),
                        "status": "live", "connected_at": datetime.utcnow().isoformat(),
                        "latest_detection": None,
                    }
                    await websocket.send_text(json.dumps({"status": "registered"}))
                elif msg.get("type") == "frame" and camera_id:
                    b64 = msg.get("frame", "")
                    if b64:
                        arr   = np.frombuffer(base64.b64decode(b64), np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            result = await asyncio.get_event_loop().run_in_executor(
                                _executor, lambda f=frame: detector.detect(f, 0.5, skip_blur=True))
                            _phone_streams[camera_id]["latest_detection"] = result
                            if result["alert_level"] != "normal":
                                alert = {
                                    "id": str(uuid.uuid4()), "source": f"phone:{camera_id}",
                                    "feed_name": _phone_streams[camera_id]["name"],
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "detection": result,
                                    "color": THREAT_COLORS[result["alert_level"]],
                                }
                                await broadcast_alert(alert)
                    await websocket.send_text(json.dumps({
                        "status": "ok",
                        "alert_level": result["alert_level"] if result else "normal",
                    }))
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"status": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        if camera_id:
            _phone_streams.pop(camera_id, None)


@app.get("/mobile/cameras")
def list_phone_cameras():
    return _phone_streams


# ── Alerts ────────────────────────────────────────────────────
@app.get("/alerts/history")
def get_alert_history(limit: int = 100):
    return {"alerts": alert_history[-limit:], "total": len(alert_history)}


@app.post("/alerts/config")
async def set_alert_config(payload: dict):
    global _alert_config, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN
    global TWILIO_FROM_NUMBER, TWILIO_WHATSAPP_FROM
    _alert_config = payload
    for key, var in [("twilio_sid",           "TWILIO_ACCOUNT_SID"),
                     ("twilio_token",         "TWILIO_AUTH_TOKEN"),
                     ("twilio_from",          "TWILIO_FROM_NUMBER"),
                     ("twilio_whatsapp_from", "TWILIO_WHATSAPP_FROM")]:
        if key in payload:
            globals()[var] = payload[key]
    return {"status": "saved"}


@app.get("/alerts/config")
def get_alert_config():
    return _alert_config


@app.post("/notify/test")
async def notify_test():
    test_alert = {
        "id": "test-" + str(uuid.uuid4())[:8], "source": "test",
        "feed_name": "TEST ALERT", "timestamp": datetime.utcnow().isoformat(),
        "detection": {
            "alert_level": "suspicious", "fused_score": 0.92,
            "confidence_pct": 92.0, "weapon_detected": False, "weapon_names": [],
            "detection_type": "activity",
            "description": "Test alert from OmniEye v6.1. System is operational.",
            "description_short": "HIGH THREAT",
        }, "color": "RED",
    }
    wa_nums = _alert_config.get("whatsapp_numbers") or WHATSAPP_TO_NUMBERS
    phones  = _alert_config.get("phone_numbers") or []
    tasks   = []
    if wa_nums and TWILIO_WHATSAPP_FROM:
        tasks.append(_send_whatsapp(test_alert, wa_nums))
    if phones and TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_FROM_NUMBER:
        for ph in phones:
            tasks.append(_twilio_call(ph, "suspicious", 92.0, [], "TEST"))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await broadcast_alert(test_alert)
    return {"status": "test sent",
            "channels": {"whatsapp": bool(wa_nums), "call": bool(phones)}}


@app.get("/notify/status")
def notify_status():
    wa_ok   = bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_WHATSAPP_FROM)
    call_ok = bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID and TWILIO_FROM_NUMBER)
    return {
        "whatsapp":    {"configured": wa_ok,
                        "enabled": _alert_config.get("whatsapp_enabled", True)},
        "twilio_call": {"configured": call_ok,
                        "enabled": _alert_config.get("call_enabled", True)},
    }


# ── Export ────────────────────────────────────────────────────
@app.get("/export/csv/{session_id}")
def export_csv(session_id: str):
    s = analysis_sessions.get(session_id)
    if not s or s["status"] != "done":
        raise HTTPException(404, "Results not ready")
    out    = io.StringIO()
    fields = ["timestamp", "timestamp_fmt", "frame", "alert_level", "color",
              "fused_score", "yolo_score", "confidence_pct",
              "weapon_detected", "weapon_names", "detection_type", "description"]
    w = csv.DictWriter(out, fieldnames=fields)
    w.writeheader()
    for ev in s["events"]:
        row = {k: ev.get(k, "") for k in fields}
        if isinstance(row.get("weapon_names"), list):
            row["weapon_names"] = ", ".join(row["weapon_names"])
        w.writerow(row)
    out.seek(0)
    fname = f"OmniEye_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(iter([out.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/export/pdf/{session_id}")
def export_pdf(session_id: str):
    if not PDF_AVAILABLE:
        raise HTTPException(503, "pip install reportlab")
    s = analysis_sessions.get(session_id)
    if not s or s["status"] != "done":
        raise HTTPException(404, "Not ready")
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter)
    styl = getSampleStyleSheet()
    els  = [Paragraph("OmniEye AI — Threat Analysis Report", styl["Title"]),
            Spacer(1, 12),
            Paragraph(f"Session: {session_id}", styl["Normal"]),
            Paragraph(f"File: {s.get('filename','N/A')}", styl["Normal"]),
            Paragraph(f"Generated: {datetime.utcnow().isoformat()}", styl["Normal"]),
            Spacer(1, 20)]
    sd = [["Metric", "Value"]] + [[k.replace("_"," ").title(), str(v)]
                                    for k, v in s.get("stats", {}).items()]
    t = Table(sd, colWidths=[250, 150])
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), rl_colors.grey),
                            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.whitesmoke),
                            ("GRID",       (0,0), (-1,-1), .5, rl_colors.black)]))
    els += [t, Spacer(1, 20)]
    ed = [["Time", "Level", "Score", "Type", "Description"]]
    for ev in s["events"][:200]:
        ed.append([ev["timestamp_fmt"], ev["alert_level"].upper(),
                   f"{ev['confidence_pct']}%",
                   ev.get("detection_type", "").upper(),
                   ev.get("description", "")[:80]])
    if len(ed) > 1:
        et = Table(ed, colWidths=[55, 75, 45, 75, 230])
        et.setStyle(TableStyle([("BACKGROUND",    (0,0), (-1,0), rl_colors.darkblue),
                                 ("TEXTCOLOR",     (0,0), (-1,0), rl_colors.white),
                                 ("FONTSIZE",      (0,0), (-1,-1), 7),
                                 ("ROWBACKGROUNDS",(0,1), (-1,-1),
                                  [rl_colors.white, rl_colors.lightgrey]),
                                 ("GRID",          (0,0), (-1,-1), .3, rl_colors.black)]))
        els.append(et)
    doc.build(els)
    buf.seek(0)
    fname = f"OmniEye_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ── Dashboard ─────────────────────────────────────────────────
@app.get("/dashboard/stats")
def dashboard_stats():
    recent  = alert_history[-100:]
    counts  = defaultdict(int)
    weapons = 0
    for a in recent:
        det = a.get("detection") or {}
        counts[det.get("alert_level", "normal")] += 1
        if det.get("weapon_detected"):
            weapons += 1
    return {
        "live_feeds": len(live_feeds), "phone_cameras": len(_phone_streams),
        "total_alerts": len(alert_history), "recent_100": dict(counts),
        "weapon_detections": weapons, "ws_clients": len(connected_ws_clients),
        "active_sessions": len([s for s in analysis_sessions.values()
                                 if s["status"] == "processing"]),
    }


# ── ONVIF discovery ───────────────────────────────────────────
@app.get("/discover/onvif")
async def discover_onvif():
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery as WSD
        wsd = WSD()
        wsd.start()
        found = []
        for svc in wsd.searchServices():
            for sc in svc.getScopes():
                if "onvif" in str(sc).lower():
                    found.append({"address": str(svc.getXAddrs()),
                                  "scopes": [str(s) for s in svc.getScopes()]})
        wsd.stop()
        return {"status": "ok", "cameras": found}
    except ImportError:
        return {"status": "unavailable", "message": "pip install wsdiscovery", "cameras": []}
    except Exception as e:
        return {"status": "error", "message": str(e), "cameras": []}


# ── Helpers ───────────────────────────────────────────────────
def _fmt_time(s: float) -> str:
    h   = int(s // 3600)
    m   = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════╗")
    logger.info("║   OmniEye Backend v6.1                            ║")
    logger.info("║   Adaptive Detection | Blur Filter | Clean Output ║")
    logger.info("╚══════════════════════════════════════════════════╝")
    logger.info(f"Activity model : {ACTIVITY_MODEL_PATH}")
    logger.info(f"Weapon model   : {WEAPON_MODEL_PATH}")
    logger.info(f"Detection      : skip={FRAME_SKIP_NORMAL} normal / {FRAME_SKIP_ALERT} alert, "
                f"min_interval={MIN_DETECT_INTERVAL}s, blur_thresh={BLUR_THRESHOLD}")
    logger.info(f"WhatsApp       : {'OK' if TWILIO_WHATSAPP_FROM else 'NOT SET'}")
    logger.info(f"Twilio call    : {'OK' if TWILIO_FROM_NUMBER else 'NOT SET'}")
    logger.info(f"MJPEG stream   : http://YOUR_IP:8000/stream/FEED_ID")
    logger.info(f"Direct webcam  : http://YOUR_IP:8000/stream/webcam/direct")
    module = os.path.splitext(os.path.basename(__file__))[0]
    try:
        uvicorn.run(f"{module}:app", host="0.0.0.0", port=8000,
                    reload=False, log_level="info", workers=1)
    except KeyboardInterrupt:
        pass  # clean Ctrl+C — no traceback
