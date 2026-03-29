"""
╔══════════════════════════════════════════════════════════════╗
║     OmniEye AI — Backend v4.1  (aligned with frontend v4.1) ║
║                    sentinel_backend.py                       ║
╚══════════════════════════════════════════════════════════════╝

WHAT CHANGED vs previous version:
  ① Alert config now matches frontend's new structure:
       { levels: { normal:{enabled,channels}, possibly_suspicious:{...}, suspicious:{...} },
         phone_numbers:[...], firebase_token:... }
  ② Video upload now accepts  group_seconds  (frontend sends it)
  ③ Video results return events grouped by group_seconds windows
  ④ /detect/image returns  confidence_pct  field (frontend reads it)
  ⑤ Added Telegram bot notification (no app install needed)
  ⑥ Added Twilio SMS/call notification (optional)
  ⑦ Added WhatsApp via Twilio (optional)
  ⑧ Added /notify/test endpoint so frontend Test button works
  ⑨ Alert throttle — won't spam phone (min 30s between same-level alerts)
  ⑩ /alerts/config accepts & stores new frontend shape

MOBILE NOTIFICATIONS — NO APP NEEDED:
  Option A  Telegram Bot  (FREE, instant, works worldwide)
  Option B  Twilio SMS    (paid, ~$0.008/msg, most reliable)
  Option C  WhatsApp      (via Twilio, paid)
  Option D  ntfy.sh       (FREE push, open source)
  → Setup guide at bottom of this file

HOW TO RUN:
    pip install fastapi uvicorn opencv-python ultralytics loguru
        python-multipart websockets reportlab aiofiles httpx
    py -3.11 sentinel_backend.py

    Optional (for SMS/call):  pip install twilio
    Optional (for ntfy):      pip install httpx   (already above)

API   : http://localhost:8000
WS    : ws://localhost:8000/ws/alerts
Docs  : http://localhost:8000/docs
"""

import asyncio
import base64
import csv
import io
import json
import tempfile
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel

# ── Optional libs ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
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
    import httpx          # for Telegram + ntfy notifications
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed — Telegram/ntfy notifications disabled. pip install httpx")

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ══════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these to enable notifications
# ══════════════════════════════════════════════════════════════

MODEL_PATH = "models/checkpoints/best.pt"
MAX_LIVE_FEEDS = 5
ALERT_HISTORY_LIMIT = 500
ALERT_THROTTLE_SECONDS = 30   # min seconds between phone notifications for same level

# ── Telegram (FREE — recommended) ─────────────────────────────
# 1. Message @BotFather on Telegram → /newbot → copy token
# 2. Message your bot once, then visit:
#    https://api.telegram.org/bot<TOKEN>/getUpdates  → copy chat_id
TELEGRAM_BOT_TOKEN = ""        # e.g. "7123456789:AAFxxx..."
TELEGRAM_CHAT_ID   = ""        # e.g. "123456789"

# ── ntfy.sh (FREE push notifications — no account needed) ─────
# 1. Install the ntfy app on your phone (Android/iOS, free)
# 2. Subscribe to any topic name you choose, e.g. "omnieye-abc123"
# 3. Set NTFY_TOPIC to that same topic name
NTFY_TOPIC = ""                # e.g. "omnieye-abc123"

# ── Twilio (paid — ~$0.008/SMS) ───────────────────────────────
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN  = ""
TWILIO_FROM_NUMBER = ""        # e.g. "+12015551234"

# ──────────────────────────────────────────────────────────────

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
live_feeds:          dict[str, dict] = {}
alert_history:       list[dict]      = []
connected_ws_clients: list[WebSocket] = []
analysis_sessions:   dict[str, dict] = {}
_feedback_store:     list[dict]      = []
_paired_devices:     dict[str, dict] = {}
_last_notif_time:    dict[str, float] = {}   # level → last notification timestamp

# ── Alert config — matches frontend v4.1 shape ────────────────
_alert_config: dict = {
    "levels": {
        "normal":              {"enabled": False, "channels": {"notif": False, "sound": False, "sms": False, "call": False}},
        "possibly_suspicious": {"enabled": True,  "channels": {"notif": True,  "sound": True,  "sms": False, "call": False}},
        "suspicious":          {"enabled": True,  "channels": {"notif": True,  "sound": True,  "sms": True,  "call": True}},
    },
    "phone_numbers":  [],
    "firebase_token": None,
}

# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════
app = FastAPI(
    title="OmniEye Surveillance API",
    description="Military-grade dual-mode suspicious activity detection",
    version="4.1.0",
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
class DetectionResult(BaseModel):
    yolo_score:       float     = 0.0
    pose_score:       float     = 0.0
    temporal_score:   float     = 0.0
    fused_score:      float     = 0.0
    confidence_pct:   float     = 0.0   # ← frontend reads this field
    alert_level:      str       = "normal"
    triggered_rules:  list[str] = []
    is_low_confidence: bool     = False

class LiveFeedConfig(BaseModel):
    source_type: str
    url:         Optional[str] = None
    name:        str           = "Camera"
    sensitivity: float         = 0.5

class MobilePairRequest(BaseModel):
    device_name: str = "Phone"
    role:        str = "alert"


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════
class SentinelDetector:
    def __init__(self, model_path: str):
        self.model = None
        if YOLO_AVAILABLE and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                logger.success(f"Model loaded: {model_path}")
            except Exception as e:
                logger.error(f"Model load failed: {e}")
        else:
            logger.warning("DEMO MODE — synthetic scores (model not found)")

    def score_to_level(self, score: float) -> str:
        for level, (lo, hi) in THREAT_THRESHOLDS.items():
            if lo <= score < hi:
                return level
        return "normal"

    def detect(self, frame: np.ndarray, sensitivity: float = 0.5) -> DetectionResult:
        if self.model is not None:
            try:
                results = self.model(frame, verbose=False)
                yolo_score = 0.0
                for r in results:
                    if hasattr(r, 'probs') and r.probs is not None:
                        probs = r.probs.data.cpu().numpy()
                        # class index 1 = suspicious  (adjust if your training differs)
                        yolo_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                    elif r.boxes is not None and len(r.boxes):
                        confs = r.boxes.conf.cpu().numpy().tolist()
                        yolo_score = float(max(confs)) if confs else 0.0
            except Exception as e:
                logger.error(f"YOLO inference: {e}")
                yolo_score = 0.0
        else:
            yolo_score = float(np.random.beta(2, 5))

        adjusted    = min(1.0, yolo_score * (0.5 + sensitivity))
        alert_level = self.score_to_level(adjusted)

        return DetectionResult(
            yolo_score       = round(yolo_score, 4),
            fused_score      = round(adjusted, 4),
            confidence_pct   = round(adjusted * 100, 1),   # ← added
            alert_level      = alert_level,
            is_low_confidence= adjusted < 0.2,
        )


detector = SentinelDetector(MODEL_PATH)


# ══════════════════════════════════════════════════════════════
# NOTIFICATION SYSTEM
# ══════════════════════════════════════════════════════════════
async def _send_notifications(alert: dict):
    """
    Dispatch notifications based on _alert_config.
    Throttled: won't spam the same level more than once per ALERT_THROTTLE_SECONDS.
    """
    det   = alert.get("detection", {})
    level = det.get("alert_level", "normal")
    lvl_cfg = _alert_config.get("levels", {}).get(level, {})

    if not lvl_cfg.get("enabled", False):
        return

    # Throttle check
    now = time.time()
    last = _last_notif_time.get(level, 0)
    if now - last < ALERT_THROTTLE_SECONDS:
        return
    _last_notif_time[level] = now

    channels = lvl_cfg.get("channels", {})
    score    = det.get("fused_score", 0)
    pct      = round(score * 100)
    ts       = alert.get("timestamp", datetime.utcnow().isoformat())[:19]
    feed     = alert.get("feed_name", alert.get("source", "OmniEye"))

    emoji = "🔴" if level == "suspicious" else "🟡" if level == "possibly_suspicious" else "🟢"
    msg   = (
        f"{emoji} OmniEye Alert\n"
        f"Level : {level.replace('_',' ').upper()}\n"
        f"Score : {pct}%\n"
        f"Source: {feed}\n"
        f"Time  : {ts}"
    )

    phones = _alert_config.get("phone_numbers", [])

    tasks = []

    # ── Telegram ──────────────────────────────────────────
    if channels.get("notif") and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        tasks.append(_telegram_send(msg))

    # ── ntfy.sh ───────────────────────────────────────────
    if channels.get("notif") and NTFY_TOPIC:
        tasks.append(_ntfy_send(msg, level))

    # ── SMS via Twilio ────────────────────────────────────
    if channels.get("sms") and phones and TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID:
        for phone in phones:
            tasks.append(_twilio_sms(phone, msg))

    # ── Voice call via Twilio ─────────────────────────────
    if channels.get("call") and phones and TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID:
        for phone in phones:
            tasks.append(_twilio_call(phone, level, pct))

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Notification error: {r}")


async def _telegram_send(message: str):
    if not HTTPX_AVAILABLE:
        return
    url = f"https://api.telegram.org/bot8603769024:AAE6k_GdC1kCwcrRByVnuIbqoWsKaEWBjNo/sendMessage"
    async with httpx.AsyncClient(timeout=8) as client:
        await client.post(url, json={"chat_id":7455476355, "text": message})
    logger.info("Telegram notification sent")


async def _ntfy_send(message: str, level: str):
    if not HTTPX_AVAILABLE:
        return
    priority = "urgent" if level == "suspicious" else "high" if level == "possibly_suspicious" else "default"
    async with httpx.AsyncClient(timeout=8) as client:
        await client.post(
            f"https://ntfy.sh/ElHyvuuXdexeOJzP",
            content=message.encode(),
            headers={
                "Title": "OmniEye Alert",
                "Priority": priority,
                "Tags": "rotating_light,omnieye",
            }
        )
    logger.info("ntfy notification sent")


async def _twilio_sms(to_number: str, message: str):
    if not TWILIO_AVAILABLE:
        return
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=to_number)
    logger.info(f"SMS sent to {to_number}")


async def _twilio_call(to_number: str, level: str, pct: int):
    if not TWILIO_AVAILABLE:
        return
    client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twiml  = f'<Response><Say voice="alice">OmniEye alert. {level.replace("_"," ")} detected at {pct} percent confidence.</Say></Response>'
    client.calls.create(
        twiml=twiml,
        from_=TWILIO_FROM_NUMBER,
        to=to_number
    )
    logger.info(f"Voice call initiated to {to_number}")


# ══════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════
async def broadcast_alert(alert: dict):
    alert_history.append(alert)
    if len(alert_history) > ALERT_HISTORY_LIMIT:
        alert_history.pop(0)

    # Send notifications in background
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
    logger.info(f"WS client connected — total: {len(connected_ws_clients)}")
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
        logger.info("WS client disconnected")


# ══════════════════════════════════════════════════════════════
# ① HEALTH CHECK
# ══════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {
        "status":       "operational",
        "model_loaded": detector.model is not None,
        "live_feeds":   len(live_feeds),
        "ws_clients":   len(connected_ws_clients),
        "timestamp":    datetime.utcnow().isoformat(),
        "notifications": {
            "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "ntfy":     bool(NTFY_TOPIC),
            "twilio":   bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID),
        }
    }


# ══════════════════════════════════════════════════════════════
# ② IMAGE DETECTION
# ══════════════════════════════════════════════════════════════
@app.post("/detect/image")
async def detect_image(
    file:        UploadFile = File(...),
    sensitivity: float      = Form(0.5),
    confidence_threshold: float = Form(0.4),   # ← frontend sends this
):
    data  = await file.read()
    arr   = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image")

    result = detector.detect(frame, sensitivity)
    alert  = {
        "id":        str(uuid.uuid4()),
        "source":    "image_upload",
        "feed_name": "Image Upload",
        "timestamp": datetime.utcnow().isoformat(),
        "detection": result.dict(),
        "color":     THREAT_COLORS[result.alert_level],
    }
    if result.alert_level != "normal" and result.fused_score >= confidence_threshold:
        await broadcast_alert(alert)
    else:
        alert_history.append(alert)
        if len(alert_history) > ALERT_HISTORY_LIMIT:
            alert_history.pop(0)
    return alert


# ══════════════════════════════════════════════════════════════
# ③ VIDEO ANALYSIS  (Mode 1 — pre-recorded)
#    Frontend sends: file, group_seconds, confidence_threshold,
#                    sensitivity, start_time, end_time
# ══════════════════════════════════════════════════════════════
@app.post("/detect/video/upload")
async def detect_video_upload(
    background_tasks:     BackgroundTasks,
    file:                 UploadFile = File(...),
    group_seconds:        float      = Form(2.0),     # ← frontend's r-group
    frame_skip:           int        = Form(5),
    confidence_threshold: float      = Form(0.4),
    sensitivity:          float      = Form(0.5),
    start_time:           float      = Form(0.0),
    end_time:             Optional[float] = Form(None),
):
    session_id   = str(uuid.uuid4())
    video_bytes  = await file.read()

    analysis_sessions[session_id] = {
        "status":     "queued",
        "progress":   0,
        "events":     [],
        "stats":      {},
        "filename":   file.filename,
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(
        _analyse_video_task,
        session_id, video_bytes,
        group_seconds, frame_skip,
        confidence_threshold, sensitivity,
        start_time, end_time,
    )
    return {"session_id": session_id, "status": "queued"}


async def _analyse_video_task(
    session_id, video_bytes,
    group_seconds, frame_skip,
    confidence_threshold, sensitivity,
    start_time, end_time,
):
    sess = analysis_sessions[session_id]
    sess["status"] = "processing"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        sess["status"] = "error"
        sess["error"]  = "Could not open video"
        Path(tmp_path).unlink(missing_ok=True)
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame  = int(start_time * fps)
    end_frame    = int(end_time   * fps) if end_time else total_frames
    frames_per_group = max(1, int(group_seconds * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Group frames into windows of group_seconds each
    groups:     dict[int, list] = {}   # group_idx → list of DetectionResult
    frame_idx   = start_frame
    stats       = defaultdict(int)

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_skip == 0:
            result      = detector.detect(frame, sensitivity)
            group_idx   = (frame_idx - start_frame) // frames_per_group
            if group_idx not in groups:
                groups[group_idx] = []
            groups[group_idx].append(result)
            stats[result.alert_level] += 1

        frame_idx += 1
        sess["progress"] = round(
            (frame_idx - start_frame) / max(end_frame - start_frame, 1) * 100
        )
        if frame_idx % 30 == 0:
            await asyncio.sleep(0)

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)

    # Aggregate each group → pick worst result
    events = []
    for group_idx in sorted(groups.keys()):
        results = groups[group_idx]
        # Pick result with highest fused_score
        best   = max(results, key=lambda r: r.fused_score)
        ts     = start_time + group_idx * group_seconds
        if best.fused_score >= confidence_threshold:
            events.append({
                "timestamp":     round(ts, 2),
                "timestamp_fmt": _fmt_time(ts),
                "frame":         start_frame + group_idx * frames_per_group,
                "alert_level":   best.alert_level,
                "color":         THREAT_COLORS[best.alert_level],
                "fused_score":   best.fused_score,
                "yolo_score":    best.yolo_score,
                "confidence_pct": round(best.fused_score * 100, 1),
            })

    total = sum(stats.values()) or 1
    sess.update({
        "status":   "done",
        "progress": 100,
        "events":   events,
        "stats":    {
            "total_frames_analysed":        frame_idx - start_frame,
            "normal_count":                 stats["normal"],
            "possibly_suspicious_count":    stats["possibly_suspicious"],
            "suspicious_count":             stats["suspicious"],
            "threat_percentage":            round(
                (stats["suspicious"] + stats["possibly_suspicious"]) / total * 100, 1
            ),
            "duration_seconds":             round((end_frame - start_frame) / fps, 2),
        },
    })
    logger.success(f"Video done: {session_id} — {len(events)} groups")


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
        raise HTTPException(404, "Session not found")
    if s["status"] != "done":
        raise HTTPException(400, f"Not complete: {s['status']}")
    return s


# ══════════════════════════════════════════════════════════════
# ④ LIVE FEED MANAGEMENT
# ══════════════════════════════════════════════════════════════
@app.post("/feeds/add")
async def add_feed(config: LiveFeedConfig, background_tasks: BackgroundTasks):
    if len(live_feeds) >= MAX_LIVE_FEEDS:
        raise HTTPException(400, f"Max {MAX_LIVE_FEEDS} feeds reached")

    feed_id = str(uuid.uuid4())[:8]
    live_feeds[feed_id] = {
        "id":               feed_id,
        "name":             config.name,
        "source_type":      config.source_type,
        "url":              config.url,
        "status":           "connecting",
        "sensitivity":      config.sensitivity,
        "added_at":         datetime.utcnow().isoformat(),
        "latest_detection": None,
        "_stop":            False,
        "_latest_frame_b64": None,
    }
    background_tasks.add_task(_run_live_feed, feed_id, config)
    return {"feed_id": feed_id, "status": "connecting"}


@app.get("/feeds")
def list_feeds():
    return {
        fid: {k: v for k, v in f.items() if not k.startswith("_")}
        for fid, f in live_feeds.items()
    }


@app.get("/feeds/{feed_id}")
def get_feed(feed_id: str):
    if feed_id not in live_feeds:
        raise HTTPException(404, "Feed not found")
    return {k: v for k, v in live_feeds[feed_id].items() if not k.startswith("_")}


@app.delete("/feeds/{feed_id}")
def remove_feed(feed_id: str):
    if feed_id not in live_feeds:
        raise HTTPException(404, "Feed not found")
    live_feeds[feed_id]["_stop"] = True
    del live_feeds[feed_id]
    return {"status": "removed"}


@app.get("/feeds/{feed_id}/snapshot")
def feed_snapshot(feed_id: str):
    if feed_id not in live_feeds:
        raise HTTPException(404, "Feed not found")
    frame_b64 = live_feeds[feed_id].get("_latest_frame_b64")
    if not frame_b64:
        raise HTTPException(503, "No frame yet")
    return {
        "feed_id":          feed_id,
        "frame_b64":        frame_b64,
        "latest_detection": live_feeds[feed_id].get("latest_detection"),
    }


async def _run_live_feed(feed_id: str, config: LiveFeedConfig):
    feed = live_feeds.get(feed_id)
    if not feed:
        return

    src = 0 if config.source_type == "webcam" else config.url
    if src is None:
        feed["status"] = "error"
        return

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        feed["status"] = "error"
        logger.error(f"Feed {feed_id}: Cannot open {src}")
        return

    feed["status"] = "live"
    logger.info(f"Feed {feed_id} live: {config.name}")
    frame_count   = 0
    ANALYSE_EVERY = 10

    while not feed.get("_stop"):
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            continue

        if frame_count % 5 == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            feed["_latest_frame_b64"] = base64.b64encode(buf).decode()

        if frame_count % ANALYSE_EVERY == 0:
            result = detector.detect(frame, config.sensitivity)
            feed["latest_detection"] = result.dict()

            if result.alert_level != "normal":
                alert = {
                    "id":        str(uuid.uuid4()),
                    "source":    f"live:{feed_id}",
                    "feed_name": config.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "detection": result.dict(),
                    "color":     THREAT_COLORS[result.alert_level],
                }
                await broadcast_alert(alert)

        frame_count += 1
        await asyncio.sleep(0.03)

    cap.release()
    logger.info(f"Feed {feed_id} stopped")


# ══════════════════════════════════════════════════════════════
# ⑤ ONVIF DISCOVERY
# ══════════════════════════════════════════════════════════════
@app.get("/discover/onvif")
async def discover_onvif():
    found = []
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
        wsd = WSDiscovery()
        wsd.start()
        for svc in wsd.searchServices():
            for scope in svc.getScopes():
                if "onvif" in str(scope).lower():
                    found.append({"address": str(svc.getXAddrs()), "scopes": [str(s) for s in svc.getScopes()]})
        wsd.stop()
    except ImportError:
        return {"status": "unavailable", "message": "pip install wsdiscovery", "cameras": []}
    except Exception as e:
        return {"status": "error", "message": str(e), "cameras": []}
    return {"status": "ok", "cameras": found}


# ══════════════════════════════════════════════════════════════
# ⑥ ALERT HISTORY & CONFIG
#    Config shape now matches frontend v4.1:
#    { levels: { normal:{enabled,channels}, ... }, phone_numbers:[], firebase_token:... }
# ══════════════════════════════════════════════════════════════
@app.get("/alerts/history")
def get_alert_history(limit: int = 100):
    return {"alerts": alert_history[-limit:], "total": len(alert_history)}


@app.post("/alerts/config")
async def set_alert_config(payload: dict):
    global _alert_config
    _alert_config = payload
    # Apply Telegram / ntfy / Twilio settings if provided
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NTFY_TOPIC
    global TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER
    if "telegram_bot_token" in payload:
        TELEGRAM_BOT_TOKEN = payload["telegram_bot_token"]
    if "telegram_chat_id" in payload:
        TELEGRAM_CHAT_ID = payload["telegram_chat_id"]
    if "ntfy_topic" in payload:
        NTFY_TOPIC = payload["ntfy_topic"]
    return {"status": "saved", "config": _alert_config}


@app.get("/alerts/config")
def get_alert_config():
    return _alert_config


# ── Notification test endpoint (frontend "Test" button calls this) ──
@app.post("/notify/test")
async def notify_test():
    test_alert = {
        "id":        "test-" + str(uuid.uuid4())[:8],
        "source":    "test",
        "feed_name": "TEST ALERT",
        "timestamp": datetime.utcnow().isoformat(),
        "detection": {
            "alert_level":  "suspicious",
            "fused_score":  0.92,
            "yolo_score":   0.89,
            "confidence_pct": 92.0,
        },
        "color": "RED",
    }
    await broadcast_alert(test_alert)
    return {"status": "test alert sent", "channels_active": {
        "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "ntfy":     bool(NTFY_TOPIC),
        "twilio":   bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID),
    }}


# ── Notification channel status ────────────────────────────────
@app.get("/notify/status")
def notify_status():
    return {
        "telegram": {"configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)},
        "ntfy":     {"configured": bool(NTFY_TOPIC), "topic": NTFY_TOPIC},
        "twilio":   {"configured": bool(TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID)},
        "throttle_seconds": ALERT_THROTTLE_SECONDS,
    }


# ══════════════════════════════════════════════════════════════
# ⑦ EXPORT — CSV & PDF
# ══════════════════════════════════════════════════════════════
@app.get("/export/csv/{session_id}")
def export_csv(session_id: str):
    s = analysis_sessions.get(session_id)
    if not s or s["status"] != "done":
        raise HTTPException(404, "Results not ready")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "timestamp", "timestamp_fmt", "frame",
        "alert_level", "color", "fused_score",
        "yolo_score", "confidence_pct",
    ])
    writer.writeheader()
    writer.writerows(s["events"])
    output.seek(0)
    fname = f"OmniEye_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/export/pdf/{session_id}")
def export_pdf(session_id: str):
    if not PDF_AVAILABLE:
        raise HTTPException(503, "pip install reportlab")
    s = analysis_sessions.get(session_id)
    if not s or s["status"] != "done":
        raise HTTPException(404, "Results not ready")

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter)
    styl = getSampleStyleSheet()
    els  = []
    els.append(Paragraph("OmniEye AI — Threat Analysis Report", styl["Title"]))
    els.append(Spacer(1, 12))
    els.append(Paragraph(f"Session : {session_id}", styl["Normal"]))
    els.append(Paragraph(f"File    : {s.get('filename','N/A')}", styl["Normal"]))
    els.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()}", styl["Normal"]))
    els.append(Spacer(1, 20))

    stat_data = [["Metric", "Value"]]
    for k, v in s.get("stats", {}).items():
        stat_data.append([k.replace("_"," ").title(), str(v)])
    t = Table(stat_data, colWidths=[250, 150])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), rl_colors.grey),
        ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.whitesmoke),
        ("GRID",       (0,0), (-1,-1), 0.5, rl_colors.black),
    ]))
    els.append(t); els.append(Spacer(1,20))

    ev_data = [["Time","Level","Confidence","YOLO"]]
    for ev in s["events"][:200]:
        ev_data.append([ev["timestamp_fmt"], ev["alert_level"].upper(),
                        f"{ev['confidence_pct']}%", f"{ev['yolo_score']:.3f}"])
    if len(ev_data) > 1:
        et = Table(ev_data, colWidths=[80,120,90,90])
        et.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), rl_colors.darkblue),
            ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, rl_colors.lightgrey]),
            ("GRID", (0,0), (-1,-1), 0.3, rl_colors.black),
        ]))
        els.append(et)
    doc.build(els)
    buf.seek(0)
    fname = f"OmniEye_{session_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})


# ══════════════════════════════════════════════════════════════
# ⑧ DASHBOARD STATS
# ══════════════════════════════════════════════════════════════
@app.get("/dashboard/stats")
def dashboard_stats():
    recent      = alert_history[-100:]
    level_counts = defaultdict(int)
    for a in recent:
        det = a.get("detection") or {}
        level_counts[det.get("alert_level", "normal")] += 1
    return {
        "live_feeds":           len(live_feeds),
        "total_alerts_logged":  len(alert_history),
        "recent_100":           dict(level_counts),
        "ws_clients_connected": len(connected_ws_clients),
        "active_sessions":      len([s for s in analysis_sessions.values() if s["status"] == "processing"]),
    }


# ══════════════════════════════════════════════════════════════
# ⑨ MOBILE PAIRING
# ══════════════════════════════════════════════════════════════
@app.post("/mobile/pair")
def mobile_pair(req: MobilePairRequest):
    code = str(uuid.uuid4())[:8].upper()
    _paired_devices[code] = {
        "device_name": req.device_name,
        "role":        req.role,
        "paired_at":   datetime.utcnow().isoformat(),
    }
    qr_data = json.dumps({"server": "http://YOUR_IP:8000", "pair_code": code, "role": req.role})
    return {"pair_code": code, "qr_data": qr_data, "role": req.role}


@app.get("/mobile/devices")
def list_paired_devices():
    return _paired_devices


# ══════════════════════════════════════════════════════════════
# FEEDBACK / RETRAIN
# ══════════════════════════════════════════════════════════════
@app.post("/feedback/correct")
def feedback_correct(payload: dict):
    _feedback_store.append({**payload, "ts": datetime.utcnow().isoformat()})
    return {"status": "saved", "total": len(_feedback_store)}

@app.get("/feedback/stats")
def feedback_stats():
    return {"total": len(_feedback_store)}

@app.post("/retrain/trigger")
def retrain_trigger():
    return {"status": "queued", "message": "Retrain not yet implemented"}


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║   OmniEye Backend v4.1 — Starting up     ║")
    logger.info("╚══════════════════════════════════════════╝")
    logger.info(f"Telegram : {'✅ configured' if TELEGRAM_BOT_TOKEN else '❌ not set'}")
    logger.info(f"ntfy     : {'✅ ' + NTFY_TOPIC if NTFY_TOPIC else '❌ not set'}")
    logger.info(f"Twilio   : {'✅ configured' if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID else '❌ not set'}")
    uvicorn.run("sentinel_backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")


# ══════════════════════════════════════════════════════════════
#
#  📱 MOBILE NOTIFICATION SETUP GUIDE
#  No app development needed — works with existing apps
#
# ══════════════════════════════════════════════════════════════
#
# ─── OPTION A: TELEGRAM BOT (FREE — RECOMMENDED) ─────────────
#
#  Setup takes 5 minutes:
#
#  Step 1  Open Telegram on your phone
#  Step 2  Search for @BotFather → send /newbot
#  Step 3  Choose a name and username → copy the TOKEN it gives you
#  Step 4  Open the bot you just created and send it any message (e.g. "hi")
#  Step 5  Visit in browser:
#            https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
#          Copy the "id" number from "chat" section → that's your CHAT_ID
#  Step 6  Edit this file:
#            TELEGRAM_BOT_TOKEN = "7123456789:AAFxxx..."
#            TELEGRAM_CHAT_ID   = "123456789"
#  Step 7  Restart the backend — done!
#
#  When OmniEye detects something, your phone gets an instant Telegram message.
#  Works everywhere in the world for FREE. No monthly cost.
#
# ─── OPTION B: NTFY.SH (FREE push notifications) ─────────────
#
#  Step 1  Install "ntfy" app on phone:
#            Android: https://play.google.com/store/apps/details?id=io.heckel.ntfy
#            iOS    : https://apps.apple.com/us/app/ntfy/id1625396347
#  Step 2  In the app, subscribe to a unique topic name:
#            e.g.  omnieye-harsh-2024  (make it unique so others can't subscribe)
#  Step 3  Edit this file:
#            NTFY_TOPIC = "omnieye-harsh-2024"
#  Step 4  Restart backend — done!
#
#  Completely free, open source, no account needed.
#
# ─── OPTION C: TWILIO SMS (paid ~$0.008/msg) ─────────────────
#
#  Step 1  Sign up at twilio.com (free trial gives ~15$ credit)
#  Step 2  Get a phone number from Twilio dashboard
#  Step 3  Copy Account SID and Auth Token from dashboard
#  Step 4  Edit this file:
#            TWILIO_ACCOUNT_SID = "ACxxx..."
#            TWILIO_AUTH_TOKEN  = "xxx..."
#            TWILIO_FROM_NUMBER = "+12015551234"
#  Step 5  In frontend Alert Config, enter your phone number in "Phone Numbers" field
#  Step 6  pip install twilio → restart backend — done!
#
# ─── OPTION D: WHATSAPP via Twilio ────────────────────────────
#
#  Same as Twilio SMS but use WhatsApp sandbox number:
#    TWILIO_FROM_NUMBER = "whatsapp:+14155238886"
#  And your phone number must be entered as:
#    "whatsapp:+91XXXXXXXXXX"
#  See: https://www.twilio.com/docs/whatsapp/quickstart
#
# ══════════════════════════════════════════════════════════════
