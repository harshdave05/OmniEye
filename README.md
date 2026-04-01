# 🧠 OmniEye AI — Suspicious Activity Detection

🚀 Real-time AI surveillance system to detect suspicious human behavior from CCTV, webcam, or video feeds.

---

## 🎯 Overview

OmniEye AI uses **computer vision + AI** to monitor environments and detect threats in real time.
Built for **smart cities, security systems, and fraud prevention use cases**.

---

## ⚡ Features

* 🎥 Live webcam detection
* 📡 CCTV / RTSP stream support
* 📂 Video file analysis
* 🚨 Real-time alerts (Telegram + WebSockets)
* 📊 Report export (CSV / PDF)
* 🧠 Pose-based behavior analysis

---

## 🧠 Tech Stack

* **AI/ML:** YOLOv8, MediaPipe, OpenCV
* **Backend:** FastAPI, WebSockets
* **Frontend:** HTML + JS
* **Integration:** Telegram Bot API

---

## 🏗️ Architecture

Frontend → FastAPI → AI Inference (YOLOv8 + Pose) → Alerts

---

## 🚀 Setup

```bash
git clone https://github.com/your-username/OmniEye-AI.git
cd OmniEye-AI
pip install -r requirements.txt
python sentinel_backend.py
```

Open `omnieye.html` in browser

---

## 📂 Structure

```bash
active_learning/  
fusion/  
inference/  
models/  
pose/  
utils/  
sentinel_backend.py  
omnieye.html  
```

---

## 📊 Model

* Trained on UCF-Crime, Violence Detection datasets
* ~200K+ images
* Optimized for real-time detection

---

## 🚨 Alerts

* Telegram notifications
* Real-time WebSocket alerts
* Event logging

---

## 🎯 Use Cases

Smart Cities • ATM Security • Retail Monitoring • Law Enforcement

---

## 📫 Contact

Harsh Dave
LinkedIn: https://www.linkedin.com/in/harsh-dave-0a0304333/ | Email: harshdavee1@gmail.com

---

⭐ Star this repo if you like it!
