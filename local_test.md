# Local Test Pipeline

Offline replica of the cloud detection system for local development and testing.
**Original cloud files are never modified** — this is a standalone harness in `local_test/`.

---

## Cloud → Local component map

| Cloud component | Local replacement | File |
|---|---|---|
| Pub/Sub `frames-in` | Direct HTTP POST | `orchestrator.py` |
| Dispatcher Cloud Function | Eliminated — orchestrator calls judge directly | — |
| Vertex AI endpoint | `HTTPServer` on `localhost:8080` | `local_judge.py` |
| YOLOv8x on T4 GPU | Same `best.pt`, CPU inference | `local_judge.py` |
| Pub/Sub `detections-out` | HTTP response body (judge already returns this) | `orchestrator.py` |
| Firestore alerts collection | In-memory list `_alert_log` | `local_alert_handler.py` |
| Firestore cooldown collection | In-memory dict `_cooldowns` | `local_alert_handler.py` |
| Gmail via Alert Manager CF | Same SMTP code, skipped if no credentials | `local_alert_handler.py` |
| FCM push | Omitted (console log only) | — |
| Raspberry Pi cameras (RTSP) | Local webcam / video file / image dir | `orchestrator.py` |

---

## Files

```
local_test/
├── orchestrator.py        Main entry point. Starts the judge server, drives the
│                          capture loop, feeds detections to the alert handler.
├── local_judge.py         YOLOv8x HTTP server (localhost:8080). Identical logic to
│                          the production judge, minus the Pub/Sub publish.
├── local_alert_handler.py Alert logic with in-memory cooldown and file logging.
│                          Same HIGH_SEV set and cooldown rules as production.
└── run_local.sh           Convenience wrapper — sets env vars and calls orchestrator.
local_test.md              This file. Runtime detection events are appended below.
```

---

## Setup

```bash
# Install dependencies (same packages the services already use)
pip install ultralytics opencv-python-headless requests

# Confirm the model exists
ls terraform_v2/services/judge/best.pt
```

---

## Usage

```bash
cd local_test

# Webcam (default: device 0, 1 fps)
./run_local.sh

# Explicit webcam at 2 fps
./run_local.sh --source 0 --fps 2.0

# Three webcams in parallel (indices 0, 1, 2)
./run_local.sh --source 0 --cameras 3

# Video file
./run_local.sh --source /path/to/print.mp4

# Directory of images (loops)
./run_local.sh --source /path/to/frames/

# Single image (loops forever — useful for checking a known defect triggers correctly)
./run_local.sh --source /path/to/spaghetti.jpg

# Custom model or confidence threshold
./run_local.sh --source 0 --model /other/model.pt --conf 0.4

# With email alerts (same credentials as production)
GMAIL_ADDRESS=you@gmail.com GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx" ./run_local.sh
```

All output goes to stdout. Every detected frame is also appended to this file below.

---

## Detection log

<!-- Runtime sessions are appended below this line by orchestrator.py -->

## Session — Mon 04 May 2026, 14:25:52 UTC

- **Source:** `0`
- **FPS:** 2.0
- **Model:** `../terraform_v2/services/judge/best.pt`
- **Confidence threshold:** 0.35

| Time | Camera | Seq | Detections |
|------|--------|-----|------------|
