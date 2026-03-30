# 3D Printing Failure Detection System — Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [ML Model & Detection Classes](#3-ml-model--detection-classes)
4. [Directory Structure](#4-directory-structure)
5. [Services](#5-services)
   - 5.1 [Frame Extractor](#51-frame-extractor)
   - 5.2 [MediaMTX — RTSP/WebRTC Relay](#52-mediamtx--rtspwebrtc-relay)
   - 5.3 [Dispatcher](#53-dispatcher)
   - 5.4 [Judge — Inference Engine](#54-judge--inference-engine)
   - 5.5 [Alert Manager](#55-alert-manager)
   - 5.6 [Budget Notifier](#56-budget-notifier)
6. [Dashboard](#6-dashboard)
7. [Infrastructure as Code (Terraform)](#7-infrastructure-as-code-terraform)
8. [CI/CD Pipelines](#8-cicd-pipelines)
9. [Security](#9-security)
10. [Configuration Reference](#10-configuration-reference)
11. [Deployment Guide](#11-deployment-guide)
12. [Cost Management](#12-cost-management)
13. [Technology Stack](#13-technology-stack)

---

## 1. Project Overview

This system performs **real-time 3D printing failure detection** using a fine-tuned YOLOv8x deep learning model hosted on Google Cloud Vertex AI. Three USB cameras mounted on a Raspberry Pi capture continuous video of 3D printers; each frame is analysed for ten categories of printing defect. When a high-severity defect is detected, operators receive email and browser push notifications, and a live web dashboard displays bounding-box overlays on the video streams.

The project is a thesis work at **Sapientia Hungarian University of Transylvania**.

**GCP Project ID:** `printermonitor-488112`
**Primary Region:** `europe-west1`
**Firebase / Web App URL:** `https://printermonitor-488112.web.app`

---

## 2. System Architecture

```
┌──────────────────────────── Raspberry Pi ──────────────────────────────┐
│                                                                          │
│  USB cam 1 ──┐                                                           │
│  USB cam 2 ──┼──► MediaMTX ──► RTSP input                               │
│  USB cam 3 ──┘        │                                                  │
│                        └──► WebRTC WHEP :8889 ──► Cloudflare Tunnel ─┐  │
│                                                                        │  │
│  Frame Extractor (Cloud Run)                                           │  │
│    - Reads RTSP URLs                                                   │  │
│    - Encodes frames as JPEG (base64)                                   │  │
│    - Publishes to Pub/Sub: frames-in ──────────────────────────────┐  │  │
└────────────────────────────────────────────────────────────────────│──│──┘
                                                                     │  │
┌────────────────────────── Google Cloud Platform ───────────────────│──│──┐
│                                                                     │  │  │
│  Pub/Sub: frames-in  ◄──────────────────────────────────────────────┘  │  │
│       │                                                                  │  │
│       ▼ (Eventarc)                                                       │  │
│  Cloud Function: Dispatcher                                              │  │
│       │                                                                  │  │
│       ▼ (HTTP POST)                                                      │  │
│  Vertex AI Endpoint ──► Judge container (YOLOv8x on T4 GPU)             │  │
│       │                                                                  │  │
│       ▼                                                                  │  │
│  Pub/Sub: detections-out                                                 │  │
│       │                                                                  │  │
│       ▼ (Eventarc)                                                       │  │
│  Cloud Function: Alert Manager                                           │  │
│       ├──► Firestore (alerts collection) ◄──── real-time listener        │  │
│       ├──► Gmail SMTP (email alerts)                                     │  │
│       └──► FCM (push notifications) ──────────────────────────────┐     │  │
│                                                                    │     │  │
│  Cloud Function: Budget Notifier                                   │     │  │
│       ├──► Firestore (budget_alerts collection)                    │     │  │
│       └──► Gmail SMTP / FCM                                        │     │  │
│                                                                    │     │  │
└────────────────────────────────────────────────────────────────────│─────┘
                                                                     │
┌─────────────────────── Browser / Dashboard ────────────────────────│──────┐
│                                                                     │      │
│  Firebase Web App                                                   │      │
│    ├──► WebRTC WHEP via Cloudflare Tunnel ◄─────────────────────────┘      │
│    ├──► Firestore onSnapshot() (live alerts + bounding boxes)               │
│    └──► FCM Service Worker (push notifications) ◄────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Step | From | To | Protocol | Payload |
|------|------|----|----------|---------|
| 1 | Cameras | Frame Extractor | USB / RTSP | Raw video |
| 2 | Frame Extractor | Pub/Sub `frames-in` | HTTPS | Base64 JPEG + metadata |
| 3 | Pub/Sub | Dispatcher (Eventarc) | Push | Cloud Event |
| 4 | Dispatcher | Vertex AI endpoint | HTTPS (REST) | `{"instances":[...]}` |
| 5 | Judge | Pub/Sub `detections-out` | HTTPS | Detections JSON |
| 6 | Pub/Sub | Alert Manager (Eventarc) | Push | Cloud Event |
| 7 | Alert Manager | Firestore | gRPC | Alert document |
| 8 | Alert Manager | Gmail SMTP | SMTP over SSL | HTML email |
| 9 | Alert Manager | FCM | HTTPS | Push notification |
| 10 | Firestore | Dashboard | WebSocket | Real-time snapshot |
| 11 | MediaMTX | Dashboard | WebRTC (WHEP) | H.264 video |

---

## 3. ML Model & Detection Classes

### Model

- **Architecture:** YOLOv8x (Extra-large — highest accuracy variant)
- **Framework:** Ultralytics + PyTorch 2.2
- **Hardware:** NVIDIA Tesla T4 GPU (CUDA 12.1)
- **Weights file:** `best.pt` (273 MB)
- **Storage:** `gs://printermonitor-488112-models/yolov8x/best.pt`
- **Overall mAP@0.5:** **0.904**
- **Confidence threshold:** 0.35 (configurable)

### Detection Classes

| Class | mAP@0.5 | Triggers Email Alert |
|-------|---------|---------------------|
| Spaghetti | 0.942 | **Yes** |
| Not sticking | 0.985 | **Yes** |
| Layer shift | 0.934 | **Yes** |
| Warping | 0.873 | **Yes** |
| Stringing | 0.971 | No |
| Under-extrusion | 0.883 | No |
| Over-extrusion | 0.812 | No |
| Nozzle clog | 0.821 | No |
| Foreign object on print area | 0.918 | No |

High-severity classes (spaghetti, not_sticking, layer_shift, warping) trigger both email notifications and FCM push notifications in addition to being stored in Firestore.

### Detection Output Format

Each detected object produces:

```json
{
  "label": "spaghetti",
  "confidence": 0.9245,
  "bbox": [x1, y1, x2, y2],
  "x": 0.1234,
  "y": 0.2345,
  "w": 0.3456,
  "h": 0.4567
}
```

Coordinates are normalised (0–1) relative to frame dimensions.

---

## 4. Directory Structure

```
3DPrinting-Failure-Detection/
├── README.MD                              # Project overview
├── firebase.json                          # Firebase hosting + rewrite rules
├── .firebaserc                            # Firebase project binding
├── .checkov.yaml                          # Checkov IaC security scan config
├── .pre-commit-config.yaml                # Pre-commit hooks (fmt, lint, checkov)
├── .tflint.hcl                            # TFLint rules (Google provider v0.30.0)
├── .gitignore                             # Excludes *.pt, credentials, state files
│
├── dashboard/                             # Web frontend
│   ├── index.html                         # SPA: live streams + alert log
│   └── firebase-messaging-sw.js           # FCM service worker
│
├── terraform_v2/
│   ├── terraform/                         # Terraform IaC
│   │   ├── main.tf                        # All GCP resources (526 lines)
│   │   ├── variables.tf                   # Input variable declarations
│   │   ├── outputs.tf                     # Output values
│   │   └── .terraform.lock.hcl            # Provider dependency lock
│   │
│   └── services/                          # Microservice source code
│       ├── frame-extractor/
│       │   ├── main.py                    # Multi-threaded RTSP → Pub/Sub publisher
│       │   └── Dockerfile                 # Python 3.12 slim + OpenCV
│       ├── mediamtx/
│       │   ├── Dockerfile                 # Based on bluenviron/mediamtx:latest
│       │   └── mediamtx.yml               # 3-camera RTSP/WebRTC config
│       ├── dispatcher/
│       │   ├── main.py                    # Cloud Function: frames-in → Vertex AI
│       │   └── requirements.txt
│       ├── judge/
│       │   ├── main.py                    # HTTP inference server (YOLOv8x)
│       │   ├── Dockerfile                 # PyTorch 2.2 + CUDA 12.1
│       │   ├── best.pt                    # Model weights (273 MB)
│       │   └── requirements.txt
│       ├── alert-manager/
│       │   ├── main.py                    # handle_detection + handle_budget_alert
│       │   └── requirements.txt
│       └── budget-notifier/
│           ├── main.py                    # FCM-based budget/defect notifications
│           └── requirements.txt
│
└── .github/workflows/
    ├── terraform.yml                      # Lint → Checkov → Plan → Apply
    ├── docker-judge.yml                   # Build & push Judge container
    └── compileLatex.yml                   # Compile thesis PDF
```

---

## 5. Services

### 5.1 Frame Extractor

**Source:** `terraform_v2/services/frame-extractor/main.py`
**Deployment:** Cloud Run (runs on Raspberry Pi)
**Container:** Python 3.12 slim + `opencv-python-headless`

#### Purpose
Reads live video from three RTSP streams published by MediaMTX, extracts frames at a configurable rate, compresses them as JPEG, encodes to base64, and publishes each frame to the `frames-in` Pub/Sub topic.

#### Runtime Behaviour
- Spawns one thread per camera.
- Each thread captures frames at `CAPTURE_FPS` (default 2 fps).
- Frames are resized to `FRAME_WIDTH × FRAME_HEIGHT` before encoding.
- An HTTP health check server listens on port 8080 for readiness probes.
- Each published message includes a monotonically increasing sequence number (`seq`) and an ISO-8601 timestamp.

#### Published Message Schema

```json
{
  "camera_id": "cam1",
  "seq": 123,
  "ts": "2026-03-30T15:45:30.123456+00:00",
  "data_b64": "<base64-encoded JPEG>"
}
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT` | — | GCP project ID |
| `FRAMES_TOPIC` | — | Pub/Sub topic name |
| `CAPTURE_FPS` | `2` | Frames per second to capture |
| `JPEG_QUALITY` | `70` | JPEG compression quality (0–100) |
| `FRAME_WIDTH` | `640` | Resize width in pixels |
| `FRAME_HEIGHT` | `480` | Resize height in pixels |
| `RTSP_URLS` | — | Comma-separated RTSP source URLs |
| `CAMERA_IDS` | — | Comma-separated IDs (e.g. `cam1,cam2,cam3`) |
| `PORT` | `8080` | Health check HTTP port |

---

### 5.2 MediaMTX — RTSP/WebRTC Relay

**Source:** `terraform_v2/services/mediamtx/`
**Deployment:** Raspberry Pi (Docker container)
**Base image:** `bluenviron/mediamtx:latest`

#### Purpose
Acts as the real-time media server on the Raspberry Pi. It accepts RTSP video pushed by the Frame Extractor's OpenCV capture threads and re-broadcasts it as WebRTC (WHEP) for the web dashboard. The WHEP endpoint is made internet-accessible via a Cloudflare Tunnel.

#### Port Assignments

| Port | Protocol | Role |
|------|----------|------|
| 8554 | RTSP | Camera input (publisher) |
| 8080 | RTMP | RTMP output (unused) |
| 8888 | HTTP | HLS output (unused) |
| 8889 | HTTP | WebRTC WHEP egress (used by dashboard) |
| 8890 | UDP | SRT (unused) |

#### Camera Paths

```yaml
paths:
  cam1:
    source: publisher   # Frame Extractor pushes here
  cam2:
    source: publisher
  cam3:
    source: publisher
```

#### Network Exposure

Cloudflare Tunnel exposes port 8889 externally. The dashboard connects to:
```
https://<cloudflare_tunnel_hostname>/cam1/whep
https://<cloudflare_tunnel_hostname>/cam2/whep
https://<cloudflare_tunnel_hostname>/cam3/whep
```

---

### 5.3 Dispatcher

**Source:** `terraform_v2/services/dispatcher/main.py`
**Deployment:** Cloud Function Gen2 (Python 3.12)
**Trigger:** Pub/Sub topic `frames-in` via Eventarc

#### Purpose
Acts as the bridge between the incoming frame queue and the Vertex AI inference endpoint. Receives a Pub/Sub Cloud Event containing a raw frame, authenticates with Google OAuth2, and forwards the frame to the Judge service via the Vertex AI prediction REST API.

#### Function Signature

```python
def dispatch_frame(cloud_event):
    ...
```

#### Behaviour

1. Decodes the base64-encoded Pub/Sub message data to JSON.
2. Extracts `data_b64`, `camera_id`, `seq`, `ts`.
3. Builds a Vertex AI prediction request body:
   ```json
   {
     "instances": [{
       "data_b64": "...",
       "camera_id": "cam1",
       "seq": 123,
       "ts": "2026-03-30T15:45:30.123456+00:00"
     }]
   }
   ```
4. POSTs to the Vertex AI endpoint using an OAuth2-authenticated `requests` session.
5. Logs the number of detections returned.
6. On failure, lets the exception propagate so Pub/Sub retries delivery.

#### Vertex AI Endpoint URL

```
https://europe-west1-aiplatform.googleapis.com/v1/projects/printermonitor-488112/locations/europe-west1/endpoints/6900414029643120640:predict
```

#### Resources

| Parameter | Value |
|-----------|-------|
| Memory | 512 MB |
| Timeout | 60 s |
| Max instances | 10 |
| Ingress | Internal only |

#### Environment Variables

| Variable | Value |
|----------|-------|
| `GCP_PROJECT` | `printermonitor-488112` |
| `VERTEX_ENDPOINT_ID` | `6900414029643120640` |
| `VERTEX_REGION` | `europe-west1` |

---

### 5.4 Judge — Inference Engine

**Source:** `terraform_v2/services/judge/main.py`
**Deployment:** Vertex AI custom container (NVIDIA Tesla T4)
**Base image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`

#### Purpose
The core ML inference service. Hosts the fine-tuned YOLOv8x model and exposes an HTTP server that accepts prediction requests from the Dispatcher, runs YOLO inference on the GPU, and publishes structured detection results to the `detections-out` Pub/Sub topic.

#### HTTP Endpoints

| Method | Path | Response |
|--------|------|----------|
| `GET` | `/healthz`, `/health`, `/` | `200 OK` |
| `POST` | `/predict` | Vertex AI-formatted prediction response |

#### Inference Pipeline

1. Receive `POST /predict` with Vertex AI `{"instances": [...]}` body.
2. Decode `data_b64` → raw JPEG bytes → OpenCV `Mat`.
3. Run `model.predict()` at the configured confidence threshold.
4. Parse bounding boxes, labels, and confidence scores.
5. Publish results to Pub/Sub `detections-out`:
   ```json
   {
     "ts": "2026-03-30T15:45:30.123456+00:00",
     "camera_id": "cam1",
     "seq": 123,
     "detections": [
       {
         "label": "spaghetti",
         "confidence": 0.9245,
         "bbox": [x1, y1, x2, y2],
         "x": 0.1234,
         "y": 0.2345,
         "w": 0.3456,
         "h": 0.4567
       }
     ]
   }
   ```
6. Return Vertex AI-formatted response to Dispatcher.

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GCP_PROJECT` | — | GCP project ID |
| `DETECTIONS_TOPIC` | — | Pub/Sub topic for results |
| `MODEL_PATH` | `/app/best.pt` | Path to YOLO weights file |
| `CONF_THRESHOLD` | `0.35` | Minimum detection confidence |
| `PORT` | `8080` | HTTP server port |

#### Container Registry

```
europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest
```

---

### 5.5 Alert Manager

**Source:** `terraform_v2/services/alert-manager/main.py`
**Deployment:** Cloud Function Gen2 (Python 3.12)
**Trigger:** Pub/Sub topic `detections-out` via Eventarc

#### Purpose
Consumes inference results, persists them to Firestore, and dispatches email notifications for high-severity detections. Implements a per-camera cooldown to prevent alert flooding.

#### Function: `handle_detection(cloud_event)`

**Workflow:**

1. Decode Pub/Sub message → JSON payload.
2. Filter detections where `confidence >= CONF_THRESHOLD`.
3. Write alert document to Firestore `alerts` collection:
   ```json
   {
     "camera_id": "cam1",
     "detections": [...],
     "timestamp": "2026-03-30T15:45:30.123456+00:00",
     "seq": 123,
     "created_at": "<server timestamp>"
   }
   ```
4. Check Firestore `alert_cooldowns` collection for per-camera cooldown.
5. If high-severity detection found **and** camera not on cooldown:
   - Build HTML email with a detection summary table.
   - Send via Gmail SMTP SSL (port 465).
   - Update cooldown document (60-second TTL).

**High-severity labels:** `spaghetti`, `not_sticking`, `layer_shift`, `warping`

#### Function: `handle_budget_alert(cloud_event)`

Triggered separately by the `budget-notifications` topic:

1. Decode budget alert payload (`costAmount`, `budgetAmount`, `budgetDisplayName`).
2. Write to Firestore `budget_alerts` collection.
3. Check cooldown (key: `"budget"`).
4. Send email notification if threshold exceeded and not on cooldown.

#### Resources

| Parameter | Value |
|-----------|-------|
| Memory | 256 MB |
| Timeout | 120 s |
| Max instances | 10 |
| Ingress | Internal only |

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `GCP_PROJECT` | GCP project ID |
| `FIRESTORE_COLLECTION` | Target collection name (`alerts`) |
| `CONF_THRESHOLD` | Confidence filter (default `0.35`) |
| `COOLDOWN_SECONDS` | Cooldown per camera (default `60`) |
| `GMAIL_ADDRESS` | Sender email address (**sensitive**) |
| `GMAIL_APP_PASSWORD` | Gmail app password (**sensitive**) |

---

### 5.6 Budget Notifier

**Source:** `terraform_v2/services/budget-notifier/main.py`
**Deployment:** Cloud Function Gen2 (Python 3.12)
**Trigger:** Pub/Sub topic `budget-notifications` via Eventarc

#### Purpose
Sends Firebase Cloud Messaging (FCM) push notifications when a GCP billing budget alert fires or when a high-severity defect is detected. Complements the Alert Manager's email functionality with browser/mobile push.

#### FCM Topics

| Topic | Triggered By |
|-------|-------------|
| `defect-alerts` | High-severity defect detections |
| `budget-alerts` | GCP budget threshold breached |

#### Notification Payload Example

```json
{
  "notification": {
    "title": "Defect Detected",
    "body": "cam1: spaghetti, layer shift"
  },
  "data": {
    "type": "defect",
    "camera_id": "cam1",
    "labels": "spaghetti,layer shift"
  },
  "topic": "defect-alerts",
  "android": { "priority": "high" },
  "webpush": {
    "notification": { "title": "...", "body": "..." },
    "fcm_options": { "link": "https://printermonitor-488112.web.app" }
  }
}
```

#### Resources

| Parameter | Value |
|-----------|-------|
| Memory | 256 MB |
| Timeout | 60 s |
| Max instances | 3 |

---

## 6. Dashboard

**Source:** `dashboard/index.html`, `dashboard/firebase-messaging-sw.js`
**Hosting:** Firebase App Hosting
**URL:** `https://printermonitor-488112.web.app`

### Technology

- **Framework:** Vanilla JavaScript (no build step)
- **Firebase SDK:** v11.4.0 (Firestore, FCM, App)
- **Video:** WebRTC via WHEP (W3C `RTCPeerConnection`)

### Features

#### Live Video Streams
- Three video tiles (cam1, cam2, cam3) displayed in a responsive grid.
- Each tile establishes a WebRTC WHEP connection to MediaMTX via the Cloudflare Tunnel.
- Status indicator: green (online), yellow (connecting), red (failed).
- "Reconnect all" button manually resets all peer connections.

#### Bounding Box Overlay
- A `<canvas>` element is overlaid on each video tile.
- On receiving a Firestore detection snapshot, the dashboard draws bounding boxes with per-class colours:

| Class | Colour |
|-------|--------|
| Stringing | `#e74c3c` (red) |
| Warping | `#e67e22` (orange) |
| Layer shift | `#f1c40f` (yellow) |
| Under-extrusion | `#3498db` (blue) |
| Over-extrusion | `#9b59b6` (purple) |
| Nozzle clog | `#1abc9c` (teal) |
| Foreign object | `#e91e63` (pink) |
| Not sticking | `#ff5722` (deep orange) |
| Spaghetti | `#2ecc71` (green) |

#### Event Log
- Displays the latest 50 Firestore alert documents in reverse-chronological order.
- Only shows alerts created after the page loaded (no historical backfill spam).
- High-severity alerts: red border + flashing header.
- Budget alerts: gold border.
- "Clear" button resets the counter.

#### Push Notifications (FCM)
- Service worker (`firebase-messaging-sw.js`) subscribes to `defect-alerts` and `budget-alerts` FCM topics.
- Foreground: custom toast notification (5-second auto-dismiss).
- Background: native browser notification.

#### UI
- Dark theme (`#0b0d10` background).
- 2 × 2 grid layout on desktop; stacked on mobile (breakpoint: 1100 px).
- Firestore connection status indicator (colour-coded).

### Firebase Configuration

```javascript
{
  projectId:         "printermonitor-488112",
  apiKey:            "AIzaSyCjp-IZzt2CfXEwBxnn2icv8LxvrsJmieQ",
  authDomain:        "printermonitor-488112.firebaseapp.com",
  storageBucket:     "printermonitor-488112.firebasestorage.app",
  messagingSenderId: "895714392909"
}
```

---

## 7. Infrastructure as Code (Terraform)

**Source:** `terraform_v2/terraform/`
**Backend:** GCS bucket `printermonitor-488112-functions-source` (remote state)
**Terraform version:** ≥ 1.6
**Providers:** `google` v5, `google-beta` v5, `archive` v2, `null` v3

### Enabled GCP APIs

Cloud Run, Pub/Sub, Firestore, Vertex AI, Cloud Functions, Cloud Build, Artifact Registry, Eventarc, Cloud Storage, Firebase, IAM, Cloud Logging, Billing Budgets, FCM, Firebase Installations.

### Key Resources

#### Pub/Sub Topics

| Topic | Retention | Purpose |
|-------|-----------|---------|
| `frames-in` | 1 hour | Raw video frames from Pi |
| `detections-out` | 1 hour | YOLO inference results |
| `budget-notifications` | 1 hour | GCP billing budget alerts |

#### Firestore

- **Location:** `eur3` (Europe multi-region)
- **Mode:** Native
- **Lifecycle:** Destroy prevention enabled
- **Composite Index:** `alerts` collection → `camera_id ASC`, `timestamp DESC`
- **Collections:** `alerts`, `alert_cooldowns`, `budget_alerts`

#### Cloud Functions (Gen2)

| Function | Entry Point | Trigger Topic | Memory | Timeout | Max Instances |
|----------|-------------|---------------|--------|---------|---------------|
| AlertManager | `handle_detection` | `detections-out` | 256 MB | 120 s | 10 |
| BudgetNotifier | `handle_budget_alert` | `budget-notifications` | 256 MB | 60 s | 3 |
| Dispatcher | `dispatch_frame` | `frames-in` | 512 MB | 60 s | 10 |

All functions use Python 3.12 runtime with `ALLOW_INTERNAL_ONLY` ingress.

#### Service Accounts & Roles

| Service Account | Roles |
|-----------------|-------|
| `sa-frame-extractor` | `pubsub.publisher`, `aiplatform.user` |
| `sa-dispatcher` | `pubsub.subscriber`, `pubsub.publisher`, `aiplatform.user`, `eventarc.eventReceiver` |
| `sa-alert-manager` | `datastore.user`, `pubsub.subscriber`, `firebase.sdkAdminServiceAgent`, `eventarc.eventReceiver` |

#### Cloud Storage Buckets

| Bucket | Purpose |
|--------|---------|
| `printermonitor-488112-models` | YOLOv8x weight files (versioning enabled) |
| `printermonitor-488112-functions-source` | Packaged Cloud Function zip archives + Terraform state |

#### Artifact Registry

- **Repository:** `printermonitor` (Docker, `europe-west1`)
- **Images:** `judge`, `dispatcher`, `frame-extractor`, `mediamtx`

#### Vertex AI

- **Endpoint ID:** `6900414029643120640`
- **Region:** `europe-west1`
- **GPU:** NVIDIA Tesla T4
- **Replicas:** 1 minimum / 1 maximum (manual scaling recommended to avoid idle GPU cost)

### Input Variables

| Variable | Default | Sensitive |
|----------|---------|-----------|
| `project_id` | `printermonitor-488112` | No |
| `region` | `europe-west1` | No |
| `model_gcs_bucket` | `printermonitor-488112-models` | No |
| `conf_threshold` | `0.35` | No |
| `cloudflare_tunnel_hostname` | `""` | No |
| `vertex_endpoint_id` | `6900414029643120640` | No |
| `gmail_address` | — | **Yes** |
| `gmail_app_password` | — | **Yes** |

### Outputs

| Output | Description |
|--------|-------------|
| `mediamtx_whep_url` | Base WebRTC WHEP URL |
| `vertex_endpoint_id` | Vertex AI endpoint ID |
| `frames_topic` | frames-in Pub/Sub topic |
| `detections_topic` | detections-out Pub/Sub topic |
| `artifact_registry` | Docker registry prefix |
| `firestore_database` | Firestore database name |
| `alert_manager_function` | Alert Manager Cloud Function name |
| `dispatcher_function` | Dispatcher Cloud Function name |
| `mediamtx_host` | Cloudflare tunnel hostname |

---

## 8. CI/CD Pipelines

### 8.1 Terraform Workflow (`.github/workflows/terraform.yml`)

**Trigger:** Push or PR to `main` with changes under `terraform_v2/`

| Job | Runner | Steps |
|-----|--------|-------|
| **Lint** | ubuntu-latest | `terraform fmt -check -recursive`, TFLint with Google ruleset v0.30.0 |
| **Checkov** | ubuntu-latest | Checkov v3.2.325 security scan (config: `.checkov.yaml`) |
| **Plan** | ubuntu-latest | GCP auth → `terraform init` → `terraform plan` → post plan to PR comment (max 60 KB) → upload `tfplan` artifact |
| **Apply** | ubuntu-latest | Download `tfplan` → `terraform apply` (main branch push only, requires "production" environment approval) |

**Required GitHub Secrets:** `GCP_SA_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`

### 8.2 Judge Docker Workflow (`.github/workflows/docker-judge.yml`)

**Trigger:** Push or PR to `main` with changes under `terraform_v2/services/judge/`

**Steps:**
1. Authenticate with GCP (`GCP_SA_KEY`).
2. Download model weights from GCS: `gs://printermonitor-488112-models/yolov8x/best.pt`.
3. `docker build` the Judge image.
4. Tag as `judge:{git_sha}` and `judge:latest`.
5. Push to Artifact Registry (push events to main only).
6. Write GitHub Step Summary.

### 8.3 LaTeX Compile Workflow (`.github/workflows/compileLatex.yml`)

**Trigger:** All push events

**Steps:**
1. Install TeX Live packages (`texlive-latex-extra`, `texlive-lang-european`, `ghostscript`).
2. Compile `docs/latex/docs/main.tex` to PDF.
3. Compress PDF with Ghostscript.
4. Commit and push compiled PDF back to the repository.

---

## 9. Security

### Secrets Management

All sensitive values are stored in **GitHub Secrets** and are never committed to the repository:

| Secret | Used By |
|--------|---------|
| `GCP_SA_KEY` | Terraform, Docker CI workflows |
| `GMAIL_ADDRESS` | Terraform (injected as Cloud Function env var) |
| `GMAIL_APP_PASSWORD` | Terraform (injected as Cloud Function env var) |

### IAM Least Privilege

Each service runs under a dedicated service account with only the IAM roles it needs (see §7 Service Accounts).

### Network Controls

- Cloud Functions use `ALLOW_INTERNAL_ONLY` ingress — only traffic from within the GCP project (Eventarc) can invoke them.
- Vertex AI is accessed via Google OAuth2 bearer tokens, not public endpoints.
- Raspberry Pi is exposed via Cloudflare Tunnel, not a direct public IP, keeping the device behind NAT.
- Storage bucket public access prevention is enforced.

### Checkov Exceptions (`.checkov.yaml`)

| Check | Reason for Skip |
|-------|----------------|
| `CKV_GCP_29` | Google-managed encryption sufficient for thesis |
| `CKV_GCP_62` | Bucket access logging not required |
| `CKV_GCP_78` | Cloud Functions are invoked by Eventarc (internal auth); no unauthenticated user access |
| `CKV_SECRET_4` | Firebase config API key flagged as false positive |
| `CKV_DOCKER_7` | `latest` tag used for development convenience |

### Pre-commit Hooks (`.pre-commit-config.yaml`)

Runs automatically before every commit:
- `terraform fmt` — enforces canonical formatting
- `tflint` — catches provider-specific mistakes
- `checkov` — security policy scan

---

## 10. Configuration Reference

### Key Thresholds & Limits

| Parameter | Value | Where Set |
|-----------|-------|-----------|
| Detection confidence threshold | 0.35 | `CONF_THRESHOLD` env var |
| Email cooldown per camera | 60 seconds | `COOLDOWN_SECONDS` env var |
| Budget alert threshold | $5 USD | GCP Billing Budget |
| Pub/Sub message retention | 3600 seconds | Terraform |
| Frame capture rate | 2 fps | `CAPTURE_FPS` env var |
| JPEG quality | 70 | `JPEG_QUALITY` env var |
| Frame resolution | 640 × 480 | `FRAME_WIDTH` / `FRAME_HEIGHT` env vars |
| Max Cloud Function instances | 10 (dispatcher, alert-manager), 3 (budget-notifier) | Terraform |

### Firestore Collections

| Collection | Documents | Purpose |
|------------|-----------|---------|
| `alerts` | One per frame with detections | Detection history + real-time dashboard feed |
| `alert_cooldowns` | One per camera | Tracks last email send time |
| `budget_alerts` | One per budget event | Budget alert history |

### Dashboard Live Query

```javascript
// Firestore query used by the dashboard
collection("alerts")
  .orderBy("timestamp", "desc")
  .limit(50)
  .onSnapshot(...)
```

The composite index `(camera_id ASC, timestamp DESC)` on the `alerts` collection ensures this query performs efficiently.

---

## 11. Deployment Guide

### Prerequisites

- GCP project with billing enabled (`printermonitor-488112`)
- Terraform ≥ 1.6 installed
- `gcloud` CLI authenticated with Owner/Editor access
- Docker installed (for building Judge container)
- Raspberry Pi with 3 USB cameras and Docker runtime
- Cloudflare account with a tunnel configured to port 8889 on the Pi
- Gmail account with an App Password generated

### Step 1 — Provision Infrastructure

```bash
cd terraform_v2/terraform
terraform init
terraform plan \
  -var="gmail_address=your@gmail.com" \
  -var="gmail_app_password=your-app-password" \
  -var="cloudflare_tunnel_hostname=your-tunnel.example.com"
terraform apply
```

### Step 2 — Upload Model Weights

```bash
gsutil cp best.pt gs://printermonitor-488112-models/yolov8x/best.pt
```

### Step 3 — Build & Push Judge Container

```bash
cd terraform_v2/services/judge
docker build -t europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest .
docker push europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest
```

### Step 4 — Deploy Judge to Vertex AI

```bash
gcloud ai endpoints deploy-model 6900414029643120640 \
  --model=<model-resource-id> \
  --display-name=yolov8x-v1 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --region=europe-west1
```

### Step 5 — Start Raspberry Pi Services

```bash
# Start MediaMTX (RTSP/WebRTC relay)
docker run -d --network=host \
  -v $(pwd)/mediamtx.yml:/mediamtx.yml \
  europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/mediamtx:latest

# Start Cloudflare Tunnel
cloudflared tunnel run <tunnel-name>

# Start Frame Extractor
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-frame-extractor-key.json
export GCP_PROJECT=printermonitor-488112
export FRAMES_TOPIC=frames-in
export RTSP_URLS=rtsp://localhost:8554/cam1,rtsp://localhost:8554/cam2,rtsp://localhost:8554/cam3
export CAMERA_IDS=cam1,cam2,cam3
docker run -d \
  -e GCP_PROJECT -e FRAMES_TOPIC -e RTSP_URLS -e CAMERA_IDS \
  -v $GOOGLE_APPLICATION_CREDENTIALS:/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/credentials.json \
  europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/frame-extractor:latest
```

### Step 6 — Deploy Dashboard

```bash
firebase deploy --only hosting
```

---

## 12. Cost Management

### Estimated Costs

| Resource | Idle | Active |
|----------|------|--------|
| Vertex AI T4 GPU | ~$0/day (when undeployed) | ~$37/day |
| Cloud Functions | Pay-per-invocation | Minimal |
| Pub/Sub | Pay-per-message | Minimal |
| Firestore | Pay-per-read/write | Minimal |
| Firebase Hosting | Free tier | Free tier |

### Budget Alert

A GCP Billing Budget is configured at **$5 USD**. When spending approaches this threshold, the `budget-notifications` Pub/Sub topic receives a message, triggering the Budget Notifier Cloud Function to send both an email and a push notification.

### Undeploying Vertex AI (to stop GPU billing)

```bash
# List deployed models on the endpoint
gcloud ai endpoints describe 6900414029643120640 --region=europe-west1

# Undeploy to stop GPU charges
gcloud ai endpoints undeploy-model 6900414029643120640 \
  --deployed-model-id=<deployed-model-id> \
  --region=europe-west1
```

---

## 13. Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| ML Model | YOLOv8x (Ultralytics) | Latest fine-tuned |
| Deep Learning | PyTorch | 2.2.0 |
| GPU / CUDA | NVIDIA Tesla T4 + CUDA | 12.1 |
| Video Capture | OpenCV (`cv2`) | Latest |
| RTSP/WebRTC Relay | MediaMTX | Latest |
| Message Queue | Google Cloud Pub/Sub | — |
| Serverless Compute | Cloud Functions Gen2 | Python 3.12 |
| ML Platform | Google Vertex AI | — |
| NoSQL Database | Google Cloud Firestore (Native) | — |
| Frontend | Vanilla JavaScript + Firebase SDK | 11.4.0 |
| Web Hosting | Firebase App Hosting | — |
| Push Notifications | Firebase Cloud Messaging (FCM) | — |
| Infrastructure as Code | Terraform | ≥ 1.7 |
| Container Registry | Google Artifact Registry | — |
| Email | Gmail SMTP (SSL, port 465) | — |
| Network Tunnel | Cloudflare Tunnel | — |
| Security Scanning | Checkov | 3.2.325 |
| Linting | TFLint (Google ruleset) | v0.30.0 |
| CI/CD | GitHub Actions | — |
| Thesis Compilation | LaTeX (TeX Live) + Ghostscript | — |
