# 3D Printing Failure Detection System

An automated real-time 3D printer failure detection system using computer vision, edge hardware, and cloud infrastructure. Three cameras monitor active print jobs; detected failures trigger email alerts and are visualised live in a browser dashboard.

---

## System Architecture

```
Raspberry Pi                    Google Cloud                      Browser
───────────────────             ────────────────────────────      ──────────────────
Cam 1 ─┐                        Pub/Sub (frames-in)
Cam 2 ─┼─ GStreamer ─ MediaMTX ─────────────────────► Vertex AI (YOLOv8x)
Cam 3 ─┘   H.264                frame_extractor.py         │
           RTSP :8554            base64 JPEG + metadata     ▼
           WebRTC :8889                               Pub/Sub (detections-out)
               │                                           │
               │ WebRTC WHEP                               ▼
               ▼                                   Cloud Function (alert-manager)
        Cloudflare Tunnel                          ├─ Firestore (alerts collection)
               │                                  └─ Gmail email alert
               ▼                                           │
        Dashboard (browser) ◄──── Firestore realtime ─────┘
```

---

## Features

- **3-camera live monitoring** via WebRTC (WHEP) streamed from a Raspberry Pi
- **YOLOv8x inference** on Google Cloud Vertex AI — 9 defect classes
- **Real-time bounding box overlay** rendered on the dashboard canvas
- **Email alerts** via Gmail SMTP for high-severity detections (with per-camera cooldown)
- **Firestore event log** — all detections stored and displayed live in the dashboard
- **GCP budget alerts** — notified by email when spend exceeds threshold

---

## Detected Defect Classes

| Class | Severity |
|---|---|
| `layer_shift` | 🔴 High |
| `warping` | 🔴 High |
| `not_sticking` | 🔴 High |
| `spagetti` | 🔴 High |
| `stringing` | 🟡 Medium |
| `under_extrusion` | 🟡 Medium |
| `over_extrusion` | 🟡 Medium |
| `nozzle_clog` | 🟡 Medium |
| `foreign_object_on_print_area` | 🟡 Medium |

High-severity detections trigger an email alert. All detections are written to Firestore and shown in the dashboard.

---

## Stack

| Layer | Technology |
|---|---|
| Edge hardware | Raspberry Pi 4 |
| Video capture | GStreamer (H.264, 30fps, 500kbps) |
| Streaming server | MediaMTX (RTSP + WebRTC WHEP) |
| Tunnel | Cloudflare Tunnel |
| Frame extraction | Python (`frame_extractor.py`) |
| Message queue | Google Cloud Pub/Sub |
| ML inference | Vertex AI — YOLOv8x (standard-4, n1) |
| Alert logic | Cloud Functions (Python 3.12, 2nd gen) |
| Database | Firestore |
| Email alerts | Gmail SMTP via `smtplib` |
| Secrets | Google Secret Manager |
| Dashboard | Vanilla JS + Firebase Hosting |
| Infrastructure | Terraform |

---

## Project Structure

```
.
├── terraform_v2/
│   ├── terraform/          # GCP infrastructure (Pub/Sub, Vertex AI, IAM, etc.)
│   └── services/
│       ├── alert-manager/  # Cloud Function — detections → Firestore + email
│       │   ├── main.py
│       │   └── requirements.txt
│       ├── dispatcher/     # Cloud Function — frames routing
│       └── budget-notifier/# Cloud Function — GCP budget alerts
├── frame_extractor.py      # Runs on Pi — pulls RTSP frames, publishes to Pub/Sub
└── dashboard/
    └── index.html          # Browser dashboard — WebRTC + Firestore live view
```

---

## Setup

### Prerequisites

- Raspberry Pi 4 with 3 USB/CSI cameras
- Google Cloud project with billing enabled
- Firebase project (same GCP project)
- `gcloud` CLI authenticated
- Terraform installed

### 1. Infrastructure

```bash
cd terraform_v2/terraform
terraform init
terraform apply
```

### 2. Deploy Cloud Functions

```bash
# Store secrets
echo -n "your@gmail.com" | gcloud secrets create GMAIL_ADDRESS --data-file=-
echo -n "your_app_password" | gcloud secrets create GMAIL_APP_PASSWORD --data-file=-

# Grant access
for SECRET in GMAIL_ADDRESS GMAIL_APP_PASSWORD; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:sa-alert-manager@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done

# Deploy
cd terraform_v2/services/alert-manager
gcloud functions deploy alert-manager \
  --region=europe-west1 \
  --set-secrets="GMAIL_ADDRESS=GMAIL_ADDRESS:latest,GMAIL_APP_PASSWORD=GMAIL_APP_PASSWORD:latest" \
  --set-env-vars="COOLDOWN_SECONDS=60,CONF_THRESHOLD=0.35"
```

### 3. Raspberry Pi

```bash
# Start MediaMTX
./mediamtx

# Start frame extractor
python3 frame_extractor.py
```

### 4. Dashboard

```bash
cd dashboard
firebase deploy --only hosting
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CONF_THRESHOLD` | Minimum confidence to store a detection | `0.35` |
| `COOLDOWN_SECONDS` | Minimum seconds between emails per camera | `60` |
| `GMAIL_ADDRESS` | Gmail address (Secret Manager) | — |
| `GMAIL_APP_PASSWORD` | Gmail App Password (Secret Manager) | — |
| `GCP_PROJECT` | GCP project ID | `printermonitor-488112` |
| `FIRESTORE_COLLECTION` | Firestore collection name | `alerts` |

---

## Email Alert Example

```
Subject: 🚨 Printer Alert — cam1

🚨 Printer Defect Detected
Camera: cam1
Time: Mon 09 Mar 2026, 19:38:05

| Class       | Confidence |
| layer_shift | 92%        |

[Open Dashboard]
```

---

