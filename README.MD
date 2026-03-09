# Monitoring 3D Printing Processes with AI

A real-time 3D printer failure detection system built as a thesis project. Three USB cameras mounted on active printers stream live video through a Raspberry Pi 4B to Google Cloud, where a custom-trained YOLOv8x model detects 9 failure classes. Detections are stored in Firestore, visualised live with bounding boxes in a browser dashboard, and trigger email alerts for high-severity failures.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RASPBERRY PI 4B                                                              │
│                                                                              │
│  USB Cam 1 ─┐                                                                │
│  USB Cam 2 ─┼──► GStreamer (H.264, 30fps, 500kbps) ──► MediaMTX             │
│  USB Cam 3 ─┘                                          ├─ RTSP  :8554        │
│                                                        └─ WebRTC :8889       │
│                                          frame-extractor/main.py             │
│                                          (pulls RTSP, encodes base64 JPEG)   │
└──────────────────────────────────────────────┬──────────────────────────────┘
                                               │ Pub/Sub: frames-in
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ GOOGLE CLOUD                                                                 │
│                                                                              │
│  Pub/Sub (frames-in)                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  dispatcher (Cloud Function)                                                 │
│       │                                                                      │
│       ▼                                                                      │
│  Vertex AI Endpoint                                                          │
│  YOLOv8x · standard-4 · n1                                                  │
│  9 defect classes                                                            │
│       │ detections JSON                                                      │
│       ▼                                                                      │
│  Pub/Sub (detections-out)                                                    │
│       │                                                                      │
│       ▼                                                                      │
│  alert-manager (Cloud Function)                                              │
│  ├── Firestore (alerts collection)  ◄──── realtime listener ──► Dashboard   │
│  └── Gmail SMTP email alert                                                  │
│                                                                              │
│  budget-notifier (Cloud Function) ── GCP spend alerts ──► Gmail             │
└─────────────────────────────────────────────────────────────────────────────┘
                    │ WebRTC WHEP via Cloudflare Tunnel (HTTPS)
                    ▼
┌───────────────────────────────┐
│ BROWSER DASHBOARD             │
│ Firebase Hosting              │
│ ├─ Live video (3 cameras)     │
│ ├─ Bounding box overlay       │
│ ├─ Firestore event log        │
│ └─ In-browser toast alerts    │
└───────────────────────────────┘
```

---

## Model Performance

The YOLOv8x model was trained from scratch on a custom dataset over **99 epochs (~17.7 hours)** on Google Cloud.

| Metric | Value |
|---|---|
| **mAP@50** | **87.6%** |
| **mAP@50-95** | **73.1%** |
| **Precision** | **93.5%** |
| **Recall** | **83.9%** |
| Best epoch | 77 |
| Training time | ~17.7 hours |

Training loss converged steadily from epoch 1 (mAP50: 62.6%) to epoch 77 (mAP50: 87.6%), with all three loss components (box, cls, dfl) decreasing consistently across validation.

---

## Detected Failure Classes

| Class | Severity | Description |
|---|---|---|
| `layer_shift` | 🔴 High | Print layers misaligned mid-print |
| `warping` | 🔴 High | Material lifting off the print bed |
| `not_sticking` | 🔴 High | First layer not adhering to bed |
| `spagetti` | 🔴 High | Extruder printing into air, filament tangling |
| `stringing` | 🟡 Medium | Thin strands between print parts |
| `under_extrusion` | 🟡 Medium | Too little filament being extruded |
| `over_extrusion` | 🟡 Medium | Too much filament being extruded |
| `nozzle_clog` | 🟡 Medium | Blocked nozzle causing flow issues |
| `foreign_object_on_print_area` | 🟡 Medium | Unexpected object on the print bed |

High-severity detections trigger an email alert with cooldown. All detections are written to Firestore and rendered live in the dashboard.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Edge hardware | Raspberry Pi 4B |
| Cameras | 3× USB webcams |
| Video pipeline | GStreamer → MediaMTX (RTSP + WebRTC WHEP) |
| Tunnel | Cloudflare Tunnel |
| Frame extraction | Python (`frame-extractor/main.py`) |
| Message queue | Google Cloud Pub/Sub |
| ML inference | Vertex AI — YOLOv8x (standard-4, n1) |
| Routing | Cloud Functions — `dispatcher` |
| Alert logic | Cloud Functions — `alert-manager` (Python 3.12, 2nd gen) |
| Database | Firestore |
| Email alerts | Gmail SMTP via Python `smtplib` (no dependencies) |
| Secrets | Google Secret Manager |
| Dashboard | Vanilla JS + WebRTC + Firebase Hosting |
| Infrastructure | Terraform (GCP provider 5.45.2) |
| CI/CD | GitHub Actions (Terraform plan/apply + Docker build) |
| Security scanning | Checkov, tflint, pre-commit |

---

## Project Structure

```
.
├── terraform_v2/
│   ├── terraform/                  # All GCP infrastructure as code
│   │   ├── main.tf                 # Pub/Sub, Vertex AI, Cloud Functions, IAM
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── services/
│   │   ├── frame-extractor/        # Runs on Pi — pulls RTSP, publishes to Pub/Sub
│   │   │   ├── main.py
│   │   │   └── Dockerfile
│   │   ├── judge/                  # YOLOv8x inference service
│   │   │   ├── main.py
│   │   │   ├── best.pt             # Trained model weights
│   │   │   ├── requirements.txt
│   │   │   └── Dockerfile
│   │   ├── dispatcher/             # Cloud Function — routes frames-in to Vertex AI
│   │   │   ├── main.py
│   │   │   └── requirements.txt
│   │   ├── alert-manager/          # Cloud Function — detections → Firestore + email
│   │   │   ├── main.py
│   │   │   └── requirements.txt
│   │   ├── budget-notifier/        # Cloud Function — GCP spend → email
│   │   │   ├── main.py
│   │   │   └── requirements.txt
│   │   └── mediamtx/               # MediaMTX config + Dockerfile for Pi
│   │       ├── mediamtx.yml
│   │       └── Dockerfile
│   └── dashboard/
│       └── index.html              # Browser dashboard — WebRTC + Firestore live view
├── old_stuff_for_teaching/         # Earlier prototype — Dockerised local inference
│   └── app/
│       ├── scripts/                # Dataset tools: scraper, augment, splitter, trainer
│       └── src/printerfail/        # Python package — YOLO wrapper, data pipeline
├── .github/workflows/
│   ├── terraform.yml               # Terraform plan on PR, apply on merge
│   └── docker-judge.yml            # Build & push judge Docker image
├── .checkov.yaml                   # IaC security scan config
├── .tflint.hcl                     # Terraform linting config
└── .pre-commit-config.yaml         # Pre-commit hooks
```

---

## Setup & Deployment

### Prerequisites

- Raspberry Pi 4B with 3 USB cameras
- Google Cloud project with billing enabled
- Firebase project (same GCP project)
- `gcloud` CLI authenticated
- Terraform >= 1.0 installed
- Firebase CLI installed

### 1. Provision GCP Infrastructure

```bash
cd terraform_v2/terraform
terraform init
terraform plan
terraform apply
```

This provisions: Pub/Sub topics, Vertex AI endpoint, Cloud Functions, Firestore, IAM service accounts, and Secret Manager.

### 2. Store Secrets

```bash
# Gmail credentials for email alerts
echo -n "your@gmail.com"       | gcloud secrets create GMAIL_ADDRESS      --data-file=-
echo -n "your_app_password"    | gcloud secrets create GMAIL_APP_PASSWORD  --data-file=-

# Grant alert-manager access
for SECRET in GMAIL_ADDRESS GMAIL_APP_PASSWORD; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:sa-alert-manager@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

> Gmail App Passwords require 2-Step Verification to be enabled on your Google account. Generate one at myaccount.google.com/security → App passwords.

### 3. Deploy Cloud Functions

```bash
# alert-manager
cd terraform_v2/services/alert-manager
gcloud functions deploy alert-manager \
  --region=europe-west1 \
  --set-secrets="GMAIL_ADDRESS=GMAIL_ADDRESS:latest,GMAIL_APP_PASSWORD=GMAIL_APP_PASSWORD:latest" \
  --set-env-vars="COOLDOWN_SECONDS=60,CONF_THRESHOLD=0.35"
```

### 4. Raspberry Pi Setup

```bash
# Start MediaMTX (RTSP + WebRTC server)
docker compose up mediamtx

# Start frame extractor
docker compose up frame-extractor
```

### 5. Deploy Dashboard

```bash
cd terraform_v2
firebase deploy --only hosting
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CONF_THRESHOLD` | Minimum detection confidence to process | `0.35` |
| `COOLDOWN_SECONDS` | Minimum seconds between emails per camera | `60` |
| `GMAIL_ADDRESS` | Sender/recipient Gmail address (Secret Manager) | — |
| `GMAIL_APP_PASSWORD` | Gmail App Password (Secret Manager) | — |
| `GCP_PROJECT` | GCP project ID | `printermonitor-488112` |
| `FIRESTORE_COLLECTION` | Firestore collection for alerts | `alerts` |

---

## CI/CD

Two GitHub Actions workflows run automatically:

- **`terraform.yml`** — runs `terraform plan` on every pull request and `terraform apply` on merge to main, keeping infrastructure in sync with code.
- **`docker-judge.yml`** — builds and pushes the `judge` Docker image (YOLOv8x inference service) on changes to `services/judge/`.

---

## Email Alert Format

```
Subject: 🚨 Printer Alert — cam1

🚨 Printer Defect Detected
Camera: cam1
Time: Mon 09 Mar 2026, 19:38:05

┌─────────────────┬────────────┐
│ Class           │ Confidence │
├─────────────────┼────────────┤
│ layer_shift     │ 92%        │
│ warping         │ 78%        │
└─────────────────┴────────────┘

[Open Dashboard]
```

Alerts include a per-camera cooldown to prevent spam. Budget alerts follow the same format.

---

## Old Prototype (`old_stuff_for_teaching/`)

An earlier local-inference prototype is preserved in `old_stuff_for_teaching/`. It ran YOLOv8 inside a Docker container on the same machine as the cameras, without cloud infrastructure. It includes the original dataset tooling used to build and augment the training set:

- `imagescraper.py` / `nagyscrape.py` — image collection scripts
- `augment.py` — data augmentation pipeline
- `splitter.py` / `merge.py` — dataset preparation
- `teacher.py` / `teacher_weighted.py` — model training scripts
- `tester.py` / `videotester.py` — evaluation scripts

---

## License

MIT