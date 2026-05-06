# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time 3D printing failure detection system using YOLOv8x on Google Cloud Platform. A thesis project at Sapientia Hungarian University of Transylvania. The system captures frames from USB cameras on a Raspberry Pi, runs inference on Vertex AI, and alerts operators via email and a live web dashboard.

- **GCP Project:** `printermonitor-488112` | **Region:** `europe-west1`
- **Dashboard:** `https://printermonitor-488112.web.app`

## Commands

### Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests with coverage (90% threshold enforced)
pytest tests/ -v --tb=short --cov=terraform_v2/services --cov-report=term-missing --cov-fail-under=90

# Run a single test file
pytest tests/test_judge.py -v

# Run a specific test
pytest tests/test_alert_manager.py::TestHandleDetection::test_sends_email_for_high_severity_detection -v
```

### Linting & Infrastructure Validation

```bash
# Terraform formatting and linting
cd terraform_v2/terraform
terraform fmt -check -recursive
tflint --init --config ../../.tflint.hcl
tflint --config ../../.tflint.hcl --recursive

# Security scan
checkov --config-file .checkov.yaml -d .

# Run all pre-commit checks (terraform fmt, tflint, checkov, pytest)
pre-commit run --all-files
```

### Deployment

```bash
# Deploy infrastructure (requires GCP credentials)
cd terraform_v2/terraform
terraform init
terraform plan -var="gmail_address=YOU@gmail.com" -var="gmail_app_password=YOUR_APP_PASSWORD"
terraform apply
# Note: terraform apply runs automatically on push to main via CI/CD

# Build and push Judge container
docker build -t europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest \
  terraform_v2/services/judge/
```

**Cost warning:** The T4 GPU for Vertex AI costs ~$37/day. Always undeploy after testing.

### Vertex AI Endpoint (manual — not managed by Terraform)

**Endpoint ID:** `6900414029643120640`

```bash
# Upload a new model version (increment vN)
gcloud ai models upload \
  --region=europe-west1 --project=printermonitor-488112 \
  --display-name=yolov8x-printermonitor-vN \
  --container-image-uri=europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest \
  --container-ports=8080 \
  --container-health-route=/healthz \
  --container-predict-route=/predict \
  --container-env-vars=GCP_PROJECT=printermonitor-488112,DETECTIONS_TOPIC=detections-out,MODEL_PATH=/app/best.pt,CONF_THRESHOLD=0.35,STREAK_REQUIRED=3

# Deploy to endpoint — MUST use --service-account=judge-svc (required for Pub/Sub publish permission)
gcloud ai endpoints deploy-model 6900414029643120640 \
  --region=europe-west1 --project=printermonitor-488112 \
  --model=MODEL_ID \
  --display-name=judge \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --min-replica-count=1 --max-replica-count=1 \
  --traffic-split=0=100 \
  --service-account=judge-svc@printermonitor-488112.iam.gserviceaccount.com

# Undeploy (stop T4 billing)
gcloud ai endpoints undeploy-model 6900414029643120640 \
  --region=europe-west1 --project=printermonitor-488112 \
  --deployed-model-id=DEPLOYED_MODEL_ID

# Get deployed model ID
gcloud ai endpoints describe 6900414029643120640 \
  --region=europe-west1 --project=printermonitor-488112 \
  --format="json(deployedModels)"
```

### Pub/Sub backlog & Firestore flush

If the pipeline falls behind, stale frames queue up and overwhelm the judge. Purge before/after testing:

```bash
# Purge frames-in backlog (Eventarc subscription — discard all queued frames)
gcloud pubsub subscriptions seek eventarc-europe-west1-dispatcher-635712-sub-152 \
  --time=$(date -u +%Y-%m-%dT%H:%M:%SZ) --project=printermonitor-488112

# Flush Firestore (alerts, cooldowns, budget_alerts)
python scripts/flush_firestore.py

# Flush only cooldown locks (if emails are being suppressed)
python scripts/flush_firestore.py --collections cooldowns
```

## Architecture

### Data Flow

```
Raspberry Pi (3 USB cameras)
  → MediaMTX (RTSP/WebRTC) → Cloudflare Tunnel → Browser Dashboard
  → Frame Extractor → Pub/Sub: frames-in
                          ↓ (Eventarc)
                    Dispatcher (Cloud Function)
                          ↓ (HTTP)
                    Judge on Vertex AI (YOLOv8x, T4 GPU)
                          ↓ (publishes)
                    Pub/Sub: detections-out
                      ├─ (Eventarc) Alert Manager CF → Firestore + Gmail + FCM
                      └─ (Eventarc) Budget Notifier CF → Firestore + FCM
                    Firestore (onSnapshot) → Dashboard live updates
```

### Services (`terraform_v2/services/`)

| Service | Type | Trigger | Key Logic |
|---|---|---|---|
| **frame-extractor** | Cloud Run container | Manual | OpenCV frame capture → base64 → Pub/Sub `frames-in` |
| **dispatcher** | Cloud Function (Gen2) | Eventarc `frames-in` | Auth token management → HTTP POST to Vertex AI endpoint |
| **judge** | Vertex AI container (PyTorch 2.2 + CUDA 12.1) | HTTP POST `/predict` | YOLOv8x inference → Pub/Sub `detections-out`; health at `/healthz` |
| **alert-manager** | Cloud Function (Gen2) | Eventarc `detections-out` | Email cooldown (60s/camera), Firestore writes, FCM push |
| **budget-notifier** | Cloud Function (Gen2) | Eventarc `budget-notifications` | Firestore write + FCM push |
| **mediamtx** | Raspberry Pi container | - | RTSP in → WebRTC WHEP out on port 8889 |

### Detection Classes & Severity

- **High-severity** (trigger email + FCM): `spaghetti`, `not_sticking`, `layer_shift`, `warping`
- **Low-severity** (Firestore only): `stringing`, `under_extrusion`, `over_extrusion`, `nozzle_clog`, `foreign_object`
- **Confidence threshold:** 0.35 (configurable via Terraform variable)

### Infrastructure (`terraform_v2/terraform/`)

All GCP resources are defined in Terraform (`main.tf`). Key resources: Pub/Sub topics, Firestore (eur3, NATIVE mode), Cloud Functions, Artifact Registry, GCS model bucket, Eventarc triggers, service accounts, and budget alerts.

## Testing Strategy

All 4 test files in `tests/` mock heavy dependencies (cv2, numpy, ultralytics, google.cloud, firebase_admin) at the `sys.modules` level before import — no real credentials or network calls are needed. The `@functions_framework.cloud_event` decorator is identity-wrapped to allow plain function calls in tests. 90% coverage is enforced by both pre-commit hooks and GitHub Actions.

## CI/CD (`.github/workflows/`)

- **`python-tests.yml`**: Triggers on changes to `terraform_v2/services/` or `tests/`; runs pytest with 90% coverage threshold; comments results on PRs.
- **`terraform.yml`**: Triggers on changes to `terraform_v2/`; lints, security-scans, plans on PR, and applies on merge to main.
- **`docker-judge.yml`**: Triggers on changes to `terraform_v2/services/judge/`; downloads `best.pt` from GCS, builds and pushes to Artifact Registry.
- **`compileLatex.yml`**: Compiles thesis PDF on changes to `docs/latex/`.

Required GitHub secrets: `GCP_SA_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`.
