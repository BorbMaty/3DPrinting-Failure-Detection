# 3D Printing Failure Detection

Real-time 3D printer failure detection system using YOLOv8x on Google Cloud Vertex AI. Three cameras stream via WebRTC from a Raspberry Pi, while frames are analyzed for defects and alerts are pushed to a live dashboard and email.

## How it works

```
Raspberry Pi (3 cameras)
  ├── MediaMTX ──► WebRTC ──► Cloudflare Tunnel ──► Dashboard (browser)
  └── frame_extractor.py ──► Pub/Sub: frames-in
                                    │
                              Dispatcher (Cloud Function)
                                    │
                              Vertex AI (YOLOv8x on T4 GPU)
                                    │
                              Pub/Sub: detections-out
                                    │
                              AlertManager (Cloud Function)
                                    ├── Firestore (alerts collection)
                                    └── Email notification (Gmail)
```

A Raspberry Pi captures frames from three USB cameras and publishes them to Google Cloud Pub/Sub. A Cloud Function dispatches each frame to a Vertex AI endpoint running a fine-tuned YOLOv8x model inside a custom container. The model runs inference on a T4 GPU and publishes detection results back to Pub/Sub. A second Cloud Function picks up detections, writes them to Firestore, and sends email alerts for high-severity defects. The browser dashboard shows live WebRTC streams alongside real-time alerts from Firestore.

## Detection classes

The model detects 10 failure types with a mean average precision of 0.904 (mAP@0.5):

| Class | mAP@0.5 | Email alert |
|-------|---------|-------------|
| Spaghetti | 0.942 | Yes |
| Not sticking | 0.985 | Yes |
| Layer shift | 0.934 | Yes |
| Warping | 0.873 | Yes |
| Stringing | 0.971 | No |
| Under-extrusion | 0.883 | No |
| Over-extrusion | 0.812 | No |
| Nozzle clog | 0.821 | No |
| Foreign object on print area | 0.918 | No |

## Dashboard

The dashboard runs on Firebase App Hosting and displays:

- Live WebRTC streams from all three cameras via WHEP
- Real-time detection alerts from Firestore
- Camera connection status

## Tech stack

**Edge (Raspberry Pi)**
- MediaMTX — RTSP to WebRTC media server
- OpenCV — frame capture and encoding
- Cloudflare Tunnel — exposes WebRTC to the internet

**Cloud (Google Cloud Platform)**
- Vertex AI — model serving with T4 GPU
- Cloud Functions (Gen2) — serverless event processing
- Pub/Sub — message broker between services
- Firestore — real-time alert storage
- Artifact Registry — Docker container images
- Firebase App Hosting — dashboard

**ML**
- YOLOv8x — fine-tuned on 10 3D printing defect classes
- PyTorch 2.2 + CUDA 12.1

**Infrastructure**
- Terraform — all cloud resources as code
- GitHub Actions — CI/CD for Terraform and Docker builds
- Checkov — security scanning
- tflint — Terraform linting

## Project structure

```
├── .github/workflows/
│   ├── terraform.yml              # Lint → Plan → Apply
│   ├── docker-judge.yml           # Build + push judge container
│   └── compileLatex.yml           # Compile thesis PDF
├── terraform_v2/
│   ├── terraform/
│   │   ├── main.tf                # All GCP resources
│   │   ├── variables.tf           # Input variables
│   │   └── outputs.tf             # Output values
│   └── services/
│       ├── alert-manager/         # Detection + budget alert handlers
│       ├── dispatcher/            # Frames → Vertex AI relay
│       ├── judge/                 # YOLOv8x inference container
│       ├── frame-extractor/       # Dockerfile (runs on Pi)
│       └── mediamtx/              # Dockerfile + config (runs on Pi)
├── docs/latex/                    # Thesis (LaTeX)
└── README.md
```

## Setup

### Prerequisites

- GCP project with billing enabled
- Raspberry Pi with cameras
- Terraform >= 1.6
- gcloud CLI authenticated
- GitHub secrets: `GCP_SA_KEY`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`

### Deploy infrastructure

```bash
cd terraform_v2/terraform
terraform init
terraform plan -var="gmail_address=YOU@gmail.com" -var="gmail_app_password=YOUR_APP_PASSWORD"
terraform apply
```

### Deploy the model

```bash
# Upload model weights
gsutil cp best.pt gs://printermonitor-488112-models/yolov8x/best.pt

# Build and push judge container (or let GitHub Actions do it)
docker build -t europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:v2 \
  terraform_v2/services/judge/
docker push europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:v2

# Deploy to Vertex AI
gcloud ai endpoints deploy-model 6900414029643120640 \
  --model=5624118524306784256 \
  --display-name=yolov8x-v3 \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --region=europe-west1
```

### Start the Pi

```bash
# Start MediaMTX
./mediamtx &

# Start Cloudflare tunnel
cloudflared tunnel --url http://localhost:8889

# Start frame extractor
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-frame-extractor-key.json
python3 frame_extractor.py
```

### Stop GPU charges

```bash
gcloud ai endpoints undeploy-model 6900414029643120640 \
  --deployed-model-id=<ID> \
  --region=europe-west1
```

## Cost

The system costs pennies when the GPU is not deployed. With the T4 GPU active, expect ~$37/day. A $5 budget alert is configured in GCP. Always undeploy the model after testing.

## License

This project is part of a thesis at Sapientia Hungarian University of Transylvania.