---
title: 12 — Glossary & Reference
tags: [glossary, reference, ids, env-vars, project/printermonitor]
aliases: [glossary, reference, cheat sheet]
type: reference
---

# Glossary & Reference

Look up a term, an ID, an env var, or a magic number.

## Project identifiers

| Name | Value |
|---|---|
| GCP project ID | `printermonitor-488112` |
| GCP project number | (look up via `gcloud projects describe printermonitor-488112 --format='value(projectNumber)'`) |
| Primary region | `europe-west1` (Belgium) |
| Firestore location | `eur3` (multi-region Europe) |
| Artifact Registry repo | `europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor` |
| Model GCS bucket | `printermonitor-488112-models` (model weights, with versioning) |
| Source GCS bucket | `printermonitor-488112-functions-source` (Cloud Function zips + Terraform state) |
| Vertex AI endpoint ID | `6900414029643120640` |
| Vertex AI endpoint resource | `projects/printermonitor-488112/locations/europe-west1/endpoints/6900414029643120640` |
| Billing budget ID | `695e9489-62cb-4e37-82cc-2d0e1ed4c8e0` (manual, not Terraformed) |
| Firebase web app ID | `1:895714392909:web:553314befa3149f6ad1504` |
| Firebase messaging sender ID | `895714392909` |
| Dashboard URL | https://printermonitor-488112.web.app |

## Service accounts

| Name | Purpose | Where it lives |
|---|---|---|
| `sa-frame-extractor` | Pi → Pub/Sub publisher | Terraform |
| `sa-dispatcher` | dispatcher CF runtime | Terraform |
| `sa-alert-manager` | alert-manager + budget-notifier CF runtime | Terraform |
| `judge-svc` | Vertex AI custom container runtime | **manual** (see [[02-cloud-services#Judge]]) |
| `service-{N}@gcp-sa-pubsub.iam` | Pub/Sub service agent (Eventarc OIDC) | GCP-managed |
| `service-{N}@gcp-sa-aiplatform.iam` | Vertex AI service agent | GCP-managed |
| `service-{N}@gcp-sa-aiplatform-cc.iam` | Vertex AI custom-code service agent | GCP-managed |
| `{N}-compute@developer.gserviceaccount.com` | default compute SA (fallback for judge) | GCP-managed |

`{N}` = project number.

## Pub/Sub topics

| Topic | Producer | Consumer |
|---|---|---|
| `frames-in` | Pi `frame_extractor.py` | dispatcher CF (via Eventarc) |
| `detections-out` | judge container | alert-manager CF (via Eventarc); also dashboard reads downstream via Firestore |
| `budget-notifications` | GCP Cloud Billing | budget-notifier CF (via Eventarc) |

Retention: `frames-in` and `detections-out` are 1 hr; `budget-notifications` uses GCP default.

## Firestore collections

| Collection | Schema (per document) | Written by | Read by |
|---|---|---|---|
| `inferences` | `{ camera_id, detections: [...], timestamp, seq, frame_url, created_at: SERVER_TIMESTAMP }` | `handle_detection` (every frame, incl. 0 detections) | dashboard page 2 (`limit(100)`); public-read |
| `alerts` | `{ camera_id, detections: [...], timestamp, seq, created_at: SERVER_TIMESTAMP }` | `handle_detection` (conf ≥ threshold only) | dashboard page 1 (`limit(50)`); public-read |
| `alert_cooldowns` | doc id is the cooldown key (`global_email` or `budget`); body `{ last_sent: ISO timestamp }` | `set_cooldown` in alert-manager | alert-manager only |
| `budget_alerts` | `{ budget_name, cost_amount, threshold, created_at: SERVER_TIMESTAMP }` | `handle_budget_alert` in alert-manager | — |
| `system_state` | doc `extraction` = `{ enabled: bool }` | dashboard capture toggle (`setDoc`) | Pi `frame_extractor.py` (polled, 5 s cache); public read+write |

`firestore.rules`: `alerts` & `inferences` are `read:true, write:false`; `system_state` is `read,write:true`; the rest have no public rule.

Index: `alerts (camera_id ASC, timestamp DESC)` defined in Terraform; the dashboard actually queries by `created_at` and doesn't use this index.

## Env vars (per service)

### `frame-extractor` (Cloud Run / Pi)
| Var | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | `printermonitor-488112` | |
| `FRAMES_TOPIC` (Cloud Run) / `PUBSUB_TOPIC` (Pi) | `frames-in` | |
| `CAPTURE_FPS` | `2` (Cloud Run) / `0.1` (Pi) | Pi runs intentionally slower |
| `JPEG_QUALITY` | `70` (Cloud Run) / **`100` (Pi)** | Pi bumped to 100 for sharper inference-log thumbnails |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | `1280` / `720` | resize cap, not target |
| `RTSP_URLS` / `RTSP_URL` (Cloud Run) | (required) | comma-separated |
| `CAMERA_IDS` (Cloud Run) | `cam1,cam2,cam3` | pads with `camN` if shorter than URLs |
| `PORT` | `8080` | health server |
| `GOOGLE_APPLICATION_CREDENTIALS` | (Pi only) | path to SA key file (now also needs Firestore read for the capture toggle) |

> The Pi extractor also reads Firestore `system_state/extraction.enabled` (no env var — hard-coded collection/doc) to pause/resume capture.

### `dispatcher`
| Var | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | `printermonitor-488112` | |
| `VERTEX_ENDPOINT_ID` | (required) | from Terraform `var.vertex_endpoint_id` |
| `VERTEX_REGION` | `europe-west1` | |
| `MAX_FRAME_AGE_S` | `5` | drop frames whose `ts` is older than this (backlog staleness filter) |

### `judge`
| Var | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | (required) | |
| `DETECTIONS_TOPIC` | (required) | usually `detections-out` |
| `MODEL_PATH` | `/app/best.pt` | baked into image |
| `CONF_THRESHOLD` | `0.35` (code) / set via `gcloud ai models upload --container-env-vars` | |
| `STREAK_REQUIRED` | `2` | consecutive frames before a detection is "confirmed" |
| `FRAMES_BUCKET` | `""` (code) / `printermonitor-488112-frames` (deployed) | enables per-inference JPEG upload + `frame_url`; empty = upload skipped |
| `JPEG_QUALITY` | `60` | quality of the uploaded inference JPEG |
| `PORT` | `8080` | |

### `alert-manager` (and `budget-notifier`, same source)
| Var | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | `printermonitor-488112` | |
| `FIRESTORE_COLLECTION` | `alerts` | |
| `CONF_THRESHOLD` | **`0.20` (code default)** / `0.35` (Terraform var, injected into the alert-manager fn env) | budget-notifier fn does **not** set it (doesn't use it) |
| `COOLDOWN_SECONDS` | `300` | 5 min |
| `GMAIL_ADDRESS` | (required, sensitive) | passed from `var.gmail_address` |
| `GMAIL_APP_PASSWORD` | (required, sensitive) | passed from `var.gmail_app_password` |

## Magic numbers

| Number | Meaning | Source |
|---|---|---|
| `0.1` | Capture fps on Pi (frame every 10 s) | `pi_codes/frame_extractor.py:14` |
| `0.35` | Judge inference conf threshold (in container) | `judge/main.py:17` |
| `0.35` | Conf threshold (judge env + Terraform var). **alert-manager *code* default is `0.20`**, overridden to `0.35` by Terraform | `variables.tf:23`, `judge/main.py:17`, `alert-manager/main.py:16` |
| `2` | Streak filter (frames required to "confirm") | `judge/main.py:18` |
| `5` | Dispatcher staleness drop (`MAX_FRAME_AGE_S`, seconds) | `dispatcher/main.py:21` |
| `300` | Email cooldown seconds | `alert-manager/main.py:17` |
| `60` | Local-test cooldown seconds | `local_alert_handler.py:19` |
| `5` | Budget threshold ($/month) | manual GCP Console |
| `~$37` | Daily cost of deployed T4 GPU | observed |
| `8554/8080/8888/8889/8890` | MediaMTX ports (RTSP/RTMP/HLS/WebRTC/SRT) | `mediamtx.yml` |
| `1280 × 720` | Default frame size cap | env defaults |
| `100` | Pi JPEG quality (was 70) | `pi_codes/frame_extractor.py:15` |
| `60` | Judge uploaded-frame JPEG quality | `judge/main.py:20` |
| `100` | Inference-log query limit (page 2) | `dashboard/index.html:1086` |
| `5.0` | Capture kill-switch poll cache (seconds) | `pi_codes/frame_extractor.py:44` |
| `10_000` | Dashboard bbox TTL (ms, clear stale overlays) | `dashboard/index.html` (`DETECTION_TTL_MS`) |
| `50` | Dashboard event log limit (`alerts` query, page 1) | `dashboard/index.html:1021` |
| `90` | Test coverage gate (percent) | `python-tests.yml`, `.pre-commit-config.yaml` |

## Detection class labels (exact spelling)

```python
[
  "spagetti",                       # HIGH_SEV — one 'h', not 'spaghetti'
  "not_sticking",                   # HIGH_SEV
  "layer_shift",                    # HIGH_SEV
  "warping",                        # HIGH_SEV
  "stringing",
  "under_extrusion",
  "over_extrusion",
  "nozzle_clog",
  "foreign_object_on_print_area",
]
```

Source of truth: `scripts/annotate.py:20-30` `CLASS_NAMES` (must match the YOLO model's `model.names` dict).

## GCP roles in play

| Role | Purpose |
|---|---|
| `roles/aiplatform.user` | call Vertex AI endpoints |
| `roles/eventarc.eventReceiver` | be the target of an Eventarc trigger |
| `roles/pubsub.publisher` | publish messages |
| `roles/pubsub.subscriber` | consume messages |
| `roles/datastore.user` | read/write Firestore docs |
| `roles/firebase.admin` | broad Firebase ops (currently unused after FCM removal from alert-manager) |
| `roles/iam.serviceAccountTokenCreator` | mint OIDC tokens (for Pub/Sub to call Eventarc push) |
| `roles/iam.serviceAccountUser` | impersonate / "actAs" another SA (Vertex AI runs containers as `judge-svc`) |
| `roles/run.invoker` | invoke a Cloud Run service (Cloud Functions Gen2 are Cloud Run under the hood) |
| `roles/billing.costsManager` | (not granted) — would let Terraform manage budgets |

## Common abbreviations

| Term | Meaning |
|---|---|
| **ADC** | Application Default Credentials (`gcloud auth application-default login`) |
| **CF** | Cloud Function |
| **CMEK** | Customer-Managed Encryption Keys (we skip — Google-managed is fine for thesis) |
| **FCM** | Firebase Cloud Messaging |
| **mAP** | mean Average Precision (object detection metric) |
| **MOC** | Map of Content (Obsidian convention; see [[00-index]]) |
| **OIDC** | OpenID Connect (the token format Pub/Sub uses for push targets) |
| **SA** | Service Account |
| **WHEP** | WebRTC-HTTP Egress Protocol (subscriber side; what the dashboard speaks) |
| **WHIP** | WebRTC-HTTP Ingest Protocol (publisher side; not used here — GStreamer uses RTSP push instead) |
| **YOLO** | You Only Look Once (object detection model family) |

## Drift between docs and source

Things `README.MD` / `documentation.md` / `CLAUDE.md` get wrong, source-of-truth in parens:

- **Class spelling**: docs say "spaghetti"; model uses `spagetti` (source: `annotate.py`, `alert-manager/main.py:22`).
- **Streak threshold**: CLAUDE.md says 3; judge defaults to 2 (`judge/main.py:18`).
- **Cooldown granularity**: CLAUDE.md says per-camera, 60s; production uses single global key `global_email`, 300s (`alert-manager/main.py:17, 118`).
- **frame-extractor location**: docs imply Cloud Run; in current deployment the Pi runs `pi_codes/frame_extractor.py`. The Cloud Run image exists but isn't deployed.
- **budget-notifier code**: `services/budget-notifier/main.py` (FCM-based) is dead code. Terraform deploys `services/alert-manager/` for both functions with different entry points.
- **Confidence threshold**: judge is explicitly set to 0.35 via `gcloud ai models upload --container-env-vars`; Terraform var default is 0.35; **alert-manager's *code* default is `0.20`** (`alert-manager/main.py:16`) but the deployed function gets `0.35` from the Terraform-injected env. Effective production value: 0.35 everywhere. (The old glossary text claiming the alert-manager code default was 0.35 was wrong.)
- **Class count**: README says 10 in one place; the actual class count is **9** (see list above).
- **`firebase-admin` in alert-manager requirements.txt**: imported only in the dead `budget-notifier/main.py`; the deployed `alert-manager/main.py` doesn't use it. Removing it from requirements.txt would shrink the cold start.

### New in the current build (was not in older docs)
- **Judge always publishes** — every inference (incl. zero detections) goes to `detections-out`, not just defect frames (`judge/main.py:159-161`).
- **`inferences` collection** — alert-manager logs every inference; `alerts` is the conf-filtered subset (`alert-manager/main.py:93-114`).
- **GCS frame upload + `frame_url`** — judge writes a JPEG per inference to the public `*-frames` bucket (`judge/main.py:55-67`).
- **Dispatcher staleness + 404/503 drop** — `MAX_FRAME_AGE_S=5`, and endpoint-not-ready frames are dropped not retried (`dispatcher/main.py:21, 59-66, 88-94`).
- **Pi capture kill-switch** — `system_state/extraction.enabled` polled by the Pi, toggled from the dashboard (`pi_codes/frame_extractor.py:40-52`).
- **Permanent Cloudflare host** — `cam.printermonitor.app` (dashboard fallback `index.html:739`); the old ephemeral `*.trycloudflare.com` fallback is gone.
- **Dashboard is two pages** — Live streams + Inference log (`dashboard/index.html:518-521`).

When in doubt, trust the code in `terraform_v2/services/` and `pi_codes/`, not the markdown files at the repo root.
