---
title: 02 — Cloud Services
tags: [services, gcp, cloud-functions, vertex-ai, project/printermonitor]
aliases: [services, cloud functions, judge, dispatcher, alert manager]
type: services
---

# Cloud Services

All Python services live in `terraform_v2/services/`. Four runtime targets:
- 3× Cloud Functions Gen2 (Python 3.12)
- 1× Vertex AI custom container (PyTorch 2.2 + CUDA 12.1)

> Note: `services/frame-extractor/` and `services/mediamtx/` describe Pi-side containers, covered in [[03-pi-edge]].

---

## Dispatcher

**File:** `terraform_v2/services/dispatcher/main.py` (78 lines)
**Trigger:** Eventarc on Pub/Sub topic `frames-in`
**Entry point:** `dispatch_frame` (decorated with `@functions_framework.cloud_event`)
**Service account:** `sa-dispatcher`
**Memory / timeout / max-instances:** 512M / 60 s / 10
**Env vars:** `GCP_PROJECT`, `VERTEX_ENDPOINT_ID`, `VERTEX_REGION`

### What it does
1. Decodes the Pub/Sub envelope (`cloud_event.data["message"]["data"]` → base64 → JSON).
2. Builds a Vertex AI `instances` payload.
3. Refreshes ADC bearer token if `_creds.valid` is False.
4. `requests.post(VERTEX_URL, ...)` with 60 s timeout.
5. On `requests.exceptions.RequestException`, **re-raises** so Pub/Sub retries.

### Why it exists
Vertex AI endpoints require Bearer auth — Pub/Sub can't speak that natively. The dispatcher is a thin auth bridge. It does **not** itself touch Firestore or detections-out (the judge publishes detections directly).

### Dependencies
`functions-framework==3.*` · `requests==2.*` · `google-auth==2.*`

---

## Judge

**File:** `terraform_v2/services/judge/main.py` (163 lines)
**Trigger:** HTTP POST `/predict` from Vertex AI custom-container interface
**Runtime:** Vertex AI Custom Prediction Container on `n1-standard-4` + 1× NVIDIA Tesla T4
**Image:** `europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest`
**Service account:** `judge-svc` (manually created, not in Terraform — required for `pubsub.publisher` to `detections-out`)
**Endpoint ID:** `6900414029643120640`
**Env vars:** `GCP_PROJECT`, `DETECTIONS_TOPIC`, `MODEL_PATH=/app/best.pt`, `CONF_THRESHOLD=0.35`, `STREAK_REQUIRED=2`

### Surface
- `GET /healthz` · `/health` · `/` → `200 ok` (Vertex AI liveness probe)
- `POST /predict` → accepts **both** envelope formats:
  - Vertex AI: `{"instances": [{"data_b64": "...", "camera_id": "cam1", "seq": 42}]}`
  - Pub/Sub push: `{"message": {"data": "<b64 of inner JSON>"}}`
- Returns `{"predictions": [<detection payload>]}`

### Inference pipeline
1. `_safe_b64decode` — accepts unpadded base64, strips whitespace, raises on empty.
2. `cv2.imdecode` → BGR `numpy` array.
3. `model(img, conf=CONF_THRESHOLD, verbose=False)` (Ultralytics YOLO).
4. Build detection dicts with both pixel `bbox` and normalized `x/y/w/h`.
5. **Streak filter** (`judge/main.py:111-123`): per-camera dict of `{label: consecutive_count}`. Increment if the label appears this frame; reset to 0 if absent. Only emit detections where the streak ≥ `STREAK_REQUIRED`.
6. If any confirmed detections, publish to `detections-out` (`publisher.publish(...).result()` — blocking; raises on failure → Vertex returns 500 → Pub/Sub retries dispatcher → re-fire).
7. Always return 200 with the prediction body (even when streak-filtered to zero — the response body just has `"detections": []`).

### Dockerfile
`pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` · adds `libglib2.0-0` + `libgl1` for OpenCV · bakes `best.pt` at `/app/best.pt` · runs as UID 1001.

### Dependencies
`ultralytics` · `opencv-python-headless` · `numpy<2.0` · `google-cloud-pubsub`

---

## Alert Manager

**File:** `terraform_v2/services/alert-manager/main.py` (160 lines)
**Trigger:** Eventarc on `detections-out`
**Entry point:** `handle_detection`
**Service account:** `sa-alert-manager`
**Memory / timeout / max-instances:** 256M / 120 s / 10
**Env vars:** `GCP_PROJECT`, `FIRESTORE_COLLECTION=alerts`, `CONF_THRESHOLD=0.20` (from Terraform var), `COOLDOWN_SECONDS=300`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`

### Behaviour
1. Decode the Pub/Sub envelope.
2. **Confidence filter**: drop any detection with `confidence < CONF_THRESHOLD` (`>=` is the passing comparison — `0.20` passes, `0.1999` does not).
3. If anything survives, `db.collection("alerts").add({...})` — `created_at` is `SERVER_TIMESTAMP`.
4. **HIGH_SEV check**: any of `{"spagetti", "not_sticking", "layer_shift", "warping"}`. If yes and **not on cooldown for the single key `global_email`** → render an HTML email and SMTP it.
5. `set_cooldown("global_email")` writes `{last_sent: now_iso()}` to `alert_cooldowns/global_email`.

### Cooldown semantics
- Single global key `global_email` — **not** per-camera. CLAUDE.md is wrong about this.
- `300 s` TTL (5 min). To unblock an email, flush the doc: `python scripts/flush_firestore.py --collections cooldowns`.
- `is_on_cooldown()` reads `alert_cooldowns/{key}`, parses `last_sent` ISO timestamp, compares to now.

### `human_time()` helper
Converts ISO to `Europe/Bucharest` local time formatted as `Mon 18 May 2026, 14:30:00`. Used in the email body subject + table caption. Falls back to the raw ISO string on parse failure.

### Email layout (subject + HTML body)
- Subject: `🚨 Printer Alert — {camera_id}`
- HTML table of `(class, confidence%)` rows, with a link to https://printermonitor-488112.web.app
- SMTP_SSL to `smtp.gmail.com:465`, app-password auth
- Sender == recipient == `GMAIL_ADDRESS` (self-mail)

### Dependencies
`functions-framework==3.8.1` · `google-cloud-firestore==2.19.0` · `firebase-admin==6.5.0`

> `firebase-admin` is in requirements.txt but the deployed `alert-manager/main.py` does not import it (no FCM in this version). It's a leftover from the FCM version that's stashed in `services/budget-notifier/main.py` (see below).

---

## Budget Notifier

**File deployed:** `terraform_v2/services/alert-manager/main.py` (same source bundle as Alert Manager — Terraform zips the same dir twice and just changes the entry point).
**Entry point:** `handle_budget_alert`
**Trigger:** Eventarc on `budget-notifications` topic (which Cloud Billing budget alerts publish to).
**Retry policy:** `RETRY_POLICY_DO_NOT_RETRY` (budget alerts shouldn't fan-out duplicates).

### What runs in production
`handle_budget_alert` in `alert-manager/main.py`: writes a `budget_alerts/{auto-id}` Firestore doc, then sends an HTML email via the same `send_email()` helper, gated by `is_on_cooldown("budget")`.

### Dead code warning
`terraform_v2/services/budget-notifier/main.py` (an FCM/push-notification-based version with `firebase-admin.messaging`) exists but is **not deployed**. The `data "archive_file" "budget_notifier_source"` block in `main.tf:328-334` points to `services/alert-manager` (the comment in `main.tf` confirms this). When making changes, edit `services/alert-manager/main.py` — not `budget-notifier/main.py`.

### Billing budget
Created **manually** via GCP Console because the CI service account lacks `billingAccounts.budgets` permissions. Budget id: `695e9489-62cb-4e37-82cc-2d0e1ed4c8e0`. See `main.tf:321-323`.

---

## Service interconnects (IAM cheat-sheet)

See [[05-infrastructure#IAM]] for the full list. Key bindings:

| Identity | Role | Resource |
|---|---|---|
| `sa-dispatcher` | `roles/aiplatform.user` | project |
| `sa-dispatcher` | `roles/eventarc.eventReceiver` | project |
| `sa-dispatcher` | `roles/pubsub.subscriber` | project |
| `sa-dispatcher` | `roles/pubsub.publisher` | `detections-out` (legacy, judge actually publishes) |
| `judge-svc` | `roles/pubsub.publisher` | `detections-out` (the working binding) |
| `service-{N}@gcp-sa-aiplatform.iam` | `roles/iam.serviceAccountUser` | `judge-svc` |
| `sa-alert-manager` | `roles/datastore.user`, `roles/firebase.admin`, `roles/eventarc.eventReceiver`, `roles/pubsub.subscriber` | project |
| `service-{N}@gcp-sa-pubsub.iam` | `roles/iam.serviceAccountTokenCreator` | project |
| `sa-frame-extractor` | `roles/pubsub.publisher` | `frames-in` |
| `sa-frame-extractor` | `roles/aiplatform.user` | project (legacy; not used after dispatcher arrived) |
