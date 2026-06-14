---
title: 05 — Infrastructure
tags: [infra, terraform, gcp, iam, pubsub, firestore, eventarc, project/printermonitor]
aliases: [terraform, IAM, GCP resources]
type: infra
---

# Infrastructure (Terraform / GCP)

Everything in `terraform_v2/terraform/` — single root module, GCS-backed state.

## Layout

```
terraform_v2/terraform/
├── main.tf       — all resources (~700 lines, single file; includes DLQ and monitoring resources)
├── variables.tf  — 8 inputs (project_id, region, model_gcs_bucket, conf_threshold,
│                   cloudflare_tunnel_hostname, vertex_endpoint_id,
│                   gmail_address [sensitive], gmail_app_password [sensitive])
├── outputs.tf    — endpoint id, WHEP URL, topic names, dashboard host helper
└── terraform.tfstate{,.backup}  — local copy of the GCS-backed state (do not edit)
```

## Backend

```hcl
backend "gcs" {
  bucket = "printermonitor-488112-functions-source"
  prefix = "terraform/state"
}
```

The state bucket is the same bucket used for Cloud Functions source archives — convenient (one bucket to permission), unusual (state and source mixed). Don't `force_destroy` it.

## Providers

`hashicorp/google ~> 5.0` · `hashicorp/google-beta ~> 5.0` (unused at present) · `hashicorp/null ~> 3.0` · `hashicorp/archive ~> 2.0`. Terraform `>= 1.6`.

## Enabled APIs (managed by `google_project_service.apis`, for_each)

`run` · `pubsub` · `firestore` · `aiplatform` · `cloudfunctions` · `cloudbuild` · `artifactregistry` · `eventarc` · `storage` · `firebase` · `firebaseapphosting` · `iam` · `logging` · `billingbudgets` · `firebaseinstallations` · `fcm` · `fcmregistrations`

The last three are required for Web Push token registration — without them you get `messaging/token-subscribe-failed` in the browser.

`disable_on_destroy = false` everywhere — safer; doesn't break unrelated workloads on a teardown.

## Resources by category

### Storage
- `google_storage_bucket.models` — `printermonitor-488112-models` (regional, uniform-bucket-level-access, public-access-prevention=`enforced`, versioning **on**, `force_destroy=false`). Holds `yolov8x/best.pt`.
- `google_storage_bucket.functions_source` — `printermonitor-488112-functions-source`. Holds the zipped Cloud Function source. `force_destroy=true` (safe — content is regenerable).
- `google_storage_bucket.frames` — **`printermonitor-488112-frames`** (`main.tf:76`). **Public-read** inference-frame store: the judge writes one JPEG per inference here and the dashboard's [[06-dashboard|Inference log]] renders them. `public-access-prevention` is *not* enforced — `google_storage_bucket_iam_member.frames_public_read` grants `roles/storage.objectViewer` to `allUsers` (checkov `CKV_GCP_28` is explicitly skipped, "frames are non-sensitive"). Write access: `frames_judge_svc_write` (the `judge-svc` runtime SA) and `frames_compute_sa_write` (default compute SA fallback, mirroring the detections-out publisher fallback).
  > **No lifecycle rule** — objects accumulate forever (one per inference, every frame). A future TTL/lifecycle policy would cap storage; see [[feedback]].

### Artifact Registry
- `google_artifact_registry_repository.printermonitor` — Docker repo at `europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/`. Two images live here: `judge` (built by CI) and conceptually `frame-extractor` / `mediamtx` if you ever deploy them.

### Pub/Sub
Three topics, all with `message_retention_duration = "3600s"` (the budget topic lacks an explicit retention — uses GCP default).

| Topic | Producer | Consumer | Notes |
|---|---|---|---|
| `frames-in` | Pi `frame_extractor.py` | Eventarc → `dispatcher` CF | 1 hr retention |
| `detections-out` | Judge container | Eventarc → `alert-manager` CF; dashboard reads via Firestore | |
| `budget-notifications` | GCP Cloud Billing budget | Eventarc → `budget-notifier` CF (= `alert-manager` source w/ different entry point) | |

Single subscription declared in Terraform: `budget-notifications-sub` (60s ack deadline). The Eventarc-managed subscriptions for the topics above are created automatically as part of the Cloud Function trigger.

### Dead-letter queues (DLQ)

Two DLQ topics are managed by Terraform:
- `frames-in-dead-letters` — receives poison-pill frames that exhaust delivery attempts
- `detections-out-dead-letters` — receives undeliverable detection events

Each has a corresponding pull subscription (`-sub` suffix) with 7-day retention for post-mortem inspection.

The Eventarc-managed subscriptions (e.g. `eventarc-europe-west1-dispatcher-635712-sub-152`) cannot have `dead_letter_policy` set declaratively in Terraform because Eventarc owns those subscriptions. Instead, two `null_resource` blocks with `local-exec` provisioners run `gcloud pubsub subscriptions update --dead-letter-topic=... --max-delivery-attempts=5` after the DLQ topics are created. They re-run only when the DLQ topic ID changes (tracked via `triggers`).

IAM: the Pub/Sub service agent (`service-{N}@gcp-sa-pubsub.iam`) is granted `pubsub.publisher` on both DLQ topics so it can forward dead letters.

### Firestore
- `google_firestore_database.default` — name `(default)`, location `eur3` (multi-region Europe), type `FIRESTORE_NATIVE`. **`prevent_destroy = true`** and `ignore_changes = all`. Once created, Terraform leaves it alone.
- Index: `alerts (camera_id ASC, timestamp DESC)`. This is what the dashboard query relies on (it orders by `created_at` though — the index is essentially unused at present; legacy).
- Collections used at runtime (not declared in Terraform, created on first write):
  - `inferences` — **one doc per inference** (every frame the judge processed, incl. zero-detection). Written by `handle_detection`. Drives the dashboard inference log + frame tiles. Public-read, write-denied via `firestore.rules`.
  - `alerts` — confidence-filtered subset (defects only). Public-read, write-denied.
  - `alert_cooldowns` — email cooldown locks (`global_email`, `budget`). Not in `firestore.rules` (server-only access path).
  - `budget_alerts` — budget alert audit log.
  - `system_state` — single doc `extraction` with field `enabled` (the [[03-pi-edge#Remote capture kill-switch|capture kill-switch]]). **`firestore.rules` allows public read+write** so the unauthenticated dashboard can toggle it.
- **`firestore.rules`** (deployed via `firebase-deploy.yml`): `alerts` & `inferences` → `read: true, write: false`; `system_state` → `read, write: true`.

### Service Accounts
- `sa-frame-extractor` — runs on the Pi
- `sa-alert-manager` — runs the AlertManager CF and the BudgetNotifier CF
- `sa-dispatcher` — runs the Dispatcher CF
- `judge-svc` — **manually created** (not in Terraform). Required for the Vertex AI custom container to publish to Pub/Sub **and** to write objects into the `*-frames` bucket. The `gcloud ai endpoints deploy-model --service-account=judge-svc@...` flag passes it in. Terraform references it by email in `google_storage_bucket_iam_member.frames_judge_svc_write` and `google_pubsub_topic_iam_member.vertex_ai_detections_publisher`, so it must exist before `terraform apply`.

### Cloud Functions (Gen2)

All three are `google_cloudfunctions2_function`, Python 3.12 runtime, `ingress_settings = ALLOW_INTERNAL_ONLY`.

| Function | Entry point | Source dir | Memory | Timeout | Max instances | Retry |
|---|---|---|---|---|---|---|
| `dispatcher` | `dispatch_frame` | `services/dispatcher` | 512M | 60 s | 10 | RETRY |
| `alert-manager` | `handle_detection` | `services/alert-manager` | 256M | 120 s | 10 | RETRY |
| `budget-notifier` | `handle_budget_alert` | `services/alert-manager` ⟵ same source bundle | 256M | 60 s | 3 | DO_NOT_RETRY |

Each source dir is `archive_file`-zipped, uploaded as `<fn>-<md5>.zip` to the functions-source bucket. Re-zip + redeploy is automatic on source change because the object name includes the MD5.

### IAM bindings

Full picture; consolidated from `main.tf:176-220, 392-465, 527-548`:

```
sa-frame-extractor
  ↳ pubsub.publisher    on topic frames-in
  ↳ aiplatform.user     project-wide (legacy — predates dispatcher)

sa-dispatcher
  ↳ aiplatform.user     project-wide
  ↳ eventarc.eventReceiver
  ↳ pubsub.subscriber
  ↳ pubsub.publisher    on detections-out (legacy; judge does the actual publish)
  ↳ run.invoker         on the dispatcher's own Cloud Run service (self-invoke for Eventarc)

sa-alert-manager
  ↳ datastore.user
  ↳ firebase.admin       (legacy / unused — FCM not in current alert-manager)
  ↳ eventarc.eventReceiver
  ↳ pubsub.subscriber
  ↳ run.invoker          on alert-manager *and* budget-notifier Cloud Run services

service-{N}@gcp-sa-pubsub.iam (built-in Pub/Sub service agent)
  ↳ iam.serviceAccountTokenCreator on project
     (lets Pub/Sub mint OIDC tokens to call Eventarc push targets)

service-{N}@gcp-sa-aiplatform.iam (Vertex AI service agent)
  ↳ pubsub.publisher    on detections-out
  ↳ iam.serviceAccountUser on judge-svc
     (so Vertex AI can run the judge container as judge-svc)

service-{N}@gcp-sa-aiplatform-cc.iam (Vertex AI custom-code service agent)
  ↳ pubsub.publisher    on detections-out

{N}-compute@developer.gserviceaccount.com  (default compute SA)
  ↳ pubsub.publisher    on detections-out
     (fallback when Vertex AI uses the default compute SA instead of judge-svc)
```

`{N}` is the numeric project number — looked up via `data "google_project" "project"` in main.tf.

### Resources NOT managed by Terraform (intentional)

- **Vertex AI endpoint** (`6900414029643120640`) and **model versions / deployments** — managed via `gcloud ai endpoints deploy-model …` ([[09-deployment-ops]]). `var.vertex_endpoint_id` references it but doesn't manage it.
- **`judge-svc` service account** — created manually because Terraform needs it to exist *before* it grants Vertex AI's service agent `roles/iam.serviceAccountUser` on it; chicken-and-egg.
- **Billing budget** — manual via GCP Console. The CI service account lacks `billingAccounts.budgets` permission (`main.tf:321-323` says how to bring it back into TF).
- **Firebase Web App config** — the API keys etc. are hard-coded in `dashboard/index.html`; not generated by Terraform.
- **MediaMTX container** — Dockerfile in repo, but runs on the Pi (not deployed to Cloud Run).

## Outputs

Listed in `outputs.tf`. Useful ones:
- `mediamtx_whep_url` — `https://<cloudflare-host>/whep`
- `vertex_endpoint_id` — passes through `var.vertex_endpoint_id`
- `frames_topic`, `detections_topic`, `firestore_database` — names for scripts
- `artifact_registry` — image path prefix
- `dashboard_host_update_command` — `sed` one-liner to bake the Cloudflare host into `dashboard/index.html`

## Locals

```hcl
locals {
  mediamtx_host = var.cloudflare_tunnel_hostname != ""
    ? var.cloudflare_tunnel_hostname
    : "PENDING-SET-cloudflare_tunnel_hostname-variable"
}
```

The placeholder string is intentionally ugly so it stands out in outputs.

## Terraform commands

```bash
cd terraform_v2/terraform
terraform init
terraform plan -var="gmail_address=…" -var="gmail_app_password=…"
terraform apply
```

CI runs apply automatically on merge to `main` — see [[08-ci-cd#terraform.yml]].

## Lint & security

- `terraform fmt -check -recursive` — formatting
- `tflint` with `terraform-linters/tflint-ruleset-google v0.30.0` (config `.tflint.hcl`)
- `checkov` with `.checkov.yaml` skips:
  - `CKV_GCP_29` — Google-managed encryption (no CMEK required for thesis)
  - `CKV_GCP_62` — bucket logging
  - `CKV_SECRET_4` — Firebase config false positive
  - `CKV_DOCKER_7` — `:latest` for dev convenience
  - `CKV_GCP_78` — Cloud Function unauthenticated (Eventarc handles auth)

Inline `#checkov:skip=…` comments cover Pub/Sub CMEK and SA-token-creator cases.

See [[02-cloud-services#Service interconnects]] for an IAM summary table, and [[09-deployment-ops]] for the manual gcloud bits.
