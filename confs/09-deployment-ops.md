---
title: 09 — Deployment & Operations
tags: [ops, deployment, runbook, vertex-ai, pubsub, firestore, project/printermonitor]
aliases: [runbook, deploy, ops]
type: ops
---

# Deployment & Operations

Runbooks for the parts that aren't fully automated. Bookmark this when something is broken.

## Cold start — full deploy from scratch

In order:

1. **Infrastructure** ([[05-infrastructure]])
   ```bash
   cd terraform_v2/terraform
   terraform init
   terraform plan -var="gmail_address=YOU@gmail.com" -var="gmail_app_password=APP_PWD"
   terraform apply
   ```
2. **Upload weights to GCS**
   ```bash
   gsutil cp terraform_v2/services/judge/best.pt \
     gs://printermonitor-488112-models/yolov8x/best.pt
   ```
3. **Create `judge-svc` service account** (manual; see [[02-cloud-services#Judge]]):
   ```bash
   gcloud iam service-accounts create judge-svc \
     --project=printermonitor-488112 \
     --display-name="Judge custom prediction container"
   ```
   *(The Terraform-managed `roles/pubsub.publisher` binding to `detections-out` and `roles/iam.serviceAccountUser` from the Vertex AI service agent require this SA to already exist — re-apply Terraform after creating it.)*
4. **Build & push judge image** — push to `main` after editing `services/judge/**` and let `docker-judge.yml` do it, **or** locally:
   ```bash
   gcloud auth configure-docker europe-west1-docker.pkg.dev
   docker build -t europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest \
     terraform_v2/services/judge/
   docker push  europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest
   ```
5. **Upload & deploy the model on Vertex AI** — see [Vertex AI deploy](#vertex-ai-deploy).
6. **Pi setup** — see [[03-pi-edge#Pi startup checklist]].
7. **Dashboard deploy** — `firebase deploy --only hosting` (manual; no CI).

---

## Vertex AI deploy

> **The endpoint is NOT managed by Terraform.** All commands below are manual gcloud.

### Upload a new model version

Each new image / weight bundle must be uploaded as a **new model version** before deploying. Increment `vN` in the display name.

```bash
gcloud ai models upload \
  --region=europe-west1 \
  --project=printermonitor-488112 \
  --display-name=yolov8x-printermonitor-vN \
  --container-image-uri=europe-west1-docker.pkg.dev/printermonitor-488112/printermonitor/judge:latest \
  --container-ports=8080 \
  --container-health-route=/healthz \
  --container-predict-route=/predict \
  --container-env-vars=GCP_PROJECT=printermonitor-488112,DETECTIONS_TOPIC=detections-out,MODEL_PATH=/app/best.pt,CONF_THRESHOLD=0.35,STREAK_REQUIRED=3
```

Note: `STREAK_REQUIRED=3` here overrides the `2` default baked into [[02-cloud-services#Judge]]. Use 2 for noisier setups, 3 if false positives are a problem.

### Deploy to the endpoint

```bash
gcloud ai endpoints deploy-model 6900414029643120640 \
  --region=europe-west1 \
  --project=printermonitor-488112 \
  --model=MODEL_ID \
  --display-name=judge \
  --machine-type=n1-standard-4 \
  --accelerator=count=1,type=nvidia-tesla-t4 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100 \
  --service-account=judge-svc@printermonitor-488112.iam.gserviceaccount.com
```

> **Critical**: `--service-account=judge-svc` is required. Without it the container runs as the default compute SA and may not have `pubsub.publisher` on `detections-out`. Terraform binds `pubsub.publisher` to the compute SA as a fallback, but it's brittle — always pass `judge-svc` explicitly.

Cold start: container takes ~30 s to load YOLOv8x. First few requests may time out (dispatcher's 60 s budget); Pub/Sub will retry.

### Get the currently-deployed model ID (needed to undeploy)

```bash
gcloud ai endpoints describe 6900414029643120640 \
  --region=europe-west1 \
  --project=printermonitor-488112 \
  --format="json(deployedModels)"
```

Extract `id` (a long numeric string).

### Undeploy (stop GPU billing)

```bash
gcloud ai endpoints undeploy-model 6900414029643120640 \
  --region=europe-west1 \
  --project=printermonitor-488112 \
  --deployed-model-id=DEPLOYED_MODEL_ID
```

T4 cost: **≈ $37 / day** ([[11-costs-and-monitoring]]). **Always undeploy after testing.**

---

## Purge `frames-in` backlog

> **Mostly automatic now.** The dispatcher drops frames older than `MAX_FRAME_AGE_S` (5 s) and drops on HTTP 404/503 (endpoint not ready) instead of retrying ([[02-cloud-services#Dispatcher]]). So a freshly-redeployed judge is no longer slammed by a stale queue — the backlog is acked-and-discarded as it's read. The manual seek below is now a **backstop** for an unusually large backlog, not a routine step.
>
> **Cleaner still:** flip the [[03-pi-edge#Remote capture kill-switch|capture kill-switch]] OFF from the dashboard ("Capture: OFF") before undeploying the judge. The Pi stops publishing, so no backlog forms at all. Turn it back ON after redeploying.

If the judge is undeployed while the Pi is publishing (and the capture toggle is left ON), `frames-in` accumulates frames. The dispatcher's staleness filter handles small backlogs automatically, but for a very large one you can still **seek the subscription past the backlog** so the queued messages are acked without delivery.

The Eventarc-managed subscription for the dispatcher has a generated name like `eventarc-europe-west1-dispatcher-635712-sub-152` (the suffix varies). To find the exact name:

```bash
gcloud pubsub subscriptions list \
  --project=printermonitor-488112 \
  --format="value(name)" | grep dispatcher
```

Then seek to "now":

```bash
gcloud pubsub subscriptions seek <SUB_NAME> \
  --time=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  --project=printermonitor-488112
```

This is a one-time skip; new messages flow normally afterwards.

---

## Flush Firestore

`scripts/flush_firestore.py`. Targets **four** collections (`COLLECTIONS` map, `scripts/flush_firestore.py:11-16`):

| Key | Collection | Why flush |
|---|---|---|
| `alerts` | `alerts` | Stale test data clutters the dashboard event log |
| `cooldowns` | `alert_cooldowns` | Email suppressed for 5 min — flush to re-fire immediately |
| `budget_alerts` | `budget_alerts` | Test budget pushes you don't want in history |
| `inferences` | `inferences` | Per-frame audit docs pile up fast (every frame, not just defects) — flush to keep the inference-log page and storage tidy |

> Note: this does **not** delete the JPEGs in the `*-frames` GCS bucket. Those are separate and have no lifecycle rule — `gsutil -m rm 'gs://printermonitor-488112-frames/frames/**'` to clear them manually.

```bash
# Flush all four
python scripts/flush_firestore.py

# Only cooldowns
python scripts/flush_firestore.py --collections cooldowns

# Only the inference audit trail
python scripts/flush_firestore.py --collections inferences

# Non-interactive
python scripts/flush_firestore.py --yes
```

Requires `gcloud auth application-default login` first so the script can pick up ADC. Batches deletes 100 docs at a time (`scripts/flush_firestore.py:19`).

---

## Common failure modes

### "Email suppressed by cooldown" (no alert email after a known failure)
- Check `alert_cooldowns/global_email` in Firestore → look at `last_sent`. If < 5 min ago, that's the cause.
- Fix: `python scripts/flush_firestore.py --collections cooldowns` (or wait 5 minutes).

### "Dashboard shows boxes but no email" (or vice versa)
- Boxes come from Firestore (low-sev included). Email is gated by HIGH_SEV. Check the detection class — if it's `stringing`/`under_extrusion`/etc., that's expected ([[04-ml-model#Severity tiers]]).

### "Judge keeps timing out on first request after deploy"
- Container is cold-starting. Pub/Sub retries with backoff. Wait ~60 s; the second wave succeeds.

### "Cloudflare tunnel hostname changed; dashboard shows blank tiles"
- Edit `dashboard/index.html:266` fallback hostname **and** redeploy dashboard.
- Or set `var.cloudflare_tunnel_hostname` in Terraform and use the `dashboard_host_update_command` output.

### "Pi can't publish: 401 Unauthorized"
- The SA key on the Pi has been rotated/revoked. Re-create:
  ```bash
  gcloud iam service-accounts keys create sa-frame-extractor-key.json \
    --iam-account=sa-frame-extractor@printermonitor-488112.iam.gserviceaccount.com
  scp sa-frame-extractor-key.json pi@…:/home/pi/
  ```
- Update `GOOGLE_APPLICATION_CREDENTIALS` env var if path changed.

### "Vertex AI deploy fails with 'service account does not exist'"
- `judge-svc` was never created or got deleted. Re-create (see [Cold start](#cold-start--full-deploy-from-scratch) step 3) and re-`terraform apply`.

### "Pub/Sub permission errors on detections-out"
- Vertex AI service agent isn't bound. Re-run `terraform apply`. The relevant resource is `google_pubsub_topic_iam_member.vertex_ai_detections_publisher` in `main.tf:436`.

---

## Rolling back a bad judge deploy

1. Get current and previous deployed model IDs:
   ```bash
   gcloud ai endpoints describe 6900414029643120640 \
     --region=europe-west1 --project=printermonitor-488112 \
     --format="json(deployedModels)"
   ```
2. Undeploy the bad one (see [Undeploy](#undeploy-stop-gpu-billing)).
3. Re-deploy the previous version using `--model=<previous_model_id>`.

Note: the previous Vertex AI **model resource** persists (uploaded with `gcloud ai models upload`) even after undeploying. List them: `gcloud ai models list --region=europe-west1 --project=printermonitor-488112`.

---

## Pre-/post-test rituals

Before a heavy test session:
```bash
# Make sure judge is deployed
gcloud ai endpoints describe 6900414029643120640 --region=europe-west1 --project=printermonitor-488112 --format="value(deployedModels[].id)"

# Flush stale state
python scripts/flush_firestore.py --yes
```

After testing:
```bash
# Get model id
DEPLOYED_ID=$(gcloud ai endpoints describe 6900414029643120640 --region=europe-west1 --project=printermonitor-488112 --format="value(deployedModels[].id)")

# Stop GPU billing
gcloud ai endpoints undeploy-model 6900414029643120640 \
  --region=europe-west1 --project=printermonitor-488112 \
  --deployed-model-id=$DEPLOYED_ID

# Purge any leftover frames-in (so next deploy isn't slammed)
gcloud pubsub subscriptions seek $(gcloud pubsub subscriptions list --project=printermonitor-488112 --format="value(name)" | grep dispatcher) \
  --time=$(date -u +%Y-%m-%dT%H:%M:%SZ) --project=printermonitor-488112
```

See [[11-costs-and-monitoring]] for the cost side of this discipline.
