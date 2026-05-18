---
title: 11 — Costs & Monitoring
tags: [costs, billing, gpu, monitoring, budget, project/printermonitor]
aliases: [costs, billing, GPU cost, budget alerts]
type: ops
---

# Costs & Monitoring

> **The T4 GPU is the only meaningful cost.** Everything else (Pub/Sub, Firestore, Cloud Functions, Cloud Storage, dashboard) sums to pennies per month at this project's scale.

## Cost breakdown (rough, EUR-west1, May 2026)

| Component | When | Approx. cost |
|---|---|---|
| **Vertex AI T4 GPU + n1-standard-4** | While deployed | **≈ $37 / day** (~$1100 / month) |
| Cloud Functions Gen2 (3 functions) | Per invoke + memory-s | ~$0.10 / month at our volume |
| Pub/Sub (3 topics, low traffic) | Per byte + ops | ~$0.01 / month |
| Firestore | Per read/write + storage | ~$0.10 / month |
| Cloud Storage (state + sources + model bucket) | Per GB-month | ~$0.02 / month |
| Artifact Registry | Per GB-month | ~$0.05 / month |
| Eventarc | Per event | ~$0.01 / month |
| Firebase Hosting | Free tier covers it | $0 |
| Cloud Logging | Free tier covers it | $0 |
| **GCP total when judge undeployed** | | **< $1 / month** |
| Cloudflare Tunnel | always | $0 (free tier) |

**Discipline**: undeploy the GPU immediately after every test session. The single highest-impact operational habit on this project.

## Why T4 specifically

T4 was chosen because:
- It's the **cheapest GPU** Vertex AI offers in europe-west1 that can run YOLOv8x at usable speed.
- 16 GB VRAM is comfortably more than YOLOv8x needs (~ 2 GB at FP32).
- Cold start is faster than A100/L4 (smaller, simpler container).
- Doesn't require GPU quota bump in a fresh GCP project.

Alternatives that were considered:
- **CPU inference (n1-standard-4 only)**: ~3 s per frame for YOLOv8x. With Pi capturing at 0.1 fps per cam × 3 cams = 0.3 frames/s aggregate, it would technically keep up — but cold-start startup time is brutal (model load on CPU is also slower). Worth re-evaluating if cost is critical.
- **Smaller model (YOLOv8s or YOLOv8m)**: would run on CPU comfortably and trade ~3-5% mAP for ~10× cost savings. Out of scope for the thesis (the thesis question is whether YOLOv8x specifically works).

## Budget alert pipeline

```
GCP Cloud Billing budget ($5 monthly cap)
  └─ Pub/Sub topic: budget-notifications
     └─ Eventarc → Cloud Function: budget-notifier
          ├─ Firestore: budget_alerts (audit log)
          └─ Gmail SMTP (cooldown key: "budget", same 5-min TTL)
```

The budget itself is **not Terraform-managed** (the CI service account lacks `billingAccounts.budgets` perms). See [[05-infrastructure#Resources NOT managed by Terraform]] and `main.tf:321-323` for the comment that documents this.

Budget metadata:
- **Display name**: configurable; current is whatever was set in the Console
- **Threshold**: $5 / month (this is the alert trigger, NOT a hard cap — GCP doesn't cap spending; you have to undeploy)
- **Budget ID** (for re-importing into Terraform later): `695e9489-62cb-4e37-82cc-2d0e1ed4c8e0`

Test the budget pipeline by publishing a fake message directly:
```bash
gcloud pubsub topics publish budget-notifications \
  --project=printermonitor-488112 \
  --message='{"costAmount":4.50,"budgetAmount":5.00,"budgetDisplayName":"monthly-cap"}'
```

You should get a Firestore doc in `budget_alerts/` within a few seconds and an email titled `💸 GCP Budget Alert — monthly-cap` (if not on cooldown).

## FCM push notification topics

`terraform_v2/services/budget-notifier/main.py` (dead code; see [[02-cloud-services#Budget Notifier]]) defines two FCM topics:
- `defect-alerts`
- `budget-alerts`

If FCM is ever re-enabled, dashboards subscribe via the Firebase SDK (`messaging.subscribeToTopic(token, "defect-alerts")` etc., server-side via Admin SDK). Service worker `dashboard/firebase-messaging-sw.js` is already in place.

The current deployed `alert-manager/main.py` does **not** publish to FCM — emails only.

## Monitoring gaps

This project does **not** have:
- **Health endpoint dashboards** — no Grafana / Cloud Monitoring dashboards configured for the services.
- **Latency tracking** — no histogram metrics; latency numbers in [[01-architecture#Latency budget]] are eyeballed.
- **Error rate alerts** — Cloud Functions log errors to Cloud Logging, but no alerting policy fires on error thresholds.
- **Dataset drift / model performance monitoring** — no inference confidence histogram, no concept drift detection. If the model silently degrades, you'll find out via missed detections.
- **Pi heartbeat** — there's no signal in GCP that the Pi is *actually* publishing. If `frame_extractor.py` dies, `frames-in` goes silent and nothing alerts. Mitigation: the dashboard's WebRTC tiles also go dark, but no email/push fires.
- **GPU undeploy reminder** — no automated check that the judge isn't accidentally running over a weekend.

## Cost-watching habits

1. **Set the budget low and aggressive** ($5 monthly is intentional; it triggers email within hours of leaving the judge deployed by mistake).
2. **Always `undeploy` at end of session** — see [[09-deployment-ops#Pre-/post-test rituals]].
3. **Check billing console manually** at the start of each work day for the first week of a new feature push.
4. **`gcloud ai endpoints describe` is your friend** — quick way to verify the judge is undeployed:
   ```bash
   gcloud ai endpoints describe 6900414029643120640 \
     --region=europe-west1 --project=printermonitor-488112 \
     --format="value(deployedModels[].id)"
   # Empty output = no deployment = no GPU cost.
   ```

## Forward-looking suggestions (out of scope)

- Move to a **Cloud Run + ONNX CPU inference** version for "demo mode" — costs cents instead of dollars, accepts slower latency.
- Add a **Cloud Scheduler** job that posts to a topic with a cron + a small Function that auto-undeploys judge after N hours of inactivity (no `frames-in` traffic).
- Wire up Cloud Monitoring alerts for `frames-in` publish rate < some threshold (Pi-down detector).

See [[09-deployment-ops]] for the deploy/undeploy commands and [[05-infrastructure]] for what's managed where.
