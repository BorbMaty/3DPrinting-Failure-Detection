---
title: 00 — Index
tags: [moc, index, project/printermonitor]
aliases: [MOC, Map of Content, Home]
type: index
---

# 3D Printing Failure Detection — Knowledge Base

> Map of Content for the **PrinterMonitor** thesis project. Each note below is self-contained and links to its neighbours via `[[wikilinks]]`. Re-derive from source, not from prior `README.md` / `documentation.md` / `CLAUDE.md` (those drift).

## Quick facts

| | |
|---|---|
| **GCP project** | `printermonitor-488112` |
| **Region** | `europe-west1` |
| **Vertex AI endpoint ID** | `6900414029643120640` |
| **Dashboard URL** | https://printermonitor-488112.web.app |
| **Model** | YOLOv8x, 9 classes (see [[04-ml-model]]) |
| **Confidence threshold** | `0.35` (Terraform var + judge env; alert-manager *code* default is `0.20`, overridden to `0.35` by Terraform) |
| **Streak filter** | 2 consecutive frames before publishing |
| **Email cooldown** | 300s, single global key `global_email` |
| **Capture rate (Pi)** | 0.1 fps (one frame every 10s per cam) |
| **WebRTC host** | `cam.printermonitor.app` (permanent Cloudflare named tunnel) |
| **Frames bucket** | `printermonitor-488112-frames` (public read; judge uploads one JPEG per inference) |
| **Capture toggle** | Dashboard writes `system_state/extraction.enabled`; Pi polls it to pause/resume capture |

## What changed recently (since the last doc pass)

The pipeline gained an **inference-logging path** and a few reliability features. If you only read the old notes, you'll miss these:

- **Every inference is now persisted**, not just defects. The judge **always publishes** to `detections-out` (even zero-detection frames), and alert-manager writes each one to a new Firestore **`inferences`** collection. See [[02-cloud-services#Judge]], [[02-cloud-services#Alert Manager]].
- **The judge uploads the frame to GCS** (`printermonitor-488112-frames`, public read) and emits a **`frame_url`** in every detection payload, so the dashboard can show the actual annotated image. See [[04-ml-model#Bounding box format]], [[05-infrastructure#Storage]].
- **Dashboard is now two pages** — *Live streams* (WebRTC) and *Inference log* (frame thumbnails + table from `inferences`). See [[06-dashboard]].
- **Remote capture kill-switch** — a dashboard button toggles `system_state/extraction.enabled`; the Pi extractor polls it (5 s cache) and pauses/resumes. See [[03-pi-edge#Frame extractor]].
- **Dispatcher self-protects against backlog** — drops frames older than `MAX_FRAME_AGE_S=5` and drops (instead of retrying) when the endpoint returns 404/503. The manual Pub/Sub seek is now mostly a backstop. See [[02-cloud-services#Dispatcher]], [[09-deployment-ops#Purge frames-in backlog]].

## Topic map

- [[01-architecture]] — End-to-end data flow, message contracts, component diagram
- [[02-cloud-services]] — Dispatcher · Judge · Alert Manager · Budget Notifier (Python services)
- [[03-pi-edge]] — Pi codes, MediaMTX, GStreamer RTSP, Cloudflare Tunnel
- [[04-ml-model]] — YOLOv8x model, 9 classes, severity tiers, streak filter, CVAT auto-annotation
- [[05-infrastructure]] — Terraform, GCP resources, IAM bindings, Pub/Sub, Firestore
- [[06-dashboard]] — Firebase hosting, WebRTC WHEP playback, FCM push, Firestore realtime
- [[07-testing]] — pytest strategy, sys.modules mocking, 90% coverage gate
- [[08-ci-cd]] — GitHub Actions: terraform · docker-judge · python-tests · compileLatex
- [[09-deployment-ops]] — Vertex AI deploy/undeploy, Pub/Sub purge, Firestore flush
- [[10-local-test]] — Local replica harness in `local_test/` (no GCP needed)
- [[11-costs-and-monitoring]] — GPU billing, budget alerts, FCM topics, monitoring gaps
- [[12-glossary]] — Terms, IDs, env vars, magic numbers, GCP resource cheat sheet
- [[feedback]] — Review notes: security/correctness asks, code smells, reliability nits, repo hygiene

## Drift / discrepancies to be aware of

Documented in [[12-glossary#Drift between docs and source]].

- `CLAUDE.md` says **spaghetti** but the actual class label in the model is **`spagetti`** (single `h`). Source: `terraform_v2/services/alert-manager/main.py` `HIGH_SEV` and `scripts/annotate.py` `CLASS_NAMES`.
- `CLAUDE.md` says streak=3; the judge code defaults to **2**.
- `CLAUDE.md` says cooldown is per-camera 60s; production code uses a **single global key** `global_email` with **300s** TTL.
- Two `frame-extractor` implementations exist: `terraform_v2/services/frame-extractor/` (Cloud Run shape, multi-cam) and `pi_codes/frame_extractor.py` (what currently runs on the Pi).
- `terraform_v2/services/budget-notifier/main.py` is **dead code** — Terraform deploys `services/alert-manager/` for *both* functions (different entry points). See [[02-cloud-services#Budget Notifier]].
- `alert-manager/main.py:16` code default for `CONF_THRESHOLD` is **`0.20`**, not `0.35`. Production gets `0.35` because Terraform injects `var.conf_threshold` into the function env. The 12-glossary "magic numbers" table previously claimed the code default was `0.35` — it's `0.20`.

## Tags

`#project/printermonitor` `#thesis/sapientia` `#stack/gcp` `#stack/yolo` `#stack/terraform` `#stack/firebase`
