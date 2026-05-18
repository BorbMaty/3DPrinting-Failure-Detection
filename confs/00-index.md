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
| **Confidence threshold** | `0.20` (Terraform default) — note CLAUDE.md says 0.35 (judge code default if unset) |
| **Streak filter** | 2 consecutive frames before publishing |
| **Email cooldown** | 300s, single global key `global_email` |
| **Capture rate (Pi)** | 0.1 fps (one frame every 10s per cam) |

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

## Drift / discrepancies to be aware of

Documented in [[12-glossary#Drift between docs and source]].

- `CLAUDE.md` says **spaghetti** but the actual class label in the model is **`spagetti`** (single `h`). Source: `terraform_v2/services/alert-manager/main.py` `HIGH_SEV` and `scripts/annotate.py` `CLASS_NAMES`.
- `CLAUDE.md` says streak=3; the judge code defaults to **2**.
- `CLAUDE.md` says cooldown is per-camera 60s; production code uses a **single global key** `global_email` with **300s** TTL.
- Two `frame-extractor` implementations exist: `terraform_v2/services/frame-extractor/` (Cloud Run shape, multi-cam) and `pi_codes/frame_extractor.py` (what currently runs on the Pi).
- `terraform_v2/services/budget-notifier/main.py` is **dead code** — Terraform deploys `services/alert-manager/` for *both* functions (different entry points). See [[02-cloud-services#Budget Notifier]].

## Tags

`#project/printermonitor` `#thesis/sapientia` `#stack/gcp` `#stack/yolo` `#stack/terraform` `#stack/firebase`
