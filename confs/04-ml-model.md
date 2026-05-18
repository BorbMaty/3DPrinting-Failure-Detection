---
title: 04 — ML Model
tags: [ml, yolo, yolov8, dataset, cvat, project/printermonitor]
aliases: [model, YOLO, yolov8x]
type: ml
---

# ML Model — YOLOv8x for 3D-Printing Failure Detection

## Model summary

| | |
|---|---|
| **Architecture** | Ultralytics YOLOv8x (largest of the v8 family) |
| **Framework** | PyTorch 2.2 + CUDA 12.1 |
| **Inference HW** | NVIDIA Tesla T4 on `n1-standard-4` (Vertex AI custom container) |
| **Weights** | `terraform_v2/services/judge/best.pt` (also stored in `gs://printermonitor-488112-models/yolov8x/best.pt`) |
| **Image at deploy** | Baked into the judge Dockerfile (`COPY best.pt /app/best.pt`) — not pulled from GCS at runtime |
| **Inference resolution** | Whatever the Pi sends — capped at 1280×720 by `FRAME_WIDTH` / `FRAME_HEIGHT` in [[03-pi-edge#Frame extractor]] |
| **Throughput at 1 fps** | Sufficient on T4 (~ 400–600 ms / frame); Pi caps at 0.1 fps so margin is huge |

## Classes (9)

Source of truth: `scripts/annotate.py:20-30` `CLASS_NAMES` and the label set in `terraform_v2/services/alert-manager/main.py` `HIGH_SEV`. YOLO embeds these in `model.names`, but for tooling we keep them mirrored in source.

| Class label (exact spelling) | Severity | Email triggered |
|---|---|---|
| `spagetti` | **high** | ✅ |
| `not_sticking` | **high** | ✅ |
| `layer_shift` | **high** | ✅ |
| `warping` | **high** | ✅ |
| `stringing` | low | — |
| `under_extrusion` | low | — |
| `over_extrusion` | low | — |
| `nozzle_clog` | low | — |
| `foreign_object_on_print_area` | low | — |

> **Spelling note** — the class is `spagetti` (one `h`). This is preserved end-to-end in code; the README/`CLAUDE.md` refer to it as "spaghetti" but that label does **not** exist in the model. Don't "fix" the spelling without retraining.

## Severity tiers

`HIGH_SEV` is hard-coded in two places:
- `terraform_v2/services/alert-manager/main.py:23` — gates email notification
- `dashboard/index.html:469-471` — gates the red flash + toast

To add or remove a high-severity class, both files must be edited.

> Low-severity detections still:
> - get written to Firestore `alerts`
> - get drawn as bounding boxes on the dashboard
> - update the dashboard event log
> They just don't email you.

## Confidence threshold

| Where | Default value | Source |
|---|---|---|
| Judge container env | `0.35` | `judge/main.py:16` (env-var fallback) |
| Terraform variable | `0.20` | `variables.tf:20-24` |
| Alert manager env | `0.20` | injected from `var.conf_threshold` in `main.tf:278` |

Effective production behaviour: judge gets `CONF_THRESHOLD` set explicitly when the model is deployed via the manual `gcloud ai models upload --container-env-vars=...,CONF_THRESHOLD=0.35,...` (see [[09-deployment-ops]]), so judge filters at **0.35**. Alert manager re-filters at **0.20** — but since judge already filtered higher, the alert-manager filter is effectively a no-op.

> The two thresholds are independent. Lowering one without the other has no effect. Lower judge's threshold if you want more candidates entering the streak filter; lower alert-manager's only if judge was retuned downward.

## Streak filter

`judge/main.py:111-123`. Per-camera state machine: a label has to appear in `STREAK_REQUIRED` (default **2**) **consecutive** frames before it's published to `detections-out`. A frame without the label resets that label's counter to 0.

Why: single-frame false positives are very common with this model on the noisier classes (a cable looks like spaghetti, a sticker looks like foreign_object). With FPS=0.1 / camera, a 2-frame streak means a real failure has been visible for ≥10 s before alerting.

State is in-memory (`_streaks` dict in the module). It's lost on:
- Container restart
- Vertex AI scaling (we run `min=max=1`, so this rarely happens)
- New deployment

## Bounding box format

Each detection has both representations:
```json
{
  "bbox": [x1, y1, x2, y2],          // pixel ints in source frame
  "x": 0.1875, "y": 0.1667,           // normalized top-left
  "w": 0.6563, "h": 0.7917            // normalized width/height
}
```

The dashboard prefers the normalized form (`dashboard/index.html:357-360`) — it scales by `canvas.width/height` which always matches the live video element. The pixel `bbox` is kept for debugging and for any future client that wants source-resolution rendering.

## Dataset

Source data is **not in git** (`.gitignore` excludes `dataset/`). It lives under `pi_codes/dataset/raw/<cam>/<failure>/` on the capture machine.

Capture tool: `pi_codes/image_taker.py` — see [[03-pi-edge#Dataset capture]].

Annotation tool: CVAT (Computer Vision Annotation Tool), self-hosted at `http://localhost:8080` (the user's local CVAT instance).

## CVAT auto-annotation — `scripts/annotate.py`

A bootstrap script: runs the current `best.pt` over an unlabeled CVAT task and uploads predicted rectangles as auto-annotations for human review/correction.

### Usage
```bash
python3 scripts/annotate.py \
  --weights /home/lucy/.../terraform_v2/services/judge/best.pt \
  --cvat-url http://localhost:8080 \
  --username mara --password alma \
  --task-id 23                # or omit to find all unlabeled tasks
  [--conf 0.25]               # YOLO confidence cutoff for proposals
  [--dry-run]                 # log shapes but don't upload
```

### Internals
- `CVATClient` — wraps the CVAT REST API. Logs in via `/api/auth/login` (handles both token and session/CSRF auth flavours).
- `is_unlabeled()` — true iff `shapes == [] and tags == []`. Skips tasks that already have any annotation.
- `annotate_task()` — for each frame, downloads bytes (`/api/tasks/{id}/data?type=frame&number=N&quality=original`), runs YOLO, builds a CVAT shape per detection, then `PUT /api/tasks/{id}/annotations` with all of them.
- Each shape is `{type: "rectangle", label_id, frame, points: [x1,y1,x2,y2], source: "auto"}`.

### Workflow
1. Capture new failure clips with `image_taker.py`.
2. Upload to a new CVAT task (manual; through CVAT UI).
3. Run `annotate.py --task-id <new>` — populates auto-labels.
4. Human reviews/corrects in CVAT.
5. Export YOLO format and retrain.

The script is intentionally hands-off; it doesn't retrain or push weights anywhere.

## Retraining (outside this repo)

The training pipeline lives outside this repo (Google Colab notebooks, by convention). When new weights arrive:
1. `gsutil cp best.pt gs://printermonitor-488112-models/yolov8x/best.pt`
2. Push to `main` → `docker-judge.yml` workflow ([[08-ci-cd]]) downloads from GCS, builds, pushes new image.
3. Manually upload as new Vertex AI model version + redeploy ([[09-deployment-ops#Vertex AI deploy]]).

## Limitations / honest assessment

- **Tiny dataset**: 9 failure classes with limited variation; the model overfits to the specific printers and camera angles used to collect data.
- **Single-printer bias**: training data is from one or two printers — generalization to wildly different printer types is unverified.
- **Lighting**: cameras assume good ambient or printer LED lighting. Low-light frames degrade noticeably.
- **Streak filter is a band-aid**: a more principled approach would be temporal models (LSTM over CNN features, or a small spatiotemporal head). Out of scope for the thesis.
- **No active learning loop**: misdetections in production don't feed back into the dataset automatically.
