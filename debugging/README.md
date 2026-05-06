# Local debug pipeline (RTX 3060 box)

Replaces the cloud `dispatcher` + Vertex AI `judge` with a local subscriber that
pulls frames from the Pi over Pub/Sub and runs YOLO inference on a local GPU.

Use this when:
- the cloud endpoint has no model deployed (no T4 burn while debugging),
- you want to see the exact bytes the Pi is publishing,
- you want to compare cloud-shaped detections against your local `best.pt`.

The Pi keeps publishing to `frames-in` as normal. A new pull subscription
(`frames-in-debug-local`) fans the same messages out to your machine, so the
cloud pipeline state is irrelevant — the dispatcher can keep failing in the
background, you still get every frame.

## Prerequisites

- `gcloud` authenticated as a principal with Pub/Sub subscriber rights on
  `printermonitor-488112` (project owner is enough).
- Python 3.10+.
- NVIDIA driver + CUDA 12.1-compatible GPU (RTX 3060 ✓).

## One-time setup

```bash
cd debugging
./setup.sh                                  # creates frames-in-debug-local (idempotent)

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu121

gcloud auth application-default login       # ADC for the Pub/Sub client
```

## Running

```bash
source .venv/bin/activate
python run.py --save-frames --show
```

Per frame received from the Pi you'll see one line:

```
[cam1] seq=42 1280x720 24ms detections=[spaghetti=0.71, stringing=0.33]
```

If `--save-frames` is on, raw and annotated copies are written to
`debugging/frames/<camera>_<seq>.jpg` (and `..._annot.jpg` when there are
detections) for offline inspection — handy for comparing against your local
dataset runs at 0.28–0.74 confidence.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--model PATH` | `../terraform_v2/services/judge/best.pt` | weights to load |
| `--conf FLOAT` | `0.20` | confidence threshold (matches v4 cloud env) |
| `--device STR` | `cuda:0` | torch device — `cpu` if no GPU available |
| `--save-frames` | off | dump raw + annotated frames to `./frames/` |
| `--show` | off | live OpenCV window per camera (needs a display) |
| `--max-messages` | `4` | Pub/Sub flow control: max in-flight messages |

## What this confirms

- **Decode integrity** — every line printed means `cv2.imdecode` succeeded on
  exactly the bytes the Pi sent. If you see `cv2.imdecode failed`, the
  problem is upstream (extractor or Pub/Sub envelope), not the judge.
- **Weights behaviour** — running on the *same* `best.pt` the cloud image
  is supposed to use. If detections look right here but disappear in the
  cloud once redeployed, the deployed image has different weights than this
  file (re-check the GCS hash vs the repo file).
- **Compression / aspect ratio** — saved frames let you eyeball the JPEG
  quality and confirm 1280×720 isn't being silently stretched anywhere.

## Cleanup

The subscription accumulates a backlog (10-min retention) when no consumer is
running. Delete it when you're done debugging:

```bash
gcloud pubsub subscriptions delete frames-in-debug-local \
  --project=printermonitor-488112
```

## Files

| File | Purpose |
|---|---|
| `setup.sh` | Creates the pull subscription. Safe to re-run. |
| `requirements.txt` | Python deps (excluding torch — install with the cu121 index URL). |
| `run.py` | Subscriber + YOLO inference loop. |
| `.gitignore` | Ignores `.venv/`, `frames/`, caches. |
