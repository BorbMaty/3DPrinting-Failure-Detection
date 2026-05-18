---
title: 03 — Pi Edge
tags: [pi, raspberry-pi, edge, mediamtx, cloudflare, gstreamer, project/printermonitor]
aliases: [edge, raspberry pi, MediaMTX]
type: edge
---

# Raspberry Pi Edge

Everything that runs locally on the Pi: USB camera capture, RTSP relay, WebRTC publishing, Cloudflare exit, and frame extraction to GCP.

## Hardware

3× USB UVC cameras, addressed by stable by-id paths (so reboots don't shuffle them):
- `cam1` → `/dev/v4l/by-id/usb-SunplusIT_Inc_FHD_Camera_Microphone_01.00.00-video-index0` — MJPEG @ 1280×720
- `cam2` → `/dev/v4l/by-id/usb-Camera_Vendor_Conference_Camera_00.00.01-video-index0` — H.264 @ 1280×720
- `cam3` → `/dev/v4l/by-id/usb-Microsoft_Microsoft®_LifeCam_HD-3000-video-index0` — MJPEG @ 1920×1080 (downscaled in pipeline)

(Sources: `pi_codes/start_3cams_rtsp.sh:42-44`)

---

## RTSP publisher — `pi_codes/start_3cams_rtsp.sh`

GStreamer pipelines that ingest each camera and publish to MediaMTX as `rtsp://localhost:8554/camN`.

Three pipeline variants:
- **MJPEG → H.264 transcode** (cam1, cam3): `v4l2src ! image/jpeg ! jpegdec ! videoconvert ! v4l2h264enc ! rtspclientsink`
- **Native H.264 passthrough** (cam2): `v4l2src ! video/x-h264 ! h264parse ! rtspclientsink` (no transcode — saves CPU)
- **MJPEG + flip** (for inverted mounts): adds `videoflip method=rotate-180`

Per-camera logs to `~/cam1.log`, etc. A `while true` loop polls each PID every 5 s and restarts a pipeline if it died (`kill -0 $pid` test). No exit on Ctrl-C → relies on shell killing the parent.

> `start_cam` takes args `(name, dev, flip, fmt)` where `fmt` is `mjpg` (default) or `h264`.

---

## RTSP/WebRTC relay — `terraform_v2/services/mediamtx/`

**Image:** `bluenviron/mediamtx:latest`
**Config (`mediamtx.yml`):** APIs disabled (`api: no`, `metrics: no`, `pprof: no`). Listens on:
- `:8554` RTSP (publisher + consumer)
- `:8080` RTMP
- `:8888` HLS
- `:8889` WebRTC WHEP (this is what the dashboard hits)
- `:8890` SRT

Paths `cam1`, `cam2`, `cam3` all have `source: publisher` — the GStreamer pipelines push, anyone can pull.

Dockerfile runs as UID `65534` (nobody) and has a `wget` healthcheck against `:8889`.

> The container exists in `terraform_v2/services/mediamtx/` but is **not built or deployed by Terraform**. It's run manually on the Pi via Docker (`docker run` or compose).

---

## Cloudflare Tunnel

External entry point for WebRTC. The Pi runs `cloudflared` with a Named Tunnel; the tunnel's hostname becomes the dashboard's `MEDIAMTX_HOST`. Tunnel hostname is fed into Terraform via `var.cloudflare_tunnel_hostname` → surfaced in output `mediamtx_whep_url`.

The dashboard reads it via `window.MEDIAMTX_HOST` (with a fallback hard-coded hostname `hire-measures-ink-buf.trycloudflare.com` in `dashboard/index.html:266`). If the tunnel hostname changes, the fallback must be updated — there's an output `dashboard_host_update_command` that prints a `sed` line for this.

Two tunnel modes:
- **Quick tunnel** (`cloudflared tunnel --url http://localhost:8889`) — ephemeral hostname, regenerates every run.
- **Named tunnel** — stable hostname, requires Cloudflare account setup. This is the intended production path; the variable being non-empty switches the system over.

---

## Frame extractor — `pi_codes/frame_extractor.py`

The script that actually publishes frames to GCP from the Pi. 86 lines.

### Config (env vars, all optional)

| Var | Default | Notes |
|---|---|---|
| `GCP_PROJECT` | `printermonitor-488112` | |
| `PUBSUB_TOPIC` | `frames-in` | |
| `CAPTURE_FPS` | `0.1` | One frame every 10 s **per camera** — this is the throttle that keeps inference cost down |
| `JPEG_QUALITY` | `70` | |
| `FRAME_WIDTH` | `1280` | Max dim, only downscales |
| `FRAME_HEIGHT` | `720` | |

`CAMERAS` is hard-coded (`pi_codes/frame_extractor.py:18-22`): `[(cam1, rtsp://localhost:8554/cam1), …]`. To change camera count, edit the list.

### Lifecycle
- One daemon thread per camera, runs `capture_loop()`.
- Outer loop: open `cv2.VideoCapture(rtsp_url)`. If `not isOpened`, sleep 5 s and retry.
- Inner loop: `cap.read()`. On `not ret`, release and break to outer loop (reconnect).
- On success: resize if oversize → JPEG-encode at `JPEG_QUALITY` → base64 → publish to Pub/Sub.
- Throttle: `sleep(max(0, interval − elapsed))`.

### Auth
Runs with `GOOGLE_APPLICATION_CREDENTIALS` pointing to a key file for `sa-frame-extractor@…`. The SA is created in Terraform and granted `roles/pubsub.publisher` on `frames-in`. **The key file lives on the Pi only** — not in git (`.gitignore` excludes `*.json`).

### vs. the Cloud Run version
`terraform_v2/services/frame-extractor/main.py` is the more flexible **multi-cam Cloud Run shape**: reads `RTSP_URLS` / `CAMERA_IDS` env vars, has a `/healthz` HTTP server, and a stricter container (UID 1001, libgl1+libglib2). It can run on the Pi or in Cloud Run.

In the current deployment, the Pi runs `pi_codes/frame_extractor.py`. The Cloud Run image is the alternative path if the Pi was retired. The two implementations are not strictly identical — `pi_codes` has hard-coded camera list and slightly different env-var names (`PUBSUB_TOPIC` vs `FRAMES_TOPIC`).

---

## Dataset capture — `pi_codes/image_taker.py`

Stand-alone CLI for collecting training data from the same 3 cameras (not part of the runtime pipeline; used only when building the dataset).

### Flow
1. Opens 3 RTSP streams in **background threads** (`CamReader` class) so reads never block. Each thread spins on `cap.read()` and keeps only the latest frame (`CAP_PROP_BUFFERSIZE=1`).
2. Shows preview windows for all three cameras.
3. Prompts for a `failure_name` (e.g. `stringing`, `warping`).
4. Continues counter from the highest existing `<failure>-<cam>-NNNN.jpg` in `dataset/raw/<cam>/<failure>/`.
5. Captures one frame per camera per second, labelled with `cv2.putText` showing `REC <failure> | <cam> | #N`.
6. Press Enter to stop.

The RTSP host in this script is **`192.168.1.10:8554`** (not `localhost`) — meaning this was probably run from a **second machine** on the LAN, not directly on the Pi. (`pi_codes/image_taker.py:3-7`).

Output goes to `dataset/raw/` which is gitignored.

---

## Pi → cloud auth setup

One-time:
1. Create SA: handled by Terraform (`google_service_account.frame_extractor`).
2. Download key JSON (manual): `gcloud iam service-accounts keys create ...`
3. Copy to Pi: scp to `/home/pi/sa-frame-extractor-key.json`.
4. Set `GOOGLE_APPLICATION_CREDENTIALS=/home/pi/sa-frame-extractor-key.json` in the shell that runs `frame_extractor.py`.

---

## Pi startup checklist

```bash
# 1. Start MediaMTX (Docker on Pi)
docker run -d --rm --name mediamtx --network host bluenviron/mediamtx:latest

# 2. Start the GStreamer pipelines
./pi_codes/start_3cams_rtsp.sh

# 3. Start Cloudflare Tunnel (named or quick)
cloudflared tunnel run my-tunnel    # or: cloudflared tunnel --url http://localhost:8889

# 4. Start frame extraction to Pub/Sub
export GOOGLE_APPLICATION_CREDENTIALS=/home/pi/sa-frame-extractor-key.json
python3 pi_codes/frame_extractor.py
```

The dashboard at https://printermonitor-488112.web.app should now show three live tiles. Bounding boxes only appear after `[[02-cloud-services#Judge]]` is deployed to the Vertex AI endpoint.

See [[09-deployment-ops]] for the cloud-side counterpart of this startup.
