"""
Local test orchestrator — runs the full detection pipeline without any GCP dependencies.

Pipeline (mirrors production):
  source (webcam / video file / image dir / single image)
    → local_judge  (HTTP server on localhost:8080, YOLOv8x CPU inference)
      → local_alert_handler  (in-memory cooldown, appends to local_test.md)

Usage:
  python orchestrator.py                            # default: webcam 0, cam1, 1 fps
  python orchestrator.py --source 0                 # explicit webcam index
  python orchestrator.py --source path/to/video.mp4
  python orchestrator.py --source path/to/frames/   # directory of jpg/png images
  python orchestrator.py --source path/to/frame.jpg  # single image (loops forever)
  python orchestrator.py --source 0 --cameras 3     # 3 webcams (indices 0,1,2)

Optional env vars:
  MODEL_PATH        path to best.pt  (default: ../terraform_v2/services/judge/best.pt)
  CONF_THRESHOLD    float 0-1        (default: 0.35)
  COOLDOWN_SECONDS  int              (default: 60)
  GMAIL_ADDRESS     your Gmail       (if set, real emails are sent for high-severity)
  GMAIL_APP_PASSWORD                 (required if GMAIL_ADDRESS is set)
"""

import argparse
import base64
import glob
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import requests

# Resolve paths before touching sys.path
_REPO_ROOT = Path(__file__).parent.parent
_LOCAL_TEST_DIR = Path(__file__).parent
_LOG_FILE = _REPO_ROOT / "local_test.md"

sys.path.insert(0, str(_LOCAL_TEST_DIR))

JUDGE_URL = "http://localhost:8080/predict"
JUDGE_HEALTH_URL = "http://localhost:8080/healthz"


def _parse_args():
    p = argparse.ArgumentParser(description="Local 3D printing failure detection pipeline")
    p.add_argument("--source", default="0",
                   help="Webcam index, video file, image directory, or single image")
    p.add_argument("--cameras", type=int, default=1,
                   help="Number of webcams when source is an integer (default 1)")
    p.add_argument("--camera-id", default="cam1",
                   help="Camera label for single-source modes")
    p.add_argument("--fps", type=float, default=1.0,
                   help="Capture / playback rate in frames per second (default 1.0)")
    p.add_argument("--model", default=None,
                   help="Override path to best.pt")
    p.add_argument("--conf", type=float, default=None,
                   help="Override confidence threshold (0–1)")
    return p.parse_args()


def _apply_env(args):
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    if args.conf is not None:
        os.environ["CONF_THRESHOLD"] = str(args.conf)


def _start_judge():
    import local_judge
    local_judge.init()
    t = threading.Thread(target=local_judge.main, daemon=True, name="judge-server")
    t.start()
    for _ in range(40):
        try:
            if requests.get(JUDGE_HEALTH_URL, timeout=1).status_code == 200:
                print("[orchestrator] Judge ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Judge server did not become healthy in time.")


def _log_session_header(source: str, fps: float, model_path: str, conf: float):
    ts = datetime.now(timezone.utc).strftime("%a %d %b %Y, %H:%M:%S UTC")
    with open(_LOG_FILE, "a") as f:
        f.write(f"\n## Session — {ts}\n\n")
        f.write(f"- **Source:** `{source}`\n")
        f.write(f"- **FPS:** {fps}\n")
        f.write(f"- **Model:** `{model_path}`\n")
        f.write(f"- **Confidence threshold:** {conf}\n\n")
        f.write("| Time | Camera | Seq | Detections |\n")
        f.write("|------|--------|-----|------------|\n")


def _encode_frame(frame) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _post_and_handle(img_b64: str, camera_id: str, seq: int):
    import local_alert_handler
    payload = {"instances": [{"data_b64": img_b64, "camera_id": camera_id, "seq": seq}]}
    try:
        r = requests.post(JUDGE_URL, json=payload, timeout=60)
        if r.status_code == 200:
            result = r.json()["predictions"][0]
            local_alert_handler.handle_detection(result)
    except Exception as e:
        print(f"[orchestrator] POST error: {e}", flush=True)


# ── Source runners ────────────────────────────────────────────────────────────

def _run_webcam(device: int, camera_id: str, fps: float):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {device}")
    interval = 1.0 / fps
    seq = 0
    print(f"[orchestrator] Webcam {device} → {camera_id} at {fps} fps  (Ctrl+C to stop)", flush=True)
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"[orchestrator] [{camera_id}] Frame read failed, retrying...", flush=True)
                time.sleep(1)
                continue
            _post_and_handle(_encode_frame(frame), camera_id, seq)
            seq += 1
            remaining = interval - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        cap.release()


def _run_webcam_multi(n_cams: int, fps: float):
    """Open n_cams consecutive webcam indices in parallel threads."""
    threads = []
    for i in range(n_cams):
        t = threading.Thread(
            target=_run_webcam, args=(i, f"cam{i+1}", fps), daemon=True, name=f"cam{i+1}"
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def _run_video(path: str, camera_id: str, fps: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    interval = 1.0 / fps
    seq = 0
    print(f"[orchestrator] Video {path} → {camera_id} at {fps} fps  (Ctrl+C to stop)", flush=True)
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[orchestrator] Video ended, looping...", flush=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        _post_and_handle(_encode_frame(frame), camera_id, seq)
        seq += 1
        remaining = interval - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)


def _run_image_dir(directory: str, camera_id: str, fps: float):
    images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png")
        for p in glob.glob(os.path.join(directory, ext))
    )
    if not images:
        raise RuntimeError(f"No jpg/png images found in {directory}")
    interval = 1.0 / fps
    seq = 0
    print(f"[orchestrator] {len(images)} images in {directory} → {camera_id} at {fps} fps  (Ctrl+C to stop)", flush=True)
    while True:
        for img_path in images:
            t0 = time.time()
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            _post_and_handle(_encode_frame(frame), camera_id, seq)
            seq += 1
            remaining = interval - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)


def _run_single_image(path: str, camera_id: str, fps: float):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {path}")
    interval = 1.0 / fps
    seq = 0
    print(f"[orchestrator] Looping {path} → {camera_id} at {fps} fps  (Ctrl+C to stop)", flush=True)
    while True:
        t0 = time.time()
        _post_and_handle(_encode_frame(frame), camera_id, seq)
        seq += 1
        remaining = interval - (time.time() - t0)
        if remaining > 0:
            time.sleep(remaining)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    _apply_env(args)

    # Import after env vars are set so local_judge reads the right MODEL_PATH
    import local_alert_handler
    local_alert_handler.LOG_FILE = str(_LOG_FILE)

    _start_judge()

    import local_judge
    _log_session_header(
        source=args.source,
        fps=args.fps,
        model_path=local_judge.MODEL_PATH,
        conf=local_judge.CONF_THRESHOLD,
    )

    source = args.source
    try:
        device = int(source)
        if args.cameras > 1:
            _run_webcam_multi(args.cameras, args.fps)
        else:
            _run_webcam(device, args.camera_id, args.fps)
    except ValueError:
        path = Path(source)
        if path.is_dir():
            _run_image_dir(str(path), args.camera_id, args.fps)
        elif path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
            _run_video(str(path), args.camera_id, args.fps)
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            _run_single_image(str(path), args.camera_id, args.fps)
        else:
            raise RuntimeError(f"Unrecognised source: {source!r}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[orchestrator] Stopped.", flush=True)
