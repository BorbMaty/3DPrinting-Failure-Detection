import base64
import json
import os
import time
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from google.cloud import pubsub_v1

PROJECT_ID   = os.environ["GCP_PROJECT"]
FRAMES_TOPIC = os.environ["FRAMES_TOPIC"]
CAPTURE_FPS  = float(os.environ.get("CAPTURE_FPS", "2"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))
FRAME_WIDTH  = int(os.environ.get("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "960"))

# Comma-separated list of RTSP URLs, e.g. "rtsp://host/cam1,rtsp://host/cam2"
RTSP_URLS_ENV = os.environ.get("RTSP_URLS", os.environ.get("RTSP_URL", ""))
# Comma-separated camera IDs matching the URLs above
CAMERA_IDS_ENV = os.environ.get("CAMERA_IDS", "cam1,cam2,cam3")

publisher  = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, FRAMES_TOPIC)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def capture_loop(rtsp_url: str, camera_id: str):
    interval = 1.0 / CAPTURE_FPS
    seq = 0

    print(f"[{camera_id}] Connecting to {rtsp_url}...", flush=True)

    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[{camera_id}] Failed to open stream, retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        print(f"[{camera_id}] Stream opened.", flush=True)

        while True:
            start = time.time()
            ret, frame = cap.read()

            if not ret:
                print(f"[{camera_id}] Frame read failed, reconnecting...", flush=True)
                cap.release()
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            payload = json.dumps({
                "camera_id": camera_id,
                "seq":       seq,
                "ts":        now_iso(),
                "data_b64":  img_b64,
            }).encode("utf-8")

            publisher.publish(topic_path, payload)
            seq += 1

            elapsed   = time.time() - start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


class Health(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def main():
    # Parse cameras
    urls = [u.strip() for u in RTSP_URLS_ENV.split(",") if u.strip()]
    ids  = [c.strip() for c in CAMERA_IDS_ENV.split(",") if c.strip()]

    if not urls:
        raise RuntimeError("RTSP_URLS env var is required")

    if len(ids) < len(urls):
        # Pad with cam1, cam2, ...
        ids = [ids[i] if i < len(ids) else f"cam{i+1}" for i in range(len(urls))]

    print(f"Starting frame extraction for {len(urls)} camera(s):", flush=True)
    for cid, url in zip(ids, urls):
        print(f"  {cid} → {url}", flush=True)

    # Start one capture thread per camera
    threads = []
    for cid, url in zip(ids, urls):
        t = threading.Thread(target=capture_loop, args=(url, cid), daemon=True)
        t.start()
        threads.append(t)

    # Health check HTTP server (required by Cloud Run)
    port   = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Health)
    print(f"Health server on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()