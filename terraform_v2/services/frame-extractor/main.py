import base64
import json
import os
import time
from datetime import datetime, timezone

import cv2
from google.cloud import pubsub_v1

PROJECT_ID = os.environ["GCP_PROJECT"]
FRAMES_TOPIC = os.environ["FRAMES_TOPIC"]
RTSP_URL = os.environ["RTSP_URL"]
CAPTURE_FPS = float(os.environ.get("CAPTURE_FPS", "2"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))
FRAME_WIDTH = int(os.environ.get("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "480"))

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, FRAMES_TOPIC)

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def capture_loop():
    interval = 1.0 / CAPTURE_FPS
    seq = 0

    print(f"Connecting to {RTSP_URL}...", flush=True)

    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("Failed to open RTSP stream, retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        print("Stream opened, capturing frames...", flush=True)

        while True:
            start = time.time()
            ret, frame = cap.read()

            if not ret:
                print("Frame read failed, reconnecting...", flush=True)
                cap.release()
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            payload = json.dumps({
                "camera_id": "cam1",
                "seq": seq,
                "ts": now_iso(),
                "data_b64": img_b64,
            }).encode("utf-8")

            publisher.publish(topic_path, payload)
            seq += 1

            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    # Health check server in background
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        def log_message(self, *args):
            pass

    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Health)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Health server on port {port}", flush=True)

    capture_loop()
