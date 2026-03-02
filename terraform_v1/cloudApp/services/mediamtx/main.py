import base64
import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from google.cloud import pubsub_v1

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/healthz", "/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass  # suppress access logs

def run_http_server():
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()

threading.Thread(target=run_http_server, daemon=True).start()

PROJECT_ID = os.environ["GCP_PROJECT"]
TOPIC_NAME = os.environ["PUBSUB_TOPIC"]
CAMERA_ID = os.environ.get("CAMERA_ID", "cam1")
FPS = float(os.environ.get("FPS", "2"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))
RTSP_URL = os.environ.get("RTSP_URL", f"rtsp://localhost:8554/{CAMERA_ID}")

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

seq = 0

def now_iso():
    return datetime.now(timezone.utc).isoformat()

print(f"Connecting to {RTSP_URL}", flush=True)

while True:
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(f"error: could not open {RTSP_URL}, retrying...", flush=True)
        time.sleep(5)
        continue

    print(f"Connected to {RTSP_URL}", flush=True)
    interval = 1.0 / FPS

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("error: lost stream, reconnecting...", flush=True)
            break

        try:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, buf = cv2.imencode(".jpg", frame, encode_params)
            img_bytes = buf.tobytes()

            msg = {
                "ts": now_iso(),
                "camera_id": CAMERA_ID,
                "seq": seq,
                "mime": "image/jpeg",
                "jpeg_q": JPEG_QUALITY,
                "data_b64": base64.b64encode(img_bytes).decode("ascii"),
            }
            seq += 1

            publisher.publish(topic_path, json.dumps(msg).encode("utf-8"))
            print(f"published seq={seq}", flush=True)

        except Exception as e:
            print(f"error: {e}", flush=True)

        dt = time.time() - t0
        time.sleep(max(0.0, interval - dt))

    cap.release()
    time.sleep(2)