import base64
import binascii
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
from google.cloud import pubsub_v1
from ultralytics import YOLO

PROJECT_ID = os.environ["GCP_PROJECT"]
DETECTIONS_TOPIC = os.environ["DETECTIONS_TOPIC"]
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, DETECTIONS_TOPIC)

print(f"Loading YOLO model from {MODEL_PATH}...", flush=True)
model = YOLO(MODEL_PATH)
print(f"Model loaded. Classes: {model.names}", flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _safe_b64decode(s: str) -> bytes:
    s = (s or "").strip()
    if not s:
        raise ValueError("empty base64 input")
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing
    return base64.b64decode(s, validate=False)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/healthz", "/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/predict":
            self.send_response(404)
            self.end_headers()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            envelope = json.loads(body.decode("utf-8"))

            # Vertex AI sends: {"instances": [...]}
            # Pub/Sub push sends: {"message": {"data": "..."}}
            if "instances" in envelope:
                # Vertex AI format
                instance = envelope["instances"][0]
                img_b64 = instance.get("data_b64") or instance.get("image_b64")
                camera_id = instance.get("camera_id", "cam1")
                seq = instance.get("seq", -1)
            elif "message" in envelope:
                # Pub/Sub push format
                payload_bytes = _safe_b64decode(envelope["message"]["data"])
                data = json.loads(payload_bytes.decode("utf-8"))
                img_b64 = data.get("data_b64") or data.get("image_b64")
                camera_id = data.get("camera_id", "cam1")
                seq = data.get("seq", -1)
            else:
                raise KeyError("unrecognised envelope format")

            img_bytes = _safe_b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed")

            h, w = img.shape[:2]
            results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2],
                    "x": round(x1 / w, 4),
                    "y": round(y1 / h, 4),
                    "w": round((x2 - x1) / w, 4),
                    "h": round((y2 - y1) / h, 4),
                })

            out = {
                "ts": now_iso(),
                "camera_id": camera_id,
                "seq": seq,
                "detections": detections,
            }

            publisher.publish(topic_path, json.dumps(out).encode("utf-8"))
            print(f"camera={camera_id} seq={seq} detections={len(detections)}", flush=True)

            # Vertex AI expects {"predictions": [...]}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"predictions": [out]}).encode("utf-8"))

        except Exception as e:
            print(f"error: {e}", flush=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"error")

    def log_message(self, *args):
        pass


def main():
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Judge server listening on port {port}...", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
