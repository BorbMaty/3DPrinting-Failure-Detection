"""
Local judge server — mirrors terraform_v2/services/judge/main.py without any GCP/Pub/Sub
dependencies. Exposes the same HTTP interface on localhost:8080 and returns predictions
in the response body (the production judge does this too — the orchestrator just reads it).

The model is loaded lazily via init() so the caller can set MODEL_PATH before importing.
"""

import base64
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_DEFAULT_MODEL = str(
    Path(__file__).parent.parent / "terraform_v2" / "services" / "judge" / "best.pt"
)

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))

_model: YOLO | None = None


def init():
    """Load the YOLO model. Called once before starting the server."""
    global _model, MODEL_PATH, CONF_THRESHOLD
    MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
    CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
    print(f"[judge] Loading model from {MODEL_PATH} ...", flush=True)
    _model = YOLO(MODEL_PATH)
    _model.to("cpu")
    print(f"[judge] Model ready. Classes: {_model.names}", flush=True)


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

            if "instances" in envelope:
                instance = envelope["instances"][0]
                img_b64 = instance.get("data_b64") or instance.get("image_b64")
                camera_id = instance.get("camera_id", "cam1")
                seq = instance.get("seq", -1)
            elif "message" in envelope:
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
            results = _model(img, conf=CONF_THRESHOLD, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = _model.names[cls_id]
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

            print(f"[judge] camera={camera_id} seq={seq} detections={len(detections)}", flush=True)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"predictions": [out]}).encode("utf-8"))

        except Exception as e:
            print(f"[judge] error: {e}", flush=True)
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"error")

    def log_message(self, *args):
        pass


def main():
    if _model is None:
        init()
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"[judge] Listening on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
