import base64
import binascii
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
from google.cloud import pubsub_v1
from google.cloud import storage as gcs_lib
from ultralytics import YOLO

PROJECT_ID        = os.environ["GCP_PROJECT"]
DETECTIONS_TOPIC  = os.environ["DETECTIONS_TOPIC"]
MODEL_PATH        = os.environ.get("MODEL_PATH", "/app/best.pt")
CONF_THRESHOLD    = float(os.environ.get("CONF_THRESHOLD", "0.35"))
STREAK_REQUIRED   = int(os.environ.get("STREAK_REQUIRED", "2"))
FRAMES_BUCKET     = os.environ.get("FRAMES_BUCKET", "")
JPEG_QUALITY      = int(os.environ.get("JPEG_QUALITY", "60"))

publisher  = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, DETECTIONS_TOPIC)

print(f"Loading YOLO model from {MODEL_PATH}...", flush=True)
model = YOLO(MODEL_PATH)
print(f"Model loaded. Classes: {model.names}", flush=True)

_gcs_client: gcs_lib.Client | None = None

def _get_gcs() -> gcs_lib.Client | None:
    global _gcs_client
    if FRAMES_BUCKET and _gcs_client is None:
        _gcs_client = gcs_lib.Client()
    return _gcs_client

# Per-camera streak counters: {camera_id: {label: consecutive_frame_count}}
_streaks: dict[str, dict[str, int]] = {}


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


def _upload_frame(img: np.ndarray, camera_id: str, ts: str) -> str:
    """Compress img to JPEG and upload to GCS; return the public URL or ''."""
    client = _get_gcs()
    if client is None:
        return ""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return ""
    safe_ts = ts.replace(":", "-").replace("+", "").replace(".", "-")
    blob_name = f"frames/{camera_id}/{safe_ts}.jpg"
    blob = client.bucket(FRAMES_BUCKET).blob(blob_name)
    blob.upload_from_string(buf.tobytes(), content_type="image/jpeg")
    return f"https://storage.googleapis.com/{FRAMES_BUCKET}/{blob_name}"


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
                instance  = envelope["instances"][0]
                img_b64   = instance.get("data_b64") or instance.get("image_b64")
                camera_id = instance.get("camera_id", "cam1")
                seq       = instance.get("seq", -1)
            elif "message" in envelope:
                payload_bytes = _safe_b64decode(envelope["message"]["data"])
                data      = json.loads(payload_bytes.decode("utf-8"))
                img_b64   = data.get("data_b64") or data.get("image_b64")
                camera_id = data.get("camera_id", "cam1")
                seq       = data.get("seq", -1)
            else:
                raise KeyError("unrecognised envelope format")

            img_bytes = _safe_b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img       = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode failed")

            h, w = img.shape[:2]
            results = model(img, conf=CONF_THRESHOLD, verbose=False)[0]
            ts      = now_iso()

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = model.names[cls_id]
                conf   = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "label":      label,
                    "confidence": round(conf, 4),
                    "bbox":       [x1, y1, x2, y2],
                    "x":          round(x1 / w, 4),
                    "y":          round(y1 / h, 4),
                    "w":          round((x2 - x1) / w, 4),
                    "h":          round((y2 - y1) / h, 4),
                })

            # ── Streak filter ─────────────────────────────────────────────────
            cam_streaks     = _streaks.setdefault(camera_id, {})
            detected_labels = {d["label"] for d in detections}

            for label in list(cam_streaks):
                if label not in detected_labels:
                    cam_streaks[label] = 0
            for label in detected_labels:
                cam_streaks[label] = cam_streaks.get(label, 0) + 1

            confirmed = [d for d in detections if cam_streaks.get(d["label"], 0) >= STREAK_REQUIRED]

            # ── Upload frame to GCS ───────────────────────────────────────────
            try:
                frame_url = _upload_frame(img, camera_id, ts)
            except Exception as e:
                print(f"frame upload error: {e}", flush=True)
                frame_url = ""

            out = {
                "ts":         ts,
                "camera_id":  camera_id,
                "seq":        seq,
                "detections": confirmed,
                "frame_url":  frame_url,
            }

            # Always publish — alert-manager writes every inference to Firestore
            future = publisher.publish(topic_path, json.dumps(out).encode("utf-8"))
            msg_id = future.result()
            print(
                f"camera={camera_id} seq={seq} confirmed={len(confirmed)} "
                f"raw={len(detections)} msg_id={msg_id}",
                flush=True,
            )

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
    port   = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Judge server listening on port {port}...", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
