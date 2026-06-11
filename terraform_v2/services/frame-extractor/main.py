import base64
import json
import os
import time
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
from google.cloud import pubsub_v1
from google.cloud import firestore as _firestore_lib

PROJECT_ID   = os.environ.get("GCP_PROJECT", "printermonitor-488112")
FRAMES_TOPIC = os.environ.get("FRAMES_TOPIC", os.environ.get("PUBSUB_TOPIC", "frames-in"))
CAPTURE_FPS  = float(os.environ.get("CAPTURE_FPS", "0.1"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "100"))
FRAME_WIDTH  = int(os.environ.get("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "720"))

# Comma-separated RTSP URLs and matching camera IDs.
# Defaults match the MediaMTX instance running on the Pi.
RTSP_URLS_ENV = os.environ.get("RTSP_URLS", os.environ.get(
    "RTSP_URL",
    "rtsp://localhost:8554/cam1,rtsp://localhost:8554/cam2,rtsp://localhost:8554/cam3",
))
CAMERA_IDS_ENV = os.environ.get("CAMERA_IDS", "cam1,cam2,cam3")

publisher  = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, FRAMES_TOPIC)

_fs_client = None
_extraction_enabled: bool = True
_extraction_checked_at: float = 0.0


def _firestore():
    global _fs_client
    if _fs_client is None:
        _fs_client = _firestore_lib.Client(project=PROJECT_ID)
    return _fs_client


def _is_extraction_enabled() -> bool:
    """Check Firestore system_state/extraction.enabled (cached 5 s)."""
    global _extraction_enabled, _extraction_checked_at
    now = time.time()
    if now - _extraction_checked_at < 5.0:
        return _extraction_enabled
    try:
        snap = _firestore().collection("system_state").document("extraction").get()
        _extraction_enabled = snap.to_dict().get("enabled", True) if snap.exists else True
    except Exception as e:
        print(f"[system_state] Firestore check failed, keeping current state: {e}", flush=True)
    _extraction_checked_at = now
    return _extraction_enabled


def now_iso():
    return datetime.now(timezone.utc).isoformat()


class FrameReader(threading.Thread):
    """Drains the RTSP stream at full rate, keeping only the newest frame.

    OpenCV/FFmpeg buffers RTSP frames internally; reading slower than the
    camera produces them serves progressively staler frames while the
    publish timestamp claims they are fresh. Reading continuously and
    stamping `ts` at grab time keeps the published timestamp honest, so
    the dispatcher's staleness filter (MAX_FRAME_AGE_S) sees real ages.
    """

    def __init__(self, camera_id: str, rtsp_url: str):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.rtsp_url  = rtsp_url
        self._lock     = threading.Lock()
        self._frame    = None
        self._ts       = ""

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.rtsp_url)
            if not cap.isOpened():
                print(f"[{self.camera_id}] Failed to open stream, retrying in 5s...", flush=True)
                time.sleep(5)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[{self.camera_id}] Stream opened.", flush=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{self.camera_id}] Frame read failed, reconnecting...", flush=True)
                    cap.release()
                    break
                with self._lock:
                    self._frame = frame
                    self._ts    = now_iso()

    def latest(self):
        """Return (frame, grab_ts), consuming the frame so a stalled stream
        is never republished as a series of identical "fresh" frames."""
        with self._lock:
            frame, ts = self._frame, self._ts
            self._frame = None
            return frame, ts


def publish_loop(reader: FrameReader):
    interval = 1.0 / CAPTURE_FPS
    seq = 0

    while True:
        loop_start = time.time()

        # The reader keeps draining the stream while capture is OFF, so no
        # stale backlog builds up during a pause.
        if _is_extraction_enabled():
            frame, ts = reader.latest()
            if frame is not None:
                h0, w0 = frame.shape[:2]
                if w0 > FRAME_WIDTH or h0 > FRAME_HEIGHT:
                    scale = min(FRAME_WIDTH / w0, FRAME_HEIGHT / h0)
                    frame = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

                publisher.publish(topic_path, json.dumps({
                    "camera_id": reader.camera_id,
                    "seq":       seq,
                    "ts":        ts,
                    "data_b64":  img_b64,
                }).encode("utf-8"))
                print(f"[{reader.camera_id}] published frame seq={seq}", flush=True)
                seq += 1

        elapsed   = time.time() - loop_start
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

    for cid, url in zip(ids, urls):
        reader = FrameReader(cid, url)
        reader.start()
        threading.Thread(target=publish_loop, args=(reader,), daemon=True).start()

    # Health check HTTP server
    port   = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), Health)
    print(f"Health server on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
