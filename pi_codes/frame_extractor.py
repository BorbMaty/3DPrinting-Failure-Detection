import base64
import json
import os
import time
import threading
from datetime import datetime, timezone

import cv2
from google.cloud import pubsub_v1

PROJECT_ID   = os.environ.get("GCP_PROJECT", "printermonitor-488112")
FRAMES_TOPIC = os.environ.get("PUBSUB_TOPIC", "frames-in")
CAPTURE_FPS  = int(os.environ.get("CAPTURE_FPS", "1"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))
FRAME_WIDTH  = int(os.environ.get("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.environ.get("FRAME_HEIGHT", "720"))

CAMERAS = [
    ("cam1", "rtsp://localhost:8554/cam1"),
    ("cam2", "rtsp://localhost:8554/cam2"),
    ("cam3", "rtsp://localhost:8554/cam3"),
]

publisher  = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, FRAMES_TOPIC)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def capture_loop(camera_id: str, rtsp_url: str):
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
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                print(f"[{camera_id}] Read failed, reconnecting...", flush=True)
                cap.release()
                break

            ts = now_iso()
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

            publisher.publish(topic_path, json.dumps({
                "camera_id": camera_id,
                "seq":       seq,
                "ts":        ts,
                "data_b64":  img_b64,
            }).encode("utf-8"))
            print(f"[{camera_id}] published frame seq={seq}", flush=True)

            seq += 1
            elapsed = time.time() - loop_start
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


if __name__ == "__main__":
    for camera_id, rtsp_url in CAMERAS:
        t = threading.Thread(target=capture_loop, args=(camera_id, rtsp_url), daemon=True)
        t.start()

    print("Frame extractor running. Press Ctrl+C to stop.", flush=True)
    while True:
        time.sleep(60)
