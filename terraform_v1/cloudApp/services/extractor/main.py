import base64
import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2
from google.cloud import pubsub_v1

PROJECT_ID = os.environ["GCP_PROJECT"]
TOPIC_NAME = os.environ["PUBSUB_TOPIC"]
FPS = float(os.environ.get("FPS", "2"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "70"))

CAMERAS = ["cam1", "cam2", "cam3"]

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

# Per-camera sequence counters
seq_locks = {cam: threading.Lock() for cam in CAMERAS}
seq_counters = {cam: 0 for cam in CAMERAS}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def next_seq(cam):
    with seq_locks[cam]:
        seq_counters[cam] += 1
        return seq_counters[cam]


def run_camera(cam):
    rtsp_url = f"rtsp://localhost:8554/{cam}"
    interval = 1.0 / FPS

    print(f"[{cam}] Starting, connecting to {rtsp_url}", flush=True)

    while True:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print(f"[{cam}] Could not open stream, retrying in 5s...", flush=True)
            time.sleep(5)
            continue

        print(f"[{cam}] Connected", flush=True)

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"[{cam}] Lost stream, reconnecting...", flush=True)
                break

            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                _, buf = cv2.imencode(".jpg", frame, encode_params)
                img_bytes = buf.tobytes()

                seq = next_seq(cam)
                msg = {
                    "ts": now_iso(),
                    "camera_id": cam,
                    "seq": seq,
                    "mime": "image/jpeg",
                    "jpeg_q": JPEG_QUALITY,
                    "data_b64": base64.b64encode(img_bytes).decode("ascii"),
                }

                publisher.publish(topic_path, json.dumps(msg).encode("utf-8"))
                print(f"[{cam}] published seq={seq}", flush=True)

            except Exception as e:
                print(f"[{cam}] error: {e}", flush=True)

            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

        cap.release()
        time.sleep(2)


# Start one thread per camera
threads = []
for cam in CAMERAS:
    t = threading.Thread(target=run_camera, args=(cam,), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.5)  # stagger startup slightly

print(f"Extractor running for cameras: {CAMERAS} at {FPS} FPS", flush=True)

# Keep main thread alive
for t in threads:
    t.join()