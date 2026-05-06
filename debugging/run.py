"""
Local debug subscriber. Replaces the dispatcher + cloud judge for end-to-end
testing on a CUDA box: pulls frames from Pub/Sub `frames-in`, runs YOLO on GPU,
prints detections, and optionally saves/visualises frames.

Pi keeps publishing as normal; this just attaches a parallel subscription so
the cloud pipeline state (deployed model, dispatcher errors, ...) is irrelevant.
"""

import argparse
import base64
import json
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from google.cloud import pubsub_v1
from ultralytics import YOLO

PROJECT_ID = "printermonitor-488112"
SUBSCRIPTION_ID = "frames-in-debug-local"
DEFAULT_MODEL = (
    Path(__file__).parent.parent
    / "terraform_v2" / "services" / "judge" / "best.pt"
)
FRAMES_DIR = Path(__file__).parent / "frames"


def parse_args():
    p = argparse.ArgumentParser(description="Local Pub/Sub → YOLO debug pipeline")
    p.add_argument("--model", default=str(DEFAULT_MODEL),
                   help="Path to best.pt (default: repo's terraform_v2/.../best.pt)")
    p.add_argument("--conf", type=float, default=0.20,
                   help="Confidence threshold (default 0.20, matches v4 cloud env)")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device: cuda:0, cpu, ... (default cuda:0)")
    p.add_argument("--save-frames", action="store_true",
                   help="Save raw + annotated frames to ./frames/")
    p.add_argument("--show", action="store_true",
                   help="Show a live OpenCV window per camera (requires display)")
    p.add_argument("--max-messages", type=int, default=4,
                   help="Pub/Sub flow control: max in-flight messages (default 4)")
    return p.parse_args()


def annotate(img, detections):
    out = img.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{d['label']} {d['confidence']:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    return out


def main():
    args = parse_args()
    if args.save_frames:
        FRAMES_DIR.mkdir(exist_ok=True)

    print(f"Loading model: {args.model} on {args.device}", flush=True)
    model = YOLO(args.model)
    model.to(args.device)
    print(f"Classes: {model.names}", flush=True)

    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_ID)

    def callback(message):
        try:
            t0 = time.time()
            payload = json.loads(message.data.decode("utf-8"))
            camera_id = payload.get("camera_id", "unknown")
            seq = payload.get("seq", -1)
            img_b64 = payload.get("data_b64") or payload.get("image_b64")

            if not img_b64:
                print(f"[{camera_id}] seq={seq} — no image data", flush=True)
                message.ack()
                return

            img_bytes = base64.b64decode(img_b64)
            img = cv2.imdecode(
                np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                print(f"[{camera_id}] seq={seq} — cv2.imdecode failed", flush=True)
                message.ack()
                return

            h, w = img.shape[:2]
            results = model(img, conf=args.conf, verbose=False)[0]

            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": model.names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [int(v) for v in box.xyxy[0].tolist()],
                })

            dt = (time.time() - t0) * 1000
            summary = (
                ", ".join(f"{d['label']}={d['confidence']:.2f}" for d in detections)
                or "none"
            )
            print(
                f"[{camera_id}] seq={seq} {w}x{h} {dt:.0f}ms "
                f"detections=[{summary}]",
                flush=True,
            )

            if args.save_frames:
                stem = f"{camera_id}_{seq:06d}"
                cv2.imwrite(str(FRAMES_DIR / f"{stem}.jpg"), img)
                if detections:
                    cv2.imwrite(
                        str(FRAMES_DIR / f"{stem}_annot.jpg"),
                        annotate(img, detections),
                    )

            if args.show:
                cv2.imshow(camera_id, annotate(img, detections))
                cv2.waitKey(1)

            message.ack()
        except Exception as e:
            print(f"callback error: {e}", flush=True)
            message.nack()

    flow = pubsub_v1.types.FlowControl(max_messages=args.max_messages)
    streaming = subscriber.subscribe(sub_path, callback=callback, flow_control=flow)
    print(f"Listening on {sub_path}. Ctrl+C to stop.", flush=True)

    def stop(_sig, _frame):
        print("\nStopping...", flush=True)
        streaming.cancel()
        sys.exit(0)
    signal.signal(signal.SIGINT, stop)

    try:
        streaming.result()
    except KeyboardInterrupt:
        streaming.cancel()


if __name__ == "__main__":
    main()
