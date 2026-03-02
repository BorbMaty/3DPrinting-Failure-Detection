import base64
import json
import os
from datetime import datetime, timezone

import functions_framework
from google.cloud import firestore

PROJECT_ID = os.environ["GCP_PROJECT"]
COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "alerts")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))

db = firestore.Client(project=PROJECT_ID)


@functions_framework.cloud_event
def handle_detection(cloud_event):
    raw = base64.b64decode(cloud_event.data["message"]["data"]).decode("utf-8")
    payload = json.loads(raw)

    camera_id = payload.get("camera_id", "unknown")
    ts = payload.get("ts", datetime.now(timezone.utc).isoformat())
    detections = payload.get("detections", [])

    # Filter by confidence threshold
    significant = [d for d in detections if d["confidence"] >= CONF_THRESHOLD]

    if not significant:
        return

    doc = {
        "camera_id": camera_id,
        "timestamp": ts,
        "detections": significant,
        "labels": list({d["label"] for d in significant}),
        "max_confidence": max(d["confidence"] for d in significant),
        "created_at": firestore.SERVER_TIMESTAMP,
    }

    db.collection(COLLECTION).add(doc)
    print(f"Alert written: camera={camera_id} labels={doc['labels']}", flush=True)
