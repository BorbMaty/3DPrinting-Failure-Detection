"""
AlertManager Cloud Function
Triggered by Pub/Sub "detections-out" topic.
Writes alert documents to Firestore "alerts" collection.

Expected Pub/Sub message format (JSON):
{
  "camera_id": "cam1",
  "timestamp": "2026-02-23T20:00:00Z",
  "detections": [
    {"label": "person", "confidence": 0.95, "bbox": [x1, y1, x2, y2]}
  ],
  "frame_id": "uuid-string"
}
"""

import base64
import json
import logging
import os
from datetime import datetime, timezone

import functions_framework
from google.cloud import firestore

logger = logging.getLogger(__name__)

PROJECT = os.environ["GCP_PROJECT"]
DB_NAME = os.environ.get("FIRESTORE_DB", "(default)")
COLLECTION = os.environ.get("ALERTS_COLLECTION", "alerts")

# Reuse client across warm invocations
_db = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=PROJECT, database=DB_NAME)
    return _db


@functions_framework.cloud_event
def handle_detection(cloud_event):
    """Entry point — receives a CloudEvent wrapping a Pub/Sub message."""
    try:
        # Decode Pub/Sub payload
        pubsub_data = cloud_event.data.get("message", {}).get("data", "")
        if not pubsub_data:
            logger.warning("Empty Pub/Sub message data, skipping.")
            return

        payload = json.loads(base64.b64decode(pubsub_data).decode("utf-8"))
    except Exception as exc:
        logger.error("Failed to decode message: %s", exc)
        return  # Return 200 to avoid infinite retry on bad messages

    detections = payload.get("detections", [])
    if not detections:
        # No detections worth storing
        return

    # Build Firestore document
    alert_doc = {
        "camera_id": payload.get("camera_id", "unknown"),
        "frame_id": payload.get("frame_id", ""),
        "timestamp": payload.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "created_at": firestore.SERVER_TIMESTAMP,
        "detections": detections,
        "detection_count": len(detections),
        # Top-level fields useful for dashboard queries
        "labels": list({d.get("label") for d in detections}),
        "max_confidence": max((d.get("confidence", 0) for d in detections), default=0),
    }

    try:
        db = get_db()
        # Auto-ID document in "alerts" collection
        ref = db.collection(COLLECTION).document()
        ref.set(alert_doc)
        logger.info("Alert written: %s (%d detections)", ref.id, len(detections))
    except Exception as exc:
        logger.error("Firestore write failed: %s", exc)
        raise  # Re-raise so Cloud Functions retries
