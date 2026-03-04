import base64
import json
import os
from datetime import datetime, timezone

import functions_framework
from google.cloud import firestore
from firebase_admin import initialize_app, messaging
import firebase_admin

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID           = os.environ.get("GCP_PROJECT", "printermonitor-488112")
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "alerts")
CONF_THRESHOLD       = float(os.environ.get("CONF_THRESHOLD", "0.35"))

# FCM topic to broadcast to all subscribed dashboard tabs
FCM_TOPIC_DEFECT = "defect-alerts"
FCM_TOPIC_BUDGET = "budget-alerts"

# High-severity classes that warrant immediate push notification
HIGH_SEV = {"spagetti", "not_sticking", "layer_shift", "warping"}

# ── Init ──────────────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    initialize_app()

db = firestore.Client(project=PROJECT_ID)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def send_fcm_defect(camera_id: str, detections: list):
    """Send FCM push to all dashboard subscribers on the defect-alerts topic."""
    labels = [d["label"] for d in detections]
    high   = [l for l in labels if l in HIGH_SEV]
    title  = "⚠️ Defect Detected" if high else "🔍 Anomaly Detected"
    body   = f"{camera_id}: {', '.join(labels)}"

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={"type": "defect", "camera_id": camera_id, "labels": ",".join(labels)},
        topic=FCM_TOPIC_DEFECT,
        android=messaging.AndroidConfig(priority="high"),
        webpush=messaging.WebpushConfig(
            notification=messaging.WebpushNotification(title=title, body=body),
            fcm_options=messaging.WebpushFCMOptions(link="https://printermonitor-488112.web.app"),
        ),
    )
    try:
        resp = messaging.send(message)
        print(f"FCM defect sent: {resp}", flush=True)
    except Exception as e:
        print(f"FCM defect error: {e}", flush=True)


def send_fcm_budget(budget_amount: float, threshold: float):
    """Send FCM push to all dashboard subscribers on the budget-alerts topic."""
    title = "💸 GCP Budget Alert"
    body  = f"Spend ${budget_amount:.2f} has exceeded ${threshold:.2f} threshold"

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data={"type": "budget", "amount": str(budget_amount), "threshold": str(threshold)},
        topic=FCM_TOPIC_BUDGET,
        android=messaging.AndroidConfig(priority="high"),
        webpush=messaging.WebpushConfig(
            notification=messaging.WebpushNotification(title=title, body=body),
            fcm_options=messaging.WebpushFCMOptions(link="https://printermonitor-488112.web.app"),
        ),
    )
    try:
        resp = messaging.send(message)
        print(f"FCM budget sent: {resp}", flush=True)
    except Exception as e:
        print(f"FCM budget error: {e}", flush=True)


# ── Defect detection handler ──────────────────────────────────────────────────
@functions_framework.cloud_event
def handle_detection(cloud_event):
    """Triggered by Pub/Sub detections-out topic."""
    try:
        data_b64 = cloud_event.data["message"]["data"]
        payload  = json.loads(base64.b64decode(data_b64).decode("utf-8"))
    except Exception as e:
        print(f"Failed to decode message: {e}", flush=True)
        return

    camera_id  = payload.get("camera_id", "unknown")
    detections = payload.get("detections", [])
    ts         = payload.get("ts", now_iso())
    seq        = payload.get("seq", -1)

    # Filter by confidence threshold
    detections = [d for d in detections if d.get("confidence", 0) >= CONF_THRESHOLD]

    if not detections:
        return  # Nothing to alert on

    # Write to Firestore
    doc = {
        "camera_id":  camera_id,
        "detections": detections,
        "timestamp":  ts,
        "seq":        seq,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    db.collection(FIRESTORE_COLLECTION).add(doc)
    print(f"Alert written: camera={camera_id} detections={len(detections)}", flush=True)

    # Send push notification for high-severity detections
    high_sev = [d for d in detections if d.get("label") in HIGH_SEV]
    if high_sev:
        send_fcm_defect(camera_id, high_sev)


# ── Budget alert handler ──────────────────────────────────────────────────────
@functions_framework.cloud_event
def handle_budget_alert(cloud_event):
    """Triggered by Pub/Sub budget-notifications topic."""
    try:
        data_b64 = cloud_event.data["message"]["data"]
        payload  = json.loads(base64.b64decode(data_b64).decode("utf-8"))
    except Exception as e:
        print(f"Failed to decode budget message: {e}", flush=True)
        return

    budget_amount    = payload.get("costAmount", 0)
    budget_threshold = payload.get("budgetAmount", 5)
    budget_name      = payload.get("budgetDisplayName", "unknown")

    print(f"Budget alert: {budget_name} ${budget_amount} / ${budget_threshold}", flush=True)

    # Write to Firestore for dashboard log
    db.collection("budget_alerts").add({
        "budget_name":  budget_name,
        "cost_amount":  budget_amount,
        "threshold":    budget_threshold,
        "created_at":   firestore.SERVER_TIMESTAMP,
    })

    # Send FCM push
    send_fcm_budget(budget_amount, budget_threshold)