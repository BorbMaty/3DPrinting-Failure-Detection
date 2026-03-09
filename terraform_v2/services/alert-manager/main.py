import base64
import json
import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import functions_framework
from google.cloud import firestore

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID           = os.environ.get("GCP_PROJECT", "printermonitor-488112")
FIRESTORE_COLLECTION = os.environ.get("FIRESTORE_COLLECTION", "alerts")
CONF_THRESHOLD       = float(os.environ.get("CONF_THRESHOLD", "0.35"))
COOLDOWN_SECONDS     = int(os.environ.get("COOLDOWN_SECONDS", "60"))

GMAIL_ADDRESS      = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]

# High-severity classes that warrant an email alert
HIGH_SEV = {"spagetti", "not_sticking", "layer_shift", "warping"}

# ── Init ──────────────────────────────────────────────────────────────────────
db = firestore.Client(project=PROJECT_ID)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def human_time(ts: str) -> str:
    """Convert ISO timestamp to readable local time e.g. Mon 09 Mar 2026, 19:38:05."""
    try:
        dt = datetime.fromisoformat(ts).astimezone(ZoneInfo("Europe/Bucharest"))
        return dt.strftime("%a %d %b %Y, %H:%M:%S")
    except Exception:
        return ts


# ── Cooldown helpers ──────────────────────────────────────────────────────────
def is_on_cooldown(key: str) -> bool:
    doc = db.collection("alert_cooldowns").document(key).get()
    if not doc.exists:
        return False
    last_sent = datetime.fromisoformat(doc.to_dict()["last_sent"])
    elapsed = (datetime.now(timezone.utc) - last_sent).total_seconds()
    return elapsed < COOLDOWN_SECONDS


def set_cooldown(key: str):
    db.collection("alert_cooldowns").document(key).set({
        "last_sent": now_iso()
    })


# ── Email helper ──────────────────────────────────────────────────────────────
def send_email(subject: str, body_html: str):
    """Send an email from and to the same Gmail address."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = GMAIL_ADDRESS
    msg["To"]      = GMAIL_ADDRESS
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
        print(f"Email sent: {subject}", flush=True)
    except Exception as e:
        print(f"Email error: {e}", flush=True)


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
        return

    # Write to Firestore
    db.collection(FIRESTORE_COLLECTION).add({
        "camera_id":  camera_id,
        "detections": detections,
        "timestamp":  ts,
        "seq":        seq,
        "created_at": firestore.SERVER_TIMESTAMP,
    })
    print(f"Alert written: camera={camera_id} detections={len(detections)}", flush=True)

    # Send email for high-severity detections only, with cooldown
    high_sev = [d for d in detections if d.get("label") in HIGH_SEV]
    if high_sev and not is_on_cooldown(camera_id):
        rows = "".join(
            f"<tr><td>{d['label']}</td><td>{d.get('confidence', 0)*100:.0f}%</td></tr>"
            for d in high_sev
        )
        body = f"""
        <h2>🚨 Printer Defect Detected</h2>
        <p><b>Camera:</b> {camera_id}<br>
        <b>Time:</b> {human_time(ts)}</p>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Class</th><th>Confidence</th></tr>
          {rows}
        </table>
        <p><a href="https://printermonitor-488112.web.app">Open Dashboard</a></p>
        """
        send_email(f"🚨 Printer Alert — {camera_id}", body)
        set_cooldown(camera_id)


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

    db.collection("budget_alerts").add({
        "budget_name":  budget_name,
        "cost_amount":  budget_amount,
        "threshold":    budget_threshold,
        "created_at":   firestore.SERVER_TIMESTAMP,
    })

    if not is_on_cooldown("budget"):
        body = f"""
        <h2>💸 GCP Budget Alert</h2>
        <p><b>Budget:</b> {budget_name}<br>
        <b>Spent:</b> ${budget_amount:.2f}<br>
        <b>Threshold:</b> ${budget_threshold:.2f}</p>
        """
        send_email(f"💸 GCP Budget Alert — {budget_name}", body)
        set_cooldown("budget")