"""
Local alert handler — mirrors terraform_v2/services/alert-manager/main.py without
Firestore or functions_framework. Replaces Firestore with an in-memory dict for cooldowns
and appends detection events to local_test.md.

Email sending is kept intact (it's plain SMTP) but is skipped if GMAIL_ADDRESS /
GMAIL_APP_PASSWORD are not set — safe to leave unset for offline testing.
"""

import os
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from zoneinfo import ZoneInfo

CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.35"))
COOLDOWN_SECONDS = int(os.environ.get("COOLDOWN_SECONDS", "60"))
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")

# Matches production alert-manager — use the same set so local behaviour is identical
HIGH_SEV = {"spagetti", "not_sticking", "layer_shift", "warping"}

# Resolved by orchestrator before first call
LOG_FILE = str(Path(__file__).parent.parent / "local_test.md")

# In-memory Firestore replacements
_cooldowns: dict[str, datetime] = {}
_alert_log: list[dict] = []


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def human_time(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts).astimezone(ZoneInfo("Europe/Bucharest"))
        return dt.strftime("%a %d %b %Y, %H:%M:%S")
    except Exception:
        return ts


# ── Cooldown (in-memory, mirrors Firestore logic) ─────────────────────────────

def is_on_cooldown(key: str) -> bool:
    if key not in _cooldowns:
        return False
    elapsed = (datetime.now(timezone.utc) - _cooldowns[key]).total_seconds()
    return elapsed < COOLDOWN_SECONDS


def set_cooldown(key: str):
    _cooldowns[key] = datetime.now(timezone.utc)


# ── File log ──────────────────────────────────────────────────────────────────

def append_to_log(camera_id: str, ts: str, seq: int, detections: list):
    det_str = ", ".join(
        f"{d['label']} ({d['confidence'] * 100:.0f}%)" for d in detections
    )
    row = f"| {human_time(ts)} | {camera_id} | {seq} | {det_str} |\n"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(row)
    except Exception as e:
        print(f"[alert] log write error: {e}", flush=True)


# ── Email (optional) ──────────────────────────────────────────────────────────

def send_email(subject: str, body_html: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print(f"[alert] Email skipped (no credentials configured): {subject}", flush=True)
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    msg.attach(MIMEText(body_html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_ADDRESS, GMAIL_ADDRESS, msg.as_string())
        print(f"[alert] Email sent: {subject}", flush=True)
    except Exception as e:
        print(f"[alert] Email error: {e}", flush=True)


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_detection(detection_data: dict):
    """
    Accepts a detection payload dict — same shape as what the judge returns:
      {"ts": str, "camera_id": str, "seq": int, "detections": [...]}
    """
    camera_id = detection_data.get("camera_id", "unknown")
    detections = detection_data.get("detections", [])
    ts = detection_data.get("ts", now_iso())
    seq = detection_data.get("seq", -1)

    detections = [d for d in detections if d.get("confidence", 0) >= CONF_THRESHOLD]
    if not detections:
        return

    _alert_log.append({"camera_id": camera_id, "detections": detections, "ts": ts, "seq": seq})
    append_to_log(camera_id, ts, seq, detections)

    labels_str = ", ".join(
        f"{d['label']} ({d['confidence'] * 100:.0f}%)" for d in detections
    )
    print(f"[alert] DETECTION  camera={camera_id} seq={seq} | {labels_str}", flush=True)

    high_sev = [d for d in detections if d.get("label") in HIGH_SEV]
    if not high_sev:
        return

    if is_on_cooldown(camera_id):
        print(f"[alert] HIGH-SEV suppressed by cooldown ({COOLDOWN_SECONDS}s) for {camera_id}", flush=True)
        return

    rows = "".join(
        f"<tr><td>{d['label']}</td><td>{d['confidence'] * 100:.0f}%</td></tr>"
        for d in high_sev
    )
    body = f"""
    <h2>Printer Defect Detected (LOCAL TEST)</h2>
    <p><b>Camera:</b> {camera_id}<br><b>Time:</b> {human_time(ts)}</p>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr><th>Class</th><th>Confidence</th></tr>
      {rows}
    </table>
    """
    send_email(f"[LOCAL TEST] Printer Alert — {camera_id}", body)
    set_cooldown(camera_id)
    print(f"[alert] HIGH-SEV fired: {[d['label'] for d in high_sev]}", flush=True)
