"""
Dispatcher Cloud Function (Gen2)
Triggered by Pub/Sub frames-in. Forwards frame to Vertex AI judge endpoint.
The judge container runs YOLO and publishes detections to detections-out.
"""

import base64
import json
import os

import functions_framework
from google.auth import default as google_auth_default
from google.auth.transport.requests import Request as GoogleAuthRequest
import requests

PROJECT_ID      = os.environ.get("GCP_PROJECT", "printermonitor-488112")
VERTEX_ENDPOINT = os.environ["VERTEX_ENDPOINT_ID"]
VERTEX_REGION   = os.environ.get("VERTEX_REGION", "europe-west1")

VERTEX_URL = (
    f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
    f"/locations/{VERTEX_REGION}/endpoints/{VERTEX_ENDPOINT}:predict"
)

_creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])


def _get_auth_header():
    if not _creds.valid:
        _creds.refresh(GoogleAuthRequest())
    return {
        "Authorization": f"Bearer {_creds.token}",
        "Content-Type": "application/json",
    }


@functions_framework.cloud_event
def dispatch_frame(cloud_event):
    """Receive frame from Pub/Sub, forward to Vertex AI judge endpoint."""
    try:
        data_b64 = cloud_event.data["message"]["data"]
        payload = json.loads(base64.b64decode(data_b64).decode("utf-8"))
    except Exception as e:
        print(f"Failed to decode message: {e}", flush=True)
        return

    camera_id = payload.get("camera_id", "unknown")
    seq       = payload.get("seq", -1)
    img_b64   = payload.get("data_b64") or payload.get("image_b64")
    ts        = payload.get("ts", "")

    if not img_b64:
        print(f"[{camera_id}] seq={seq} — no image data, skipping", flush=True)
        return

    body = {
        "instances": [{
            "data_b64":  img_b64,
            "camera_id": camera_id,
            "seq":       seq,
            "ts":        ts,
        }]
    }

    try:
        resp = requests.post(
            VERTEX_URL,
            headers=_get_auth_header(),
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        predictions = resp.json().get("predictions", [{}])
        n = len(predictions[0].get("detections", []))
        print(f"[{camera_id}] seq={seq} — {n} detection(s)", flush=True)
    except requests.exceptions.RequestException as e:
        print(f"[{camera_id}] seq={seq} — Vertex AI error: {e}", flush=True)
        raise  # Re-raise so Pub/Sub retries
