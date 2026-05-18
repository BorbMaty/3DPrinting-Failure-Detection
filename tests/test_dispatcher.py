"""
Unit tests for terraform_v2/services/dispatcher/main.py

Covers: _get_auth_header() — token refresh logic, header format
        dispatch_frame()   — payload parsing, Vertex AI body structure,
                             missing-image guard, HTTP error re-raise
Strategy: google.auth and requests are mocked at sys.modules before import so
          no real GCP credentials or network calls are made.
"""
import base64
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Env vars ──────────────────────────────────────────────────────────────────
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("VERTEX_ENDPOINT_ID", "test-endpoint-123")
os.environ.setdefault("VERTEX_REGION", "europe-west1")

# ── Mock google.auth before import ───────────────────────────────────────────
_mock_creds = MagicMock()
_mock_creds.valid = True
_mock_creds.token = "initial-token"

_mock_google_auth = MagicMock()
_mock_google_auth.default.return_value = (_mock_creds, "test-project")

sys.modules.setdefault("google.auth", _mock_google_auth)
sys.modules.setdefault("google.auth.transport", MagicMock())
sys.modules.setdefault("google.auth.transport.requests", MagicMock())

# functions_framework: identity decorator
_mock_ff = sys.modules.get("functions_framework") or MagicMock()
_mock_ff.cloud_event = lambda f: f
sys.modules.setdefault("functions_framework", _mock_ff)

# requests: mock with real exception classes so except clauses work
class _HttpError(Exception):
    def __init__(self, msg="", status_code=None):
        super().__init__(msg)
        self.response = MagicMock(status_code=status_code) if status_code else None

_mock_requests = MagicMock()
_mock_requests.exceptions.RequestException = Exception
_mock_requests.exceptions.HTTPError = _HttpError
sys.modules.setdefault("requests", _mock_requests)

_spec = importlib.util.spec_from_file_location(
    "dispatcher",
    Path(__file__).parents[1] / "terraform_v2/services/dispatcher/main.py",
)
disp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(disp)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fresh_ts():
    return datetime.now(timezone.utc).isoformat()


def _stale_ts(seconds=60):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _make_cloud_event(payload: dict):
    data_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    event = MagicMock()
    event.data = {"message": {"data": data_b64}}
    return event


# ── _get_auth_header ──────────────────────────────────────────────────────────

class TestGetAuthHeader:
    def setup_method(self):
        disp._creds.reset_mock()

    def test_does_not_refresh_when_credentials_are_valid(self):
        disp._creds.valid = True
        disp._creds.token = "valid-token"
        disp._get_auth_header()
        disp._creds.refresh.assert_not_called()

    def test_refreshes_token_when_credentials_are_invalid(self):
        disp._creds.valid = False
        disp._get_auth_header()
        disp._creds.refresh.assert_called_once()

    def test_returns_bearer_authorization_header(self):
        disp._creds.valid = True
        disp._creds.token = "my-access-token"
        headers = disp._get_auth_header()
        assert headers["Authorization"] == "Bearer my-access-token"
        assert headers["Content-Type"] == "application/json"


# ── dispatch_frame ────────────────────────────────────────────────────────────

class TestDispatchFrame:
    def setup_method(self):
        disp.requests.post.reset_mock(side_effect=True, return_value=True)
        disp._creds.valid = True
        disp._creds.token = "test-token"

    def test_skips_when_payload_has_no_image_data(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 1, "ts": _fresh_ts(),
            # no data_b64 or image_b64
        })
        disp.dispatch_frame(event)
        disp.requests.post.assert_not_called()

    def test_posts_to_vertex_ai_with_instances_envelope(self):
        event = _make_cloud_event({
            "camera_id": "cam2", "seq": 5, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"predictions": [{"detections": []}]}
        disp.requests.post.return_value = mock_resp

        disp.dispatch_frame(event)

        disp.requests.post.assert_called_once()
        _, kwargs = disp.requests.post.call_args
        body = kwargs["json"]
        assert "instances" in body
        instance = body["instances"][0]
        assert instance["data_b64"] == "AAAABBBB"
        assert instance["camera_id"] == "cam2"
        assert instance["seq"] == 5

    def test_reraises_network_error_so_pubsub_can_retry(self):
        event = _make_cloud_event({
            "camera_id": "cam3", "seq": 2, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        disp.requests.post.side_effect = Exception("connection refused")

        with pytest.raises(Exception, match="connection refused"):
            disp.dispatch_frame(event)

    def test_reraises_unexpected_http_error_so_pubsub_can_retry(self):
        event = _make_cloud_event({
            "camera_id": "cam3", "seq": 2, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        disp.requests.post.side_effect = _HttpError("500 Internal Server Error", status_code=500)

        with pytest.raises(_HttpError):
            disp.dispatch_frame(event)

    def test_drops_404_without_retry(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 3, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        disp.requests.post.side_effect = _HttpError("404 Not Found", status_code=404)
        disp.dispatch_frame(event)  # must not raise

    def test_drops_503_without_retry(self):
        event = _make_cloud_event({
            "camera_id": "cam2", "seq": 4, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        disp.requests.post.side_effect = _HttpError("503 Service Unavailable", status_code=503)
        disp.dispatch_frame(event)  # must not raise

    def test_malformed_cloud_event_does_not_raise(self):
        bad_event = MagicMock()
        bad_event.data = {"message": {"data": "!!!not-valid-base64!!!"}}
        disp.dispatch_frame(bad_event)   # must not propagate

    def test_accepts_image_b64_field_as_fallback(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 3, "ts": _fresh_ts(),
            "image_b64": "CCCCDDDD",   # alternative field name
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"predictions": [{"detections": []}]}
        disp.requests.post.return_value = mock_resp

        disp.dispatch_frame(event)

        _, kwargs = disp.requests.post.call_args
        assert kwargs["json"]["instances"][0]["data_b64"] == "CCCCDDDD"

    def test_drops_stale_frame_without_calling_vertex(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 99, "ts": _stale_ts(seconds=60),
            "data_b64": "AAAABBBB",
        })
        disp.dispatch_frame(event)
        disp.requests.post.assert_not_called()

    def test_forwards_fresh_frame_to_vertex(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 100, "ts": _fresh_ts(),
            "data_b64": "AAAABBBB",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"predictions": [{"detections": []}]}
        disp.requests.post.return_value = mock_resp

        disp.dispatch_frame(event)
        disp.requests.post.assert_called_once()

    def test_proceeds_when_ts_is_missing(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 101,
            # no ts field — should still forward
            "data_b64": "AAAABBBB",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"predictions": [{"detections": []}]}
        disp.requests.post.return_value = mock_resp

        disp.dispatch_frame(event)
        disp.requests.post.assert_called_once()

    def test_proceeds_when_ts_is_malformed(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "seq": 102, "ts": "not-a-date",
            "data_b64": "AAAABBBB",
        })
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"predictions": [{"detections": []}]}
        disp.requests.post.return_value = mock_resp

        disp.dispatch_frame(event)
        disp.requests.post.assert_called_once()
