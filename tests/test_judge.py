"""
Unit tests for terraform_v2/services/judge/main.py

Covers: now_iso(), _safe_b64decode()
Strategy: heavy cloud/ML dependencies are patched in sys.modules before the
          module is loaded, so no real GCP credentials or YOLO weights are needed.
"""
import base64
import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import ANY, MagicMock

import pytest

# ── Patch env vars and heavy dependencies before importing the module ─────────
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("DETECTIONS_TOPIC", "test-detections")
os.environ.setdefault("MODEL_PATH", "/tmp/fake_model.pt")

for _mod in ["cv2", "numpy", "ultralytics",
             "google", "google.cloud", "google.cloud.pubsub_v1"]:
    sys.modules.setdefault(_mod, MagicMock())

_spec = importlib.util.spec_from_file_location(
    "judge_main",
    Path(__file__).parents[1] / "terraform_v2/services/judge/main.py",
)
judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge)


# ── now_iso ───────────────────────────────────────────────────────────────────

class TestNowIso:
    def test_returns_a_string(self):
        assert isinstance(judge.now_iso(), str)

    def test_is_valid_iso8601(self):
        ts = judge.now_iso()
        dt = datetime.fromisoformat(ts)   # raises ValueError if malformed
        assert dt is not None

    def test_is_utc(self):
        ts = judge.now_iso()
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_two_consecutive_calls_are_non_decreasing(self):
        t1 = datetime.fromisoformat(judge.now_iso())
        t2 = datetime.fromisoformat(judge.now_iso())
        assert t2 >= t1


# ── _safe_b64decode ───────────────────────────────────────────────────────────

class TestSafeB64Decode:
    def test_decodes_standard_padded_input(self):
        original = b"hello world"
        encoded = base64.b64encode(original).decode()
        assert judge._safe_b64decode(encoded) == original

    def test_repairs_single_missing_padding(self):
        # base64 of b"test!" is "dGVzdCE=" — strip one '='
        original = b"test!"
        encoded = base64.b64encode(original).decode().rstrip("=")
        assert judge._safe_b64decode(encoded) == original

    def test_repairs_double_missing_padding(self):
        # base64 of b"ab" is "YWI=" — strip the '='
        original = b"ab"
        encoded = base64.b64encode(original).decode().rstrip("=")
        assert judge._safe_b64decode(encoded) == original

    def test_strips_surrounding_whitespace(self):
        original = b"hello"
        encoded = "  " + base64.b64encode(original).decode() + "\n"
        assert judge._safe_b64decode(encoded) == original

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError, match="empty base64 input"):
            judge._safe_b64decode("")

    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="empty base64 input"):
            judge._safe_b64decode(None)

    def test_whitespace_only_raises_value_error(self):
        with pytest.raises(ValueError, match="empty base64 input"):
            judge._safe_b64decode("   ")


# ── HTTP Handler helpers ──────────────────────────────────────────────────────

def _make_get_handler(path: str):
    h = judge.Handler.__new__(judge.Handler)
    h.path = path
    h.wfile = io.BytesIO()
    h.send_response = MagicMock()
    h.send_header = MagicMock()
    h.end_headers = MagicMock()
    return h


def _make_post_handler(body: bytes):
    h = judge.Handler.__new__(judge.Handler)
    h.path = "/predict"
    h.headers = {"Content-Length": str(len(body))}
    h.rfile = io.BytesIO(body)
    h.wfile = io.BytesIO()
    h.send_response = MagicMock()
    h.send_header = MagicMock()
    h.end_headers = MagicMock()
    return h


def _configure_successful_inference():
    """Set up cv2 and model mocks for a successful POST /predict call."""
    mock_img = MagicMock()
    mock_img.shape = (480, 640, 3)
    judge.cv2.imdecode.return_value = mock_img

    mock_results = MagicMock()
    mock_results.boxes = []          # no detections keeps assertions simple
    judge.model.return_value = [mock_results]
    judge.model.names = {}


# ── Handler.do_GET ────────────────────────────────────────────────────────────

class TestHandlerGet:
    def test_healthz_returns_200(self):
        h = _make_get_handler("/healthz")
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_health_returns_200(self):
        h = _make_get_handler("/health")
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_root_returns_200(self):
        h = _make_get_handler("/")
        h.do_GET()
        h.send_response.assert_called_with(200)

    def test_unknown_path_returns_404(self):
        h = _make_get_handler("/unknown")
        h.do_GET()
        h.send_response.assert_called_with(404)


# ── Handler.do_POST ───────────────────────────────────────────────────────────

class TestHandlerPost:
    def setup_method(self):
        judge.cv2.reset_mock()
        judge.publisher.reset_mock()
        _configure_successful_inference()

    def test_vertex_ai_envelope_returns_200(self):
        img_b64 = base64.b64encode(b"fake_image").decode()
        body = json.dumps({
            "instances": [{"data_b64": img_b64, "camera_id": "cam1", "seq": 1}]
        }).encode()
        h = _make_post_handler(body)
        h.do_POST()
        h.send_response.assert_called_with(200)

    def test_vertex_ai_envelope_publishes_to_pubsub(self):
        img_b64 = base64.b64encode(b"fake_image").decode()
        body = json.dumps({
            "instances": [{"data_b64": img_b64, "camera_id": "cam1", "seq": 1}]
        }).encode()
        h = _make_post_handler(body)
        h.do_POST()
        judge.publisher.publish.assert_called_once()

    def test_pubsub_envelope_returns_200(self):
        img_b64 = base64.b64encode(b"fake_image").decode()
        inner = json.dumps({"data_b64": img_b64, "camera_id": "cam2", "seq": 2})
        inner_b64 = base64.b64encode(inner.encode()).decode()
        body = json.dumps({"message": {"data": inner_b64}}).encode()
        h = _make_post_handler(body)
        h.do_POST()
        h.send_response.assert_called_with(200)

    def test_unknown_post_path_returns_404(self):
        h = _make_post_handler(b"")
        h.path = "/wrong"
        h.do_POST()
        h.send_response.assert_called_with(404)

    def test_undecodable_image_returns_500(self):
        img_b64 = base64.b64encode(b"fake_image").decode()
        body = json.dumps({
            "instances": [{"data_b64": img_b64, "camera_id": "cam1", "seq": 1}]
        }).encode()
        judge.cv2.imdecode.return_value = None   # simulate corrupt image
        h = _make_post_handler(body)
        h.do_POST()
        h.send_response.assert_called_with(500)
