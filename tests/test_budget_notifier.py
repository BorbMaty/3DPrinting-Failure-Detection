"""
Unit tests for terraform_v2/services/budget-notifier/main.py

Covers: now_iso(), send_fcm_defect(), send_fcm_budget(),
        handle_detection(), handle_budget_alert()
Strategy: Firestore client, firebase_admin, and functions_framework are patched
          in sys.modules before the module is loaded; no real credentials needed.
"""
import base64
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Patch env vars and module-level dependencies before importing ─────────────
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("CONF_THRESHOLD", "0.35")
os.environ.setdefault("FIRESTORE_COLLECTION", "alerts")

_mock_ff = MagicMock()
_mock_ff.cloud_event = lambda f: f
sys.modules.setdefault("functions_framework", _mock_ff)

for _mod in ["google", "google.cloud", "google.cloud.firestore"]:
    sys.modules.setdefault(_mod, MagicMock())

# firebase_admin._apps must be non-empty so initialize_app() is skipped
_mock_fa = MagicMock()
_mock_fa._apps = {"default": MagicMock()}
sys.modules.setdefault("firebase_admin", _mock_fa)

_spec = importlib.util.spec_from_file_location(
    "budget_notifier",
    Path(__file__).parents[1] / "terraform_v2/services/budget-notifier/main.py",
)
bn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bn)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _make_cloud_event(payload: dict):
    data_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    event = MagicMock()
    event.data = {"message": {"data": data_b64}}
    return event


# ── now_iso ───────────────────────────────────────────────────────────────────

class TestNowIso:
    def test_returns_string(self):
        assert isinstance(bn.now_iso(), str)

    def test_is_valid_iso8601(self):
        datetime.fromisoformat(bn.now_iso())

    def test_contains_utc_marker(self):
        result = bn.now_iso()
        assert "+00:00" in result or result.endswith("Z")


# ── send_fcm_defect ───────────────────────────────────────────────────────────

class TestSendFcmDefect:
    def setup_method(self):
        bn.messaging.reset_mock()

    def test_calls_messaging_send_once(self):
        bn.send_fcm_defect("cam1", [{"label": "warping", "confidence": 0.9}])
        bn.messaging.send.assert_called_once()

    def test_uses_warning_title_for_high_severity(self):
        bn.send_fcm_defect("cam1", [{"label": "layer_shift", "confidence": 0.85}])
        title = bn.messaging.Notification.call_args.kwargs["title"]
        assert "⚠️" in title

    def test_uses_anomaly_title_for_low_severity(self):
        bn.send_fcm_defect("cam1", [{"label": "stringing", "confidence": 0.85}])
        title = bn.messaging.Notification.call_args.kwargs["title"]
        assert "🔍" in title

    def test_does_not_raise_on_send_error(self):
        bn.messaging.send.side_effect = Exception("FCM unavailable")
        bn.send_fcm_defect("cam1", [{"label": "warping", "confidence": 0.9}])
        bn.messaging.send.side_effect = None


# ── send_fcm_budget ───────────────────────────────────────────────────────────

class TestSendFcmBudget:
    def setup_method(self):
        bn.messaging.reset_mock()

    def test_calls_messaging_send_once(self):
        bn.send_fcm_budget(4.50, 5.00)
        bn.messaging.send.assert_called_once()

    def test_does_not_raise_on_send_error(self):
        bn.messaging.send.side_effect = Exception("FCM unavailable")
        bn.send_fcm_budget(4.50, 5.00)
        bn.messaging.send.side_effect = None


# ── handle_detection ──────────────────────────────────────────────────────────

class TestHandleDetection:
    def setup_method(self):
        bn.db.reset_mock()
        bn.messaging.reset_mock()

    def test_early_return_when_no_detections_pass_threshold(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 1,
            "detections": [{"label": "warping", "confidence": 0.10}],
        })
        bn.handle_detection(event)
        bn.db.collection.return_value.add.assert_not_called()

    def test_writes_to_firestore_when_detection_passes_threshold(self):
        event = _make_cloud_event({
            "camera_id": "cam2", "ts": "2026-01-01T00:00:00+00:00", "seq": 2,
            "detections": [{"label": "warping", "confidence": 0.90}],
        })
        bn.handle_detection(event)
        bn.db.collection.return_value.add.assert_called_once()

    def test_firestore_doc_contains_expected_fields(self):
        event = _make_cloud_event({
            "camera_id": "cam3", "ts": "2026-01-01T00:00:00+00:00", "seq": 3,
            "detections": [{"label": "not_sticking", "confidence": 0.80}],
        })
        bn.handle_detection(event)
        written = bn.db.collection.return_value.add.call_args[0][0]
        assert written["camera_id"] == "cam3"
        assert written["seq"] == 3

    def test_sends_fcm_for_high_severity(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 4,
            "detections": [{"label": "spagetti", "confidence": 0.85}],
        })
        with patch.object(bn, "send_fcm_defect") as mock_fcm:
            bn.handle_detection(event)
        mock_fcm.assert_called_once()

    def test_no_fcm_for_low_severity(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 5,
            "detections": [{"label": "stringing", "confidence": 0.80}],
        })
        with patch.object(bn, "send_fcm_defect") as mock_fcm:
            bn.handle_detection(event)
        mock_fcm.assert_not_called()

    def test_detection_at_exact_threshold_passes(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 6,
            "detections": [{"label": "warping", "confidence": 0.35}],
        })
        bn.handle_detection(event)
        bn.db.collection.return_value.add.assert_called_once()

    def test_only_high_sev_detections_forwarded_to_fcm(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 7,
            "detections": [
                {"label": "stringing",   "confidence": 0.90},
                {"label": "layer_shift", "confidence": 0.90},
            ],
        })
        with patch.object(bn, "send_fcm_defect") as mock_fcm:
            bn.handle_detection(event)
        mock_fcm.assert_called_once()
        sent_detections = mock_fcm.call_args[0][1]
        assert all(d["label"] == "layer_shift" for d in sent_detections)

    def test_all_high_sev_labels_trigger_fcm(self):
        for label in ("spagetti", "not_sticking", "layer_shift", "warping"):
            bn.db.reset_mock()
            event = _make_cloud_event({
                "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 8,
                "detections": [{"label": label, "confidence": 0.80}],
            })
            with patch.object(bn, "send_fcm_defect") as mock_fcm:
                bn.handle_detection(event)
            mock_fcm.assert_called_once(), f"{label} should trigger FCM"

    def test_malformed_event_does_not_raise(self):
        bad_event = MagicMock()
        bad_event.data = {"message": {"data": "!!!not-valid-base64!!!"}}
        bn.handle_detection(bad_event)


# ── handle_budget_alert ───────────────────────────────────────────────────────

class TestHandleBudgetAlert:
    def setup_method(self):
        bn.db.reset_mock()
        bn.messaging.reset_mock()

    def test_writes_to_firestore(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(bn, "send_fcm_budget"):
            bn.handle_budget_alert(event)
        bn.db.collection.return_value.add.assert_called_once()

    def test_firestore_doc_contains_budget_fields(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(bn, "send_fcm_budget"):
            bn.handle_budget_alert(event)
        written = bn.db.collection.return_value.add.call_args[0][0]
        assert written["budget_name"] == "monthly-cap"
        assert written["cost_amount"] == 4.50
        assert written["threshold"] == 5.00

    def test_calls_send_fcm_budget_with_amounts(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(bn, "send_fcm_budget") as mock_fcm:
            bn.handle_budget_alert(event)
        mock_fcm.assert_called_once_with(4.50, 5.00)

    def test_malformed_event_does_not_raise(self):
        bad_event = MagicMock()
        bad_event.data = {"message": {"data": "!!!not-valid-base64!!!"}}
        bn.handle_budget_alert(bad_event)
