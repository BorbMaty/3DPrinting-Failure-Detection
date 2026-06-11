"""
Unit tests for terraform_v2/services/alert-manager/main.py

Covers: human_time(), is_on_cooldown(), set_cooldown(),
        send_email(), handle_detection(), handle_budget_alert()
Strategy: Firestore client and functions_framework are patched in sys.modules
          before the module is loaded; smtplib.SMTP_SSL is patched per-test.
"""
import base64
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Patch env vars and module-level dependencies before importing ─────────────
os.environ.setdefault("GMAIL_ADDRESS", "test@example.com")
os.environ.setdefault("GMAIL_APP_PASSWORD", "secret")
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("CONF_THRESHOLD", "0.35")
os.environ.setdefault("COOLDOWN_SECONDS", "60")

_mock_ff = MagicMock()
_mock_ff.cloud_event = lambda f: f
sys.modules.setdefault("functions_framework", _mock_ff)

for _mod in ["google", "google.cloud", "google.cloud.firestore"]:
    sys.modules.setdefault(_mod, MagicMock())

_spec = importlib.util.spec_from_file_location(
    "alert_manager",
    Path(__file__).parents[1] / "terraform_v2/services/alert-manager/main.py",
)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)


# ── Shared helper ─────────────────────────────────────────────────────────────

def _make_cloud_event(payload: dict):
    data_b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    event = MagicMock()
    event.data = {"message": {"data": data_b64}}
    return event


def _collections_written(db_mock) -> list[str]:
    """Return the collection names passed to db.collection() in the last call."""
    return [c[0][0] for c in db_mock.collection.call_args_list]


# ── human_time ────────────────────────────────────────────────────────────────

class TestHumanTime:
    def test_converts_utc_to_bucharest_time(self):
        result = am.human_time("2026-03-09T17:38:05+00:00")
        assert "19:38:05" in result

    def test_output_contains_date_parts(self):
        result = am.human_time("2026-03-09T17:38:05+00:00")
        assert "Mar" in result
        assert "2026" in result
        assert "09" in result

    def test_invalid_input_returns_original_string(self):
        bad = "not-a-timestamp"
        assert am.human_time(bad) == bad


# ── is_on_cooldown ────────────────────────────────────────────────────────────

class TestIsOnCooldown:
    def setup_method(self):
        am.db.reset_mock()

    def _set_doc(self, exists: bool, last_sent: str = None):
        mock_doc = MagicMock()
        mock_doc.exists = exists
        if last_sent:
            mock_doc.to_dict.return_value = {"last_sent": last_sent}
        am.db.collection.return_value.document.return_value.get.return_value = mock_doc

    def test_returns_false_when_document_does_not_exist(self):
        self._set_doc(exists=False)
        assert am.is_on_cooldown("cam1") is False

    def test_returns_true_when_sent_recently(self):
        recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        self._set_doc(exists=True, last_sent=recent)
        assert am.is_on_cooldown("cam1") is True

    def test_returns_false_when_cooldown_has_expired(self):
        old = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
        self._set_doc(exists=True, last_sent=old)
        assert am.is_on_cooldown("cam1") is False

    def test_boundary_just_inside_cooldown(self):
        just_inside = (datetime.now(timezone.utc) - timedelta(seconds=59)).isoformat()
        self._set_doc(exists=True, last_sent=just_inside)
        assert am.is_on_cooldown("cam1") is True


# ── set_cooldown ──────────────────────────────────────────────────────────────

class TestSetCooldown:
    def setup_method(self):
        am.db.reset_mock()

    def test_writes_last_sent_field_to_firestore(self):
        am.set_cooldown("cam2")
        mock_set = am.db.collection.return_value.document.return_value.set
        mock_set.assert_called_once()
        written = mock_set.call_args[0][0]
        assert "last_sent" in written

    def test_last_sent_value_is_valid_iso8601(self):
        am.set_cooldown("cam2")
        mock_set = am.db.collection.return_value.document.return_value.set
        written = mock_set.call_args[0][0]
        datetime.fromisoformat(written["last_sent"])

    def test_uses_camera_id_as_document_key(self):
        am.set_cooldown("cam3")
        am.db.collection.return_value.document.assert_called_with("cam3")


# ── send_email ────────────────────────────────────────────────────────────────

class TestSendEmail:
    def test_connects_to_gmail_ssl(self):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_ssl.return_value.__enter__.return_value = MagicMock()
            am.send_email("Subject", "<p>body</p>")
            mock_ssl.assert_called_once_with("smtp.gmail.com", 465)

    def test_logs_in_with_configured_credentials(self):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_server = MagicMock()
            mock_ssl.return_value.__enter__.return_value = mock_server
            am.send_email("Subject", "<p>body</p>")
            mock_server.login.assert_called_once_with("test@example.com", "secret")

    def test_subject_appears_in_sent_message(self):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_server = MagicMock()
            mock_ssl.return_value.__enter__.return_value = mock_server
            am.send_email("Defect Alert!", "<p>body</p>")
            _, _, raw_msg = mock_server.sendmail.call_args[0]
            assert "Defect Alert!" in raw_msg

    def test_does_not_raise_on_smtp_connection_error(self):
        with patch("smtplib.SMTP_SSL", side_effect=Exception("connection refused")):
            am.send_email("Subject", "<p>body</p>")

    def test_returns_true_on_success(self):
        with patch("smtplib.SMTP_SSL") as mock_ssl:
            mock_ssl.return_value.__enter__.return_value = MagicMock()
            assert am.send_email("Subject", "<p>body</p>") is True

    def test_returns_false_on_smtp_error(self):
        with patch("smtplib.SMTP_SSL", side_effect=Exception("connection refused")):
            assert am.send_email("Subject", "<p>body</p>") is False


# ── handle_detection ──────────────────────────────────────────────────────────

class TestHandleDetection:
    def setup_method(self):
        am.db.reset_mock()

    def test_filters_out_detections_below_confidence_threshold(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 1,
            "detections": [{"label": "spagetti", "confidence": 0.10}],
        })
        with patch.object(am, "send_email") as mock_email:
            am.handle_detection(event)

        # inferences always written; alerts never written for low-confidence
        cols = _collections_written(am.db)
        assert "inferences" in cols
        assert "alerts" not in cols
        mock_email.assert_not_called()

    def test_writes_to_inferences_for_every_message(self):
        event = _make_cloud_event({
            "camera_id": "cam2", "ts": "2026-01-01T00:00:00+00:00", "seq": 2,
            "detections": [],
        })
        with patch.object(am, "send_email"):
            am.handle_detection(event)

        cols = _collections_written(am.db)
        assert "inferences" in cols

    def test_writes_to_alerts_when_confidence_passes(self):
        event = _make_cloud_event({
            "camera_id": "cam2", "ts": "2026-01-01T00:00:00+00:00", "seq": 3,
            "detections": [{"label": "spagetti", "confidence": 0.90}],
        })
        with patch.object(am, "send_email"), \
             patch.object(am, "is_on_cooldown", return_value=True):
            am.handle_detection(event)

        cols = _collections_written(am.db)
        assert "inferences" in cols
        assert "alerts" in cols

    def test_no_email_for_low_severity_label(self):
        event = _make_cloud_event({
            "camera_id": "cam3", "ts": "2026-01-01T00:00:00+00:00", "seq": 4,
            "detections": [{"label": "stringing", "confidence": 0.80}],
        })
        with patch.object(am, "send_email") as mock_email:
            am.handle_detection(event)

        mock_email.assert_not_called()

    def test_sends_email_for_high_severity_detection(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 5,
            "detections": [{"label": "warping", "confidence": 0.85}],
        })
        with patch.object(am, "send_email") as mock_email, \
             patch.object(am, "is_on_cooldown", return_value=False), \
             patch.object(am, "set_cooldown"):
            am.handle_detection(event)

        mock_email.assert_called_once()
        subject = mock_email.call_args[0][0]
        assert "cam1" in subject

    def test_no_email_when_on_cooldown(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 6,
            "detections": [{"label": "not_sticking", "confidence": 0.75}],
        })
        with patch.object(am, "send_email") as mock_email, \
             patch.object(am, "is_on_cooldown", return_value=True):
            am.handle_detection(event)

        mock_email.assert_not_called()

    def test_sets_cooldown_after_sending_email(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 7,
            "detections": [{"label": "layer_shift", "confidence": 0.80}],
        })
        with patch.object(am, "send_email"), \
             patch.object(am, "is_on_cooldown", return_value=False), \
             patch.object(am, "set_cooldown") as mock_set_cd:
            am.handle_detection(event)

        mock_set_cd.assert_called_once_with("global_email")

    def test_no_cooldown_when_email_fails(self):
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 7,
            "detections": [{"label": "layer_shift", "confidence": 0.80}],
        })
        with patch.object(am, "send_email", return_value=False), \
             patch.object(am, "is_on_cooldown", return_value=False), \
             patch.object(am, "set_cooldown") as mock_set_cd:
            am.handle_detection(event)

        mock_set_cd.assert_not_called()

    def test_all_high_sev_labels_trigger_email(self):
        for label in ("spagetti", "not_sticking", "layer_shift", "warping"):
            am.db.reset_mock()
            event = _make_cloud_event({
                "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 8,
                "detections": [{"label": label, "confidence": 0.80}],
            })
            with patch.object(am, "send_email") as mock_email, \
                 patch.object(am, "is_on_cooldown", return_value=False), \
                 patch.object(am, "set_cooldown"):
                am.handle_detection(event)

            mock_email.assert_called_once(), f"{label} should trigger email"

    def test_detection_at_exact_threshold_passes(self):
        am.db.reset_mock()
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 9,
            "detections": [{"label": "spagetti", "confidence": 0.35}],
        })
        with patch.object(am, "send_email"), \
             patch.object(am, "is_on_cooldown", return_value=True):
            am.handle_detection(event)

        cols = _collections_written(am.db)
        assert "alerts" in cols

    def test_detection_just_below_threshold_is_filtered(self):
        am.db.reset_mock()
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 10,
            "detections": [{"label": "spagetti", "confidence": 0.3499}],
        })
        with patch.object(am, "send_email") as mock_email:
            am.handle_detection(event)

        cols = _collections_written(am.db)
        assert "inferences" in cols
        assert "alerts" not in cols
        mock_email.assert_not_called()

    def test_empty_detections_writes_to_inferences_only(self):
        am.db.reset_mock()
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 11,
            "detections": [],
        })
        with patch.object(am, "send_email") as mock_email:
            am.handle_detection(event)

        cols = _collections_written(am.db)
        assert "inferences" in cols
        assert "alerts" not in cols
        mock_email.assert_not_called()

    def test_frame_url_stored_in_inferences(self):
        am.db.reset_mock()
        url = "https://storage.googleapis.com/printermonitor-488112-frames/frames/cam1/ts.jpg"
        event = _make_cloud_event({
            "camera_id": "cam1", "ts": "2026-01-01T00:00:00+00:00", "seq": 12,
            "detections": [], "frame_url": url,
        })
        am.handle_detection(event)

        inferences_add = am.db.collection.return_value.add
        inferences_add.assert_called()
        written = inferences_add.call_args_list[0][0][0]
        assert written["frame_url"] == url

    def test_malformed_cloud_event_does_not_raise(self):
        bad_event = MagicMock()
        bad_event.data = {"message": {"data": "!!!not-valid-base64!!!"}}
        am.handle_detection(bad_event)


# ── handle_budget_alert ───────────────────────────────────────────────────────

class TestHandleBudgetAlert:
    def setup_method(self):
        am.db.reset_mock()

    def test_writes_budget_data_to_firestore(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(am, "send_email"), \
             patch.object(am, "is_on_cooldown", return_value=True):
            am.handle_budget_alert(event)

        am.db.collection.return_value.add.assert_called_once()
        written = am.db.collection.return_value.add.call_args[0][0]
        assert written["budget_name"] == "monthly-cap"
        assert written["cost_amount"] == 4.50
        assert written["threshold"] == 5.00

    def test_sends_email_when_not_on_cooldown(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(am, "send_email") as mock_email, \
             patch.object(am, "is_on_cooldown", return_value=False), \
             patch.object(am, "set_cooldown"):
            am.handle_budget_alert(event)

        mock_email.assert_called_once()
        subject = mock_email.call_args[0][0]
        assert "monthly-cap" in subject

    def test_no_email_when_on_cooldown(self):
        event = _make_cloud_event({
            "costAmount": 4.50,
            "budgetAmount": 5.00,
            "budgetDisplayName": "monthly-cap",
        })
        with patch.object(am, "send_email") as mock_email, \
             patch.object(am, "is_on_cooldown", return_value=True):
            am.handle_budget_alert(event)

        mock_email.assert_not_called()

    def test_malformed_event_does_not_raise(self):
        bad_event = MagicMock()
        bad_event.data = {"message": {"data": "!!!not-valid-base64!!!"}}
        am.handle_budget_alert(bad_event)
