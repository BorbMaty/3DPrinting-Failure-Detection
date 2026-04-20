"""
Unit tests for terraform_v2/services/frame-extractor/main.py

Covers: capture_loop() — stream lifecycle, payload structure, sequencing
        main()         — URL/ID parsing, thread spawning, missing-config guard
Strategy: cv2 and pubsub are mocked at sys.modules before import; time.sleep is
          patched per-test; the infinite while-loop is broken by having the
          second cv2.VideoCapture() call raise _StopLoop.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import ANY, MagicMock, call, patch

import pytest

# ── Patch env vars and heavy deps before importing ───────────────────────────
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("FRAMES_TOPIC", "test-frames")
os.environ.setdefault("RTSP_URLS", "")          # overridden per-test
os.environ.setdefault("CAPTURE_FPS", "2")

for _mod in ["cv2", "google", "google.cloud", "google.cloud.pubsub_v1"]:
    sys.modules.setdefault(_mod, MagicMock())

_spec = importlib.util.spec_from_file_location(
    "frame_extractor",
    Path(__file__).parents[1] / "terraform_v2/services/frame-extractor/main.py",
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


# ── Sentinel exception to escape the infinite loop ───────────────────────────
class _StopLoop(Exception):
    pass


# ── Shared fake-frame helper ──────────────────────────────────────────────────

def _fake_cap(read_results):
    """Return a mock VideoCapture that opens successfully and yields read_results."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = list(read_results)
    return mock_cap


def _fake_frame():
    """Return a minimal mock frame that cv2.resize / imencode will accept."""
    buf = MagicMock()
    buf.tobytes.return_value = b"\xff\xd8\xff\xe0fake_jpeg"  # fake JPEG bytes
    fe.cv2.resize.return_value = MagicMock()
    fe.cv2.imencode.return_value = (True, buf)
    return MagicMock()


# ── capture_loop: stream lifecycle ────────────────────────────────────────────

class TestCaptureLoop:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def test_retries_when_stream_fails_to_open(self):
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        fe.cv2.VideoCapture.side_effect = [bad_cap, _StopLoop]

        with patch.object(fe.time, "sleep") as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://cam/stream", "cam1")

        mock_sleep.assert_any_call(5)

    def test_releases_cap_when_frame_read_fails(self):
        cap = _fake_cap([(False, None)])           # read fails immediately
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://cam/stream", "cam1")

        cap.release.assert_called_once()

    def test_publishes_on_successful_frame(self):
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://cam/stream", "cam1")

        fe.publisher.publish.assert_called_once()

    def test_payload_contains_required_keys(self):
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://cam/stream", "cam1")

        raw = fe.publisher.publish.call_args[0][1]
        payload = json.loads(raw.decode())
        assert payload["camera_id"] == "cam1"
        assert payload["seq"] == 0
        assert "ts" in payload
        assert "data_b64" in payload

    def test_seq_increments_across_frames(self):
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (True, frame), (True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://cam/stream", "cam1")

        assert fe.publisher.publish.call_count == 3
        seqs = [
            json.loads(c[0][1].decode())["seq"]
            for c in fe.publisher.publish.call_args_list
        ]
        assert seqs == [0, 1, 2]


# ── main(): URL/ID parsing ────────────────────────────────────────────────────

class TestMainParsing:
    def _run_main(self, rtsp_urls: str, camera_ids: str = "cam1,cam2,cam3"):
        """Run main() with patched env globals, threads, and HTTP server."""
        with patch.object(fe, "RTSP_URLS_ENV", rtsp_urls), \
             patch.object(fe, "CAMERA_IDS_ENV", camera_ids), \
             patch.object(fe.threading, "Thread") as mock_thread, \
             patch.object(fe, "HTTPServer") as mock_server_class:
            mock_server_class.return_value.serve_forever.side_effect = KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                fe.main()
        return mock_thread

    def test_raises_when_rtsp_urls_is_empty(self):
        with patch.object(fe, "RTSP_URLS_ENV", ""):
            with pytest.raises(RuntimeError, match="RTSP_URLS"):
                fe.main()

    def test_starts_one_thread_per_camera(self):
        mock_thread = self._run_main("rtsp://a,rtsp://b,rtsp://c")
        assert mock_thread.call_count == 3

    def test_pads_camera_ids_when_fewer_than_urls(self):
        mock_thread = self._run_main(
            rtsp_urls="rtsp://a,rtsp://b,rtsp://c",
            camera_ids="cam1",
        )
        thread_kwargs = [c[1] for c in mock_thread.call_args_list]
        used_ids = [kw["args"][1] for kw in thread_kwargs]
        assert used_ids == ["cam1", "cam2", "cam3"]

    def test_uses_provided_ids_when_count_matches(self):
        mock_thread = self._run_main(
            rtsp_urls="rtsp://a,rtsp://b",
            camera_ids="front,back",
        )
        thread_kwargs = [c[1] for c in mock_thread.call_args_list]
        used_ids = [kw["args"][1] for kw in thread_kwargs]
        assert used_ids == ["front", "back"]


# ── capture_loop: frame timing ────────────────────────────────────────────────

class TestCaptureLoopTiming:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def test_sleeps_remaining_interval_when_frame_is_fast(self):
        # CAPTURE_FPS=2 → interval=0.5s; elapsed=0.1s → sleep_for=0.4s
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        # time.time() is called: (1) start of frame 1, (2) end of frame 1,
        # (3) start of frame 2 — then read() fails and loop exits before a 4th call
        with patch.object(fe.time, "time", side_effect=[0.0, 0.1, 0.0]), \
             patch.object(fe.time, "sleep") as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://test", "cam1")

        sleep_args = [c[0][0] for c in mock_sleep.call_args_list]
        assert any(abs(v - 0.4) < 0.001 for v in sleep_args)

    def test_no_throttle_sleep_when_frame_is_slow(self):
        # elapsed=0.6s > interval=0.5s → sleep_for is negative → no sleep
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "time", side_effect=[0.0, 0.6, 0.0]), \
             patch.object(fe.time, "sleep") as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://test", "cam1")

        # The only allowed sleep is the reconnect sleep(5) — no throttle sleep
        throttle_sleeps = [c[0][0] for c in mock_sleep.call_args_list if c[0][0] != 5]
        assert throttle_sleeps == []


# ── capture_loop: encoding parameters ────────────────────────────────────────

class TestCaptureLoopEncoding:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def test_resize_uses_configured_dimensions(self):
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://test", "cam1")

        fe.cv2.resize.assert_called_once_with(ANY, (fe.FRAME_WIDTH, fe.FRAME_HEIGHT))

    def test_imencode_uses_jpeg_format_and_configured_quality(self):
        frame = _fake_frame()
        cap = _fake_cap([(True, frame), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.capture_loop("rtsp://test", "cam1")

        encode_args = fe.cv2.imencode.call_args[0]
        assert encode_args[0] == ".jpg"
        assert fe.JPEG_QUALITY in encode_args[2]
