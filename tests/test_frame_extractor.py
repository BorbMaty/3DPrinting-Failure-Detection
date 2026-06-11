"""
Unit tests for terraform_v2/services/frame-extractor/main.py

Covers: FrameReader        — stream lifecycle, latest-frame semantics
        publish_loop()     — payload structure, sequencing, capture toggle
        _is_extraction_enabled() — Firestore toggle caching and failure modes
        main()             — URL/ID parsing, thread spawning, missing-config guard
Strategy: cv2, pubsub and firestore are mocked at sys.modules before import;
          time.sleep is patched per-test; infinite loops are broken by raising
          _StopLoop from a mocked call (second VideoCapture / time.sleep).
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
os.environ.setdefault("CAPTURE_FPS", "2")

for _mod in ["cv2", "google", "google.cloud", "google.cloud.pubsub_v1",
             "google.cloud.firestore"]:
    sys.modules.setdefault(_mod, MagicMock())

_spec = importlib.util.spec_from_file_location(
    "frame_extractor",
    Path(__file__).parents[1] / "terraform_v2/services/frame-extractor/main.py",
)
fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fe)


# ── Sentinel exception to escape the infinite loops ──────────────────────────
class _StopLoop(Exception):
    pass


# ── Shared fake-frame helpers ─────────────────────────────────────────────────

def _fake_cap(read_results):
    """Return a mock VideoCapture that opens successfully and yields read_results."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = list(read_results)
    return mock_cap


def _fake_frame():
    """Return a minimal mock frame that cv2.resize / imencode will accept."""
    frame = MagicMock()
    # Use dimensions larger than defaults (1280×720) so the resize branch is exercised
    frame.shape = (1440, 2560, 3)
    buf = MagicMock()
    buf.tobytes.return_value = b"\xff\xd8\xff\xe0fake_jpeg"
    fe.cv2.resize.return_value = MagicMock()
    fe.cv2.imencode.return_value = (True, buf)
    return frame


def _fake_reader(frames):
    """A stub FrameReader whose latest() yields the given (frame, ts) pairs."""
    reader = MagicMock()
    reader.camera_id = "cam1"
    reader.latest.side_effect = list(frames)
    return reader


# ── FrameReader: stream lifecycle ─────────────────────────────────────────────

class TestFrameReader:
    def setup_method(self):
        fe.cv2.reset_mock()

    def test_retries_when_stream_fails_to_open(self):
        bad_cap = MagicMock()
        bad_cap.isOpened.return_value = False
        fe.cv2.VideoCapture.side_effect = [bad_cap, _StopLoop]

        with patch.object(fe.time, "sleep") as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.FrameReader("cam1", "rtsp://cam/stream").run()

        mock_sleep.assert_any_call(5)

    def test_releases_cap_when_frame_read_fails(self):
        cap = _fake_cap([(False, None)])           # read fails immediately
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.FrameReader("cam1", "rtsp://cam/stream").run()

        cap.release.assert_called_once()

    def test_sets_buffer_size_to_one(self):
        cap = _fake_cap([(False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                fe.FrameReader("cam1", "rtsp://cam/stream").run()

        cap.set.assert_called_once_with(fe.cv2.CAP_PROP_BUFFERSIZE, 1)

    def test_latest_returns_newest_frame_with_grab_timestamp(self):
        frame1, frame2 = MagicMock(), MagicMock()
        cap = _fake_cap([(True, frame1), (True, frame2), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]
        reader = fe.FrameReader("cam1", "rtsp://cam/stream")

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                reader.run()

        frame, ts = reader.latest()
        assert frame is frame2                     # only the newest survives
        assert ts                                  # stamped at grab time

    def test_latest_consumes_the_frame(self):
        frame1 = MagicMock()
        cap = _fake_cap([(True, frame1), (False, None)])
        fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]
        reader = fe.FrameReader("cam1", "rtsp://cam/stream")

        with patch.object(fe.time, "sleep"):
            with pytest.raises(_StopLoop):
                reader.run()

        first, _  = reader.latest()
        second, _ = reader.latest()
        assert first is frame1
        assert second is None                      # no double-publish of a stalled stream


# ── publish_loop: payload + sequencing ────────────────────────────────────────

class TestPublishLoop:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def _run(self, frames, time_side_effect=None):
        """Run publish_loop with stubbed reader; escape via sleep → _StopLoop."""
        reader = _fake_reader(frames)
        sleep_patch = patch.object(fe.time, "sleep", side_effect=_StopLoop)
        time_patch = (patch.object(fe.time, "time", side_effect=time_side_effect)
                      if time_side_effect else patch.object(fe.time, "time", return_value=0.0))
        with patch.object(fe, "_is_extraction_enabled", return_value=True), \
             sleep_patch, time_patch:
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)
        return reader

    def test_publishes_on_available_frame(self):
        self._run([(_fake_frame(), "2026-01-01T00:00:00+00:00")])
        fe.publisher.publish.assert_called_once()

    def test_payload_contains_required_keys(self):
        self._run([(_fake_frame(), "2026-01-01T00:00:00+00:00")])
        raw = fe.publisher.publish.call_args[0][1]
        payload = json.loads(raw.decode())
        assert payload["camera_id"] == "cam1"
        assert payload["seq"] == 0
        assert "data_b64" in payload

    def test_ts_is_grab_time_not_publish_time(self):
        grab_ts = "2026-01-01T00:00:00+00:00"
        self._run([(_fake_frame(), grab_ts)])
        payload = json.loads(fe.publisher.publish.call_args[0][1].decode())
        assert payload["ts"] == grab_ts

    def test_skips_publish_when_no_frame_available(self):
        self._run([(None, "")])
        fe.publisher.publish.assert_not_called()

    def test_skips_publish_when_extraction_disabled(self):
        reader = _fake_reader([(_fake_frame(), "ts")])
        with patch.object(fe, "_is_extraction_enabled", return_value=False), \
             patch.object(fe.time, "sleep", side_effect=_StopLoop), \
             patch.object(fe.time, "time", return_value=0.0):
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)
        reader.latest.assert_not_called()
        fe.publisher.publish.assert_not_called()

    def test_seq_increments_across_frames(self):
        frame = _fake_frame()
        reader = _fake_reader([(frame, "t0"), (frame, "t1"), (frame, "t2")])
        # Two sleeps pass, the third raises to exit the loop
        with patch.object(fe, "_is_extraction_enabled", return_value=True), \
             patch.object(fe.time, "sleep", side_effect=[None, None, _StopLoop]), \
             patch.object(fe.time, "time", return_value=0.0):
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)

        assert fe.publisher.publish.call_count == 3
        seqs = [
            json.loads(c[0][1].decode())["seq"]
            for c in fe.publisher.publish.call_args_list
        ]
        assert seqs == [0, 1, 2]


# ── publish_loop: frame timing ────────────────────────────────────────────────

class TestPublishLoopTiming:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def test_sleeps_remaining_interval_when_frame_is_fast(self):
        # CAPTURE_FPS=2 → interval=0.5s; elapsed=0.1s → sleep_for=0.4s
        reader = _fake_reader([(_fake_frame(), "ts")])
        with patch.object(fe, "_is_extraction_enabled", return_value=True), \
             patch.object(fe.time, "time", side_effect=[0.0, 0.1]), \
             patch.object(fe.time, "sleep", side_effect=_StopLoop) as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)

        assert abs(mock_sleep.call_args[0][0] - 0.4) < 0.001

    def test_no_throttle_sleep_when_frame_is_slow(self):
        # elapsed=0.6s > interval=0.5s → sleep_for is negative → no sleep;
        # the loop runs again and the second latest() raises to exit
        reader = _fake_reader([(_fake_frame(), "ts"), _StopLoop])
        with patch.object(fe, "_is_extraction_enabled", return_value=True), \
             patch.object(fe.time, "time", side_effect=[0.0, 0.6, 1.0]), \
             patch.object(fe.time, "sleep") as mock_sleep:
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)

        mock_sleep.assert_not_called()


# ── publish_loop: encoding parameters ─────────────────────────────────────────

class TestPublishLoopEncoding:
    def setup_method(self):
        fe.cv2.reset_mock()
        fe.publisher.reset_mock()

    def _run_one_frame(self):
        reader = _fake_reader([(_fake_frame(), "ts")])
        with patch.object(fe, "_is_extraction_enabled", return_value=True), \
             patch.object(fe.time, "time", return_value=0.0), \
             patch.object(fe.time, "sleep", side_effect=_StopLoop):
            with pytest.raises(_StopLoop):
                fe.publish_loop(reader)

    def test_resize_uses_configured_dimensions(self):
        self._run_one_frame()
        fe.cv2.resize.assert_called_once_with(ANY, (fe.FRAME_WIDTH, fe.FRAME_HEIGHT))

    def test_imencode_uses_jpeg_format_and_configured_quality(self):
        self._run_one_frame()
        encode_args = fe.cv2.imencode.call_args[0]
        assert encode_args[0] == ".jpg"
        assert fe.JPEG_QUALITY in encode_args[2]


# ── _is_extraction_enabled: Firestore toggle ──────────────────────────────────

class TestIsExtractionEnabled:
    def setup_method(self):
        # Reset the cache so each test performs a real check
        fe._extraction_checked_at = 0.0
        fe._extraction_enabled = True

    def test_returns_value_from_firestore_document(self):
        snap = MagicMock()
        snap.exists = True
        snap.to_dict.return_value = {"enabled": False}
        fs = MagicMock()
        fs.collection.return_value.document.return_value.get.return_value = snap
        with patch.object(fe, "_firestore", return_value=fs):
            assert fe._is_extraction_enabled() is False

    def test_defaults_to_enabled_when_document_missing(self):
        snap = MagicMock()
        snap.exists = False
        fs = MagicMock()
        fs.collection.return_value.document.return_value.get.return_value = snap
        with patch.object(fe, "_firestore", return_value=fs):
            assert fe._is_extraction_enabled() is True

    def test_keeps_current_state_on_firestore_error(self):
        fe._extraction_enabled = False
        with patch.object(fe, "_firestore", side_effect=Exception("offline")):
            assert fe._is_extraction_enabled() is False

    def test_uses_cached_value_within_five_seconds(self):
        fe._extraction_checked_at = fe.time.time()
        fe._extraction_enabled = False
        with patch.object(fe, "_firestore") as mock_fs:
            assert fe._is_extraction_enabled() is False
        mock_fs.assert_not_called()


# ── main(): URL/ID parsing ────────────────────────────────────────────────────

class TestMainParsing:
    def _run_main(self, rtsp_urls: str, camera_ids: str = "cam1,cam2,cam3"):
        """Run main() with patched env globals, threads, and HTTP server."""
        with patch.object(fe, "RTSP_URLS_ENV", rtsp_urls), \
             patch.object(fe, "CAMERA_IDS_ENV", camera_ids), \
             patch.object(fe, "FrameReader") as mock_reader, \
             patch.object(fe.threading, "Thread") as mock_thread, \
             patch.object(fe, "HTTPServer") as mock_server_class:
            mock_server_class.return_value.serve_forever.side_effect = KeyboardInterrupt
            with pytest.raises(KeyboardInterrupt):
                fe.main()
        return mock_reader, mock_thread

    def test_raises_when_rtsp_urls_is_empty(self):
        with patch.object(fe, "RTSP_URLS_ENV", ""):
            with pytest.raises(RuntimeError, match="RTSP_URLS"):
                fe.main()

    def test_starts_one_reader_and_publisher_per_camera(self):
        mock_reader, mock_thread = self._run_main("rtsp://a,rtsp://b,rtsp://c")
        assert mock_reader.call_count == 3
        assert mock_thread.call_count == 3

    def test_pads_camera_ids_when_fewer_than_urls(self):
        mock_reader, _ = self._run_main(
            rtsp_urls="rtsp://a,rtsp://b,rtsp://c",
            camera_ids="cam1",
        )
        used_ids = [c[0][0] for c in mock_reader.call_args_list]
        assert used_ids == ["cam1", "cam2", "cam3"]

    def test_uses_provided_ids_when_count_matches(self):
        mock_reader, _ = self._run_main(
            rtsp_urls="rtsp://a,rtsp://b",
            camera_ids="front,back",
        )
        used_ids = [c[0][0] for c in mock_reader.call_args_list]
        assert used_ids == ["front", "back"]
