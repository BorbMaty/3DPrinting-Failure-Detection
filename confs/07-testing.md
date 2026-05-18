---
title: 07 — Testing
tags: [testing, pytest, ci, mocking, coverage, project/printermonitor]
aliases: [tests, pytest, unit tests]
type: testing
---

# Testing

Pure unit tests over the four Python service modules. No integration tests, no real GCP, no real PyTorch — heavy deps are stubbed at `sys.modules` level *before* the target module is imported.

## Layout

```
tests/
├── test_judge.py            (207 lines)
├── test_dispatcher.py       (152 lines)
├── test_alert_manager.py    (346 lines)
└── test_frame_extractor.py  (242 lines)
```

Plus `requirements-test.txt`:
```
pytest>=8.0
pytest-cov>=5.0
tzdata
pre-commit
```

## Run commands

```bash
# Install test deps
pip install -r requirements-test.txt

# Full suite + coverage + 90% gate
pytest tests/ -v --tb=short --cov=terraform_v2/services --cov-report=term-missing --cov-fail-under=90

# One file
pytest tests/test_judge.py -v

# One test
pytest tests/test_alert_manager.py::TestHandleDetection::test_sends_email_for_high_severity_detection -v
```

CI runs the same command (minus `-v`, plus `--cov-report=xml` artifact). See [[08-ci-cd#python-tests.yml]].

## The mocking pattern (key to understanding every test file)

Every test file follows the same setup before importing the module under test:

```python
# 1. Patch env vars the module reads at import-time
os.environ.setdefault("GCP_PROJECT", "test-project")
os.environ.setdefault("DETECTIONS_TOPIC", "test-detections")
...

# 2. Stub heavy dependencies in sys.modules
for _mod in ["cv2", "numpy", "ultralytics", "google", "google.cloud", "google.cloud.pubsub_v1"]:
    sys.modules.setdefault(_mod, MagicMock())

# 3. Stub functions_framework so @cloud_event becomes a no-op decorator
_mock_ff = MagicMock()
_mock_ff.cloud_event = lambda f: f
sys.modules.setdefault("functions_framework", _mock_ff)

# 4. importlib-load the real module after stubs are in place
_spec = importlib.util.spec_from_file_location("judge_main", Path(__file__).parents[1] / "terraform_v2/services/judge/main.py")
judge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge)
```

This means:
- The target module's top-level code runs (reading env vars, building `topic_path`, calling `YOLO(...)` — which becomes a MagicMock call).
- Inside tests, `judge.model`, `judge.publisher`, `judge.cv2` are all MagicMocks — fully controllable via `judge.cv2.imdecode.return_value = …` etc.
- No real network, no real ML, no real GCP credentials needed.

### Identity decorator for Cloud Functions

`@functions_framework.cloud_event` would wrap `dispatch_frame`/`handle_detection` into a runtime adapter that expects a real CloudEvent SDK type. The test stub replaces it with `lambda f: f` so the wrapped function remains a plain callable:

```python
_mock_ff.cloud_event = lambda f: f
sys.modules.setdefault("functions_framework", _mock_ff)
```

This is critical — without it, you can't call `dispatch_frame(event)` directly in tests.

### Sentinel-exception to escape infinite loops

`test_frame_extractor.py` injects a sentinel exception (`_StopLoop`) into `cv2.VideoCapture.side_effect` so the infinite `while True:` loop in `capture_loop()` raises after the test has its assertions in flight:

```python
fe.cv2.VideoCapture.side_effect = [cap, _StopLoop]
with pytest.raises(_StopLoop):
    fe.capture_loop("rtsp://test", "cam1")
```

The same pattern is used in `test_frame_extractor.py::TestMainParsing` to escape `HTTPServer.serve_forever()` via `KeyboardInterrupt`.

## What each file actually covers

### `test_judge.py`

| Test class | Surface |
|---|---|
| `TestNowIso` | `now_iso()` returns ISO-8601 UTC strings, monotonic |
| `TestSafeB64Decode` | `_safe_b64decode()` padding repair, whitespace strip, empty/None raises |
| `TestHandlerGet` | `/healthz`, `/health`, `/` all return 200; unknown → 404 |
| `TestHandlerPost` | Vertex AI envelope works; Pub/Sub envelope works; `/wrong` → 404; corrupt image → 500; publish() called |

`_make_post_handler` / `_make_get_handler` build a `Handler` instance bypassing the normal HTTP server lifecycle (`Handler.__new__` to skip `__init__`).

### `test_dispatcher.py`

| Test class | Surface |
|---|---|
| `TestGetAuthHeader` | Token refresh skipped when valid, called when invalid; Bearer header format |
| `TestDispatchFrame` | Skips when no `data_b64`; builds correct Vertex AI `instances` body; re-raises HTTP errors so Pub/Sub retries; malformed event doesn't crash; `image_b64` accepted as alias |

`requests.exceptions.RequestException` is patched to a real exception class so the `except` clause in the SUT works.

### `test_alert_manager.py`

| Test class | Surface |
|---|---|
| `TestHumanTime` | UTC→Europe/Bucharest conversion (`+02:00`), date format, malformed input returns original |
| `TestIsOnCooldown` | doc-not-exists → False, recent → True, expired → False, boundary 59s (still in cooldown) |
| `TestSetCooldown` | writes `last_sent` ISO timestamp under correct key |
| `TestSendEmail` | SMTP_SSL connection, login, subject in payload, doesn't raise on connection error |
| `TestHandleDetection` | confidence filter (`>=`), Firestore write happens, low-sev no email, all 4 HIGH_SEV labels trigger email, cooldown suppresses, threshold boundary (0.35 passes, 0.3499 doesn't), empty detections no-op, malformed event no-crash |
| `TestHandleBudgetAlert` | budget Firestore write, email when not cooldown, no email when cooldown, malformed event no-crash |

> **Boundary test mismatch alert**: `TestHandleDetection.test_detection_at_exact_threshold_passes` uses confidence=0.35 and `os.environ.setdefault("CONF_THRESHOLD", "0.35")`. The deployed alert-manager actually defaults to `0.20` (from `variables.tf:23`). The test setup explicitly overrides to 0.35, which means **the test is testing the code, not the deployed config**. Both should pass; just be aware.

### `test_frame_extractor.py`

| Test class | Surface |
|---|---|
| `TestCaptureLoop` | retry on stream-open failure (5 s sleep), release on read-fail, publish on success, payload schema (`camera_id`/`seq`/`ts`/`data_b64`), seq increments across frames |
| `TestMainParsing` | empty `RTSP_URLS` raises; one thread per camera; ID padding when fewer IDs than URLs; explicit IDs preserved |
| `TestCaptureLoopTiming` | sleeps `interval - elapsed` when frame is fast; no throttle when frame is slow |
| `TestCaptureLoopEncoding` | resize uses `FRAME_WIDTH × FRAME_HEIGHT`; encode uses `.jpg` + `JPEG_QUALITY` |

This file tests `terraform_v2/services/frame-extractor/main.py` — **not** `pi_codes/frame_extractor.py` (which is the actually-deployed version, but is uncovered by tests).

## Coverage gate

90% line coverage, enforced both in CI (`--cov-fail-under=90` in `python-tests.yml`) and locally via pre-commit (`.pre-commit-config.yaml`). The pre-commit hook runs the full pytest suite on every commit, which is slow — bypass with `--no-verify` only when you have a non-code change.

Coverage scope: `terraform_v2/services/`. The Pi codes, scripts, dashboard JS, and local_test/ are **not measured**. They're treated as out-of-scope (`pi_codes/`) or test scaffolding (`local_test/`).

## Pre-commit hooks (`.pre-commit-config.yaml`)

Three hook groups:
1. `pre-commit-terraform v1.97.0` — `terraform_fmt`, `terraform_tflint`
2. `bridgecrewio/checkov v3.2.325` — full project scan with `.checkov.yaml`
3. local `pytest` hook — full suite + 90% gate

All run on `git commit`. The pytest hook uses `language: system` and `always_run: true` so it fires on every commit regardless of which files changed.

## Things NOT tested

- **Pi-side code** (`pi_codes/`) — would require a Pi or hardware mocks for V4L2, RTSP. Skipped.
- **GStreamer pipelines** — out of scope for unit tests.
- **Dashboard JS** — no JS test harness configured.
- **Terraform** — covered by `terraform plan` + tflint + checkov, not unit tests.
- **End-to-end integration** — there's no e2e test harness. The closest analog is the [[10-local-test]] harness which is for development, not CI.
- **YOLO inference correctness** — model is mocked; tests can't catch a bad weights file.
- **`scripts/annotate.py`** — no tests; manual tool.

## Strategy notes for future tests

- Adding a new service: copy the bootstrap pattern from `test_dispatcher.py` (smallest of the four).
- Module-level state (e.g. `_streaks` dict in judge): tests in this codebase don't reset it between cases. New streak-filter tests should call `judge._streaks.clear()` in `setup_method`.
- When adding tests for `firebase_admin.messaging` calls, stub `firebase_admin` in `sys.modules` exactly like google.cloud is stubbed.
