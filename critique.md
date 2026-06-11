# Project Critique — 3DPrinting-Failure-Detection

*Reviewed 2026-06-11. Scope: full repo — services, Terraform, dashboard, Pi scripts, tests, CI.*

## Verdict

The pipeline is genuinely well-engineered for a thesis project: dead-letter queues, a
staleness filter, monitoring alert policies, a documented backlog-purge procedure, and a
test suite that mocks heavy dependencies cleanly. It will run "smoothly" in the sense
that frames flow end-to-end and alerts arrive. But there are several **hidden
discrepancies between what the docs/code claim and what is actually deployed**, the
**live-feed overlay is not synchronized with detections** (and structurally cannot be),
and a few latent bugs will surface under specific conditions.

---

## 1. Is detection synchronized with the live feed? — No.

There are two displays, and only one of them is honest:

- **Page 2 (Inference log) is correctly synchronized.** Bounding boxes are drawn on the
  exact GCS-uploaded frame the judge ran inference on. This is the trustworthy view.
- **Page 1 (Live streams) is not synchronized, and cannot be.** The WebRTC feed is
  sub-second live. The boxes drawn on top of it come from the `alerts` Firestore
  collection and arrive **tens of seconds after the moment they describe**:

  | Stage | Delay |
  |---|---|
  | Capture interval (Pi `frame_extractor.py`, `CAPTURE_FPS=0.1`) | up to 10 s until a defect is even photographed |
  | Streak confirmation (`STREAK_REQUIRED` × consecutive frames) | 2–3 more frames = **20–30 s** at 0.1 FPS |
  | Pub/Sub → Eventarc → dispatcher (cold start) → Vertex → Pub/Sub → alert-manager → Firestore → onSnapshot | ~1–5 s, more on cold starts |

  The dashboard makes **no attempt to match detection timestamps to video frames** — it
  draws the latest boxes over whatever the camera shows *now* and keeps them for
  `DETECTION_TTL_MS = 10 s`. By the time a box appears, the print head has moved; the
  box may outline empty space. Note also that the TTL (10 s) equals the capture
  interval (10 s), so boxes flicker on/off between confirmations.

  This is an acceptable design trade-off (true frame-locked overlay would require
  running inference on the WebRTC stream itself), but the thesis should state it
  explicitly: **page 1 boxes are "recent detections", not "what you are looking at".**

### Worse: the synchronization that *does* exist is built on a lie

`pi_codes/frame_extractor.py:88` stamps `ts = now_iso()` **after** `cap.read()`
returns. OpenCV/FFmpeg buffers RTSP frames internally; at 0.1 FPS the code reads one
buffered frame every 10 s while the camera produces ~30/s. `cap.read()` returns the
*next buffered* frame, not the *latest* frame, so capture drifts progressively behind
real time — and the freshly-stamped `ts` hides that drift. Consequences:

- The dispatcher's staleness filter (`MAX_FRAME_AGE_S=5`) is defeated: a frame that is
  minutes old carries a timestamp seconds old and sails through.
- The "Capture: OFF" pause makes it worse — the buffer keeps filling for the entire
  pause, then resumes serving ancient frames.

**Fix:** call `cap.grab()` in a tight loop (or a reader thread that keeps only the last
frame) and `retrieve()` on the capture tick, or set `CAP_PROP_BUFFERSIZE=1`. Stamp `ts`
from the moment the frame is grabbed.

---

## 2. FCM is intentionally broken

Push notifications were deliberately removed in commit `06b7a01` ("removed push
notifications and added email notifications instead"), immediately after `6c09a63`
added them. Email is the intended alert channel. **This is a known, accepted state, not
a regression** — but the removal was incomplete, and the corpse is scattered across the
repo in a way that will mislead any reader:

- `terraform_v2/services/budget-notifier/main.py` still contains the full FCM
  implementation (`send_fcm_defect`, `send_fcm_budget`) — but it is **dead code that is
  never deployed**: `main.tf:375-380` builds *both* Cloud Functions from
  `services/alert-manager/`, which has no FCM code at all.
- `tests/test_budget_notifier.py` tests the dead file (41 references to FCM/messaging),
  so the 90 % coverage gate is partly measuring code that never runs in production.
- `dashboard/firebase-messaging-sw.js` is an orphaned service worker: `index.html`
  never imports `firebase-messaging`, never requests notification permission, never
  calls `getToken()`, and nothing ever subscribes a client to the `defect-alerts` /
  `budget-alerts` topics. `firebase.json` still ships special headers for it.
- Terraform still enables `fcm.googleapis.com`, `fcmregistrations.googleapis.com`,
  `firebaseinstallations.googleapis.com` (`main.tf:59-62`) and grants
  `roles/firebase.admin` to the alert-manager SA (`main.tf:237-241`) "for
  `messaging.send()`" — which no deployed code calls.
- `services/alert-manager/requirements.txt` still pins `firebase-admin==6.5.0`, unused.
- `CLAUDE.md` architecture diagram and service table still claim both functions do
  "Firestore + Gmail + **FCM**" — the documentation contradicts the deployed reality.

**Recommendation:** either delete the dead path entirely (file, test, SA role, APIs,
service worker, requirements line, doc claims) or move the FCM code behind a feature
flag with a comment saying it's parked. Right now a grader/reviewer reading the repo
will conclude FCM works.

---

## 3. Other hidden things

### 3.1 Two divergent frame extractors

`terraform_v2/services/frame-extractor/main.py` (2 FPS, JPEG q70, env-driven camera
list, HTTP health server, **no capture toggle**) and `pi_codes/frame_extractor.py`
(0.1 FPS, JPEG q100, hardcoded cameras, **with** the Firestore capture toggle) are two
different programs. CLAUDE.md's table says frame-extractor is a "Cloud Run container",
while `main.tf:208-209` says it runs on the Pi. Only the Pi variant honors the
dashboard's Capture ON/OFF button — if anyone ever deploys the `terraform_v2` variant,
the dashboard kill switch silently does nothing. Pick one implementation.

### 3.2 The dashboard kill switch is world-writable

`firestore.rules` line 12-14: `match /system_state/{doc} { allow read, write: if true; }`.
The dashboard is public and unauthenticated, so **anyone on the internet can disable
frame capture** (or write arbitrary documents into `system_state`). For a failure
*detection* system, an anonymous remote off-switch is the single worst security issue
here. `alerts` and `inferences` are also world-readable, and the frames bucket is
public (`allUsers` objectViewer) — accepted via checkov skips, but the combination
means an outsider can watch your prints and turn off monitoring.

### 3.3 Budget alerts never reach the dashboard

The dashboard has budget CSS (`body.budget-alert`, `.eventLine.budget`), a budget toast
type, and the docs promise "Budget Notifier CF → Firestore + FCM → dashboard". In
reality there is **no `onSnapshot` listener on `budget_alerts`** in `index.html`, and
`firestore.rules` has no rule for that collection anyway (default deny). Budget alerts
reach email only; the dashboard budget UI is dead code.

### 3.4 Config contradictions

- CLAUDE.md's deploy command sets `STREAK_REQUIRED=3`; `judge/main.py:18` defaults to 2.
  Whichever was used at last deploy decides alert latency — undocumented.
- CLAUDE.md says CONF_THRESHOLD 0.35; alert-manager's code default is 0.20 (Terraform
  overrides it to 0.35, so the code default is misleading).
- The monitoring comment in `main.tf:762` assumes 0.1 FPS while the cloud extractor
  defaults to 2 FPS — the backlog threshold (20 messages) is tuned for one of them only.

### 3.5 Fragile hardcoded Eventarc subscription IDs

`eventarc-europe-west1-dispatcher-635712-sub-152` and
`...alert-manager-030900-sub-669` are auto-generated names baked into `main.tf`
(DLQ wiring, monitoring filters), CLAUDE.md, and ops runbooks. Recreating either
trigger regenerates the suffix and silently breaks the DLQ policy, both backlog alert
conditions, and the documented purge command. At minimum, derive them via a data
source or output a post-apply check.

### 3.6 Judge state is single-replica-only, by accident

`_streaks` in `judge/main.py:38` is an in-process dict. With min=max=1 replica it
works; scaling to 2 replicas silently halves every streak (alternating requests), and
every redeploy/restart resets counters mid-confirmation. This constraint is real but
written nowhere.

### 3.7 No ordering guarantees under the streak filter

Pub/Sub publishes without ordering keys and the dispatcher runs up to 10 concurrent
instances, so frames can reach the judge **out of capture order**. "N consecutive
frames" actually means "N consecutive *arrivals*". At 0.1 FPS reordering is unlikely;
at the cloud extractor's 2 FPS it is not. `seq` is carried end-to-end but never used to
enforce order.

### 3.8 Alerting logic edge cases

- **Cooldown is set even when the email fails.** `send_email()` swallows all exceptions
  (`alert-manager/main.py:71-72`) and `set_cooldown()` runs unconditionally after —
  one SMTP hiccup buys 5 minutes of silence on a genuine failure.
- **Single global cooldown across all cameras**: a low-grade warping alert on cam1
  suppresses a spaghetti catastrophe on cam3 for 300 s. Documented behavior, debatable
  choice.
- **Cooldown check is racy**: read-then-write with no transaction and
  `max_instance_count=10` → burst of detections can send duplicate emails.

### 3.9 Unbounded growth

- `inferences` gets a Firestore write **per frame per camera** with no TTL policy and
  no scheduled cleanup (only the manual `flush_firestore.py`). At the cloud extractor's
  2 FPS × 3 cams that's ~518 k writes/day — both a cost and quota problem.
- Dashboard `seenAlerts`/`seenInf` sets and the event-log DOM grow forever in long-lived
  tabs (kiosk use). Minor, but it's a dashboard meant to be left open.

### 3.10 Live-overlay rendering bugs (page 1)

- The bbox fallback in `index.html:1041-1044` normalizes by **640×480**, but frames are
  1280×720 — wrong whenever `x/y/w/h` are absent. (The judge always sends normalized
  fields today, so this is latent, but it will misplace every box if that ever changes.)
- The color palette inverts severity: high-sev `spagetti` renders **green** and
  `not_sticking` orange, while low-sev `stringing` renders **red**. Page 2 colors by
  `HIGH_SEV` correctly; page 1 contradicts it.
- Palette key `foreign_object_on_print_area` doesn't match the class name
  (`foreign_object`) used everywhere else — that class always falls back to blue.

### 3.11 Repo hygiene

`coomm` (a pasted terminal/Claude transcript), `texput.log`, `oldReadMe.MD`, and the
`confs/.obsidian` plugin tree are committed. The Gmail app password travels as a plain
Terraform variable and therefore sits in the GCS-backed Terraform state — Secret
Manager would be the correct home.

---

## 4. What is genuinely good

Credit where due, because a critique should be calibrated:

- Dead-letter queues with 7-day pull subscriptions, plus the staleness drop in the
  dispatcher, show real thought about backlog failure modes (the documented
  `subscriptions seek` runbook too).
- 404/503 handling in the dispatcher (drop instead of retry-storm when the endpoint is
  undeployed) is exactly right for a system where the T4 is intentionally torn down.
- Monitoring policies on 5xx and backlog, with email notification, close the loop.
- The test strategy (sys.modules-level mocking, identity-wrapped decorators) is clean
  and runs with zero credentials; CI gates on coverage, fmt, tflint, checkov.
- The dashboard's WHEP reconnect with exponential backoff and the per-camera status
  dots are solid; page 2's frame-accurate bbox rendering is the right way to present
  inference results.

## 5. Priority fixes

1. **Fix the RTSP buffer/timestamp lie** in `pi_codes/frame_extractor.py` (grab-flush +
   honest `ts`) — it undermines the staleness filter and any latency claims in the thesis.
2. **Lock down `system_state`** in `firestore.rules` (auth, or at least a shared-secret
   field checked by the Pi).
3. **Delete or clearly quarantine the dead FCM path** (intentionally broken — see §2)
   so docs, tests, IAM, and reality agree.
4. **Document the page-1 overlay semantics** (recent detections, not frame-synced) and
   fix the TTL/flicker + severity-color inversion.
5. Reconcile the two frame extractors and the STREAK/FPS/threshold doc mismatches.
