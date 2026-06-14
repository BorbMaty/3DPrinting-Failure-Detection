---
title: 06 — Dashboard
tags: [dashboard, firebase, webrtc, fcm, frontend, project/printermonitor]
aliases: [frontend, web app, browser UI]
type: frontend
---

# Dashboard (Firebase Hosting + WebRTC + Firestore + FCM)

A **two-page** vanilla-JS dashboard. **No build step, no framework.** Two files in `dashboard/`:
- `index.html` — UI + WebRTC + Firestore listeners + alert UX + capture toggle (**~1145 lines**)
- `firebase-messaging-sw.js` — service worker for FCM background push (26 lines)

Served from Firebase Hosting at https://printermonitor-488112.web.app.

## Two pages (tab nav)

A top `<nav class="tab-nav">` switches between two `.page` divs (`index.html:518-521, 682-688`):

| Tab | Page id | What it shows |
|---|---|---|
| **Live streams** | `page-live` | The 2×2 WebRTC grid (cam1/2/3 + event log) — the original dashboard |
| **Inference log** | `page-inference` | Per-camera frame thumbnails + a live table of the last 100 `inferences` docs |

Switching just toggles the `.active` class; both pages keep their Firestore subscriptions live in the background.

## Firebase config

Same config in both files (intentional — service worker can't share globals):
```js
{
  apiKey:            "AIzaSyCjp-IZzt2CfXEwBxnn2icv8LxvrsJmieQ",
  authDomain:        "printermonitor-488112.firebaseapp.com",
  projectId:         "printermonitor-488112",
  storageBucket:     "printermonitor-488112.firebasestorage.app",
  messagingSenderId: "895714392909",
  appId:             "1:895714392909:web:553314befa3149f6ad1504",
  measurementId:     "G-WNNGT0SQT1",
}
```

This is **client-side public config**, not a secret. Firebase API keys identify the project, they don't authorize anything by themselves.

## Hosting config (`firebase.json`)

```json
{
  "hosting": {
    "public": "dashboard",
    "headers": [
      { "source": "/firebase-messaging-sw.js",
        "headers": [
          { "key": "Service-Worker-Allowed", "value": "/" },
          { "key": "Cache-Control", "value": "no-cache" }
        ]
      }
    ]
  }
}
```

- `public: dashboard` — Firebase serves the contents of that directory.
- The `Service-Worker-Allowed: /` header is required so the service worker can manage notifications across the whole origin (not just `/firebase-messaging-sw.js`).
- `no-cache` on the SW ensures hot-fix updates aren't blocked by browser cache (FCM clients are notoriously sticky).

`firebase deploy --only hosting` pushes the contents of `dashboard/` to the CDN. This is now automated via `.github/workflows/firebase-deploy.yml` — any push to `main` that touches `dashboard/**`, `firestore.rules`, or `firebase.json` triggers an automatic deploy. See [[08-ci-cd#firebase-deploy.yml]].

## Layout (page 1 — Live streams)

CSS-grid 2×2:
- Top-left: cam1 tile (`<video>` + overlay `<canvas>` over a black stage)
- Top-right: cam2 tile
- Bottom-left: cam3 tile
- Bottom-right: dashboard tile (event log + clear button)

Header: title pill, MediaMTX host pill, Firestore status pill, **"Capture: ON/OFF" toggle**, "Reconnect all" button.

Responsive: < 1100 px → single column.

## Page 2 — Inference log

`index.html:587-676` (markup), `923-1004` (JS). Two regions:
- **Frame tiles** (one per cam): the latest inference frame as an `<img>` (from `frame_url`) with an SVG bbox overlay drawn by `drawBoxesSvg(...)`, a timestamp, and a `clean`/label badge. `updateFrameTile()` only updates a tile when the incoming `ts` is newer than the one shown (`latestFrameTs[cam]`).
- **Inference table** (`addInferenceRow`): newest-first rows of `{time, camera, seq, detection pills, thumbnail}` for the last 100 `inferences` docs. Detection pills are colour-coded HIGH_SEV vs low-sev. Rows with detections get a `.has-det` red tint. A thumbnail click opens a **lightbox** (`openLightbox(frameUrl, detections)`) showing the full frame with bbox overlay.

Bounding boxes on this page are rendered as **SVG** over the static frame images (`drawBoxesSvg`), distinct from the page-1 `<canvas>` overlay on the live video.

## WebRTC playback (WHEP)

`index.html:265-330`. For each cam:
1. `new RTCPeerConnection()` and `pc.addTransceiver("video", { direction: "recvonly" })`.
2. `pc.createOffer()` → `pc.setLocalDescription(offer)`.
3. `POST https://<HOST>/<cam>/whep` with `Content-Type: application/sdp` and the offer SDP as body.
4. The response is the SDP answer → `pc.setRemoteDescription({type:"answer", sdp: …})`.
5. `pc.ontrack` → wire the incoming stream into the `<video>` element.

`HOST` comes from `window.MEDIAMTX_HOST` with a hard-coded fallback of **`cam.printermonitor.app`** (`index.html:739`). This is the permanent Cloudflare named-tunnel host ([[03-pi-edge#Cloudflare Tunnel]]), so the fallback is stable and no longer rots between sessions.

Reconnect strategy: on `iceConnectionState` `"failed"` or `"disconnected"`, the tile schedules an automatic retry with **exponential backoff** (2 s → 4 s → 8 s → … capped at 30 s). The status dot shows `"retry in Ns…"` during the wait. On successful `ontrack`/`connected`, the delay resets to 2 s. The "Reconnect all" button also resets all delays to 2 s before restarting. Per-camera state: `retryDelay[cam]` and `retryTimer[cam]` in `index.html`.

## Bounding box overlay

Each tile has a `<canvas>` absolutely positioned over its `<video>`. `requestAnimationFrame` loop calls `drawBoxes(cam, boxes)`:
- If video has loaded, set `canvas.width = video.videoWidth`, etc. — so the bbox coordinates are in the natural resolution of the stream.
- Clear, then for each box: stroke rectangle in a per-class color (`palette` object at `index.html:346-351`), then a black 65% alpha label background with the class name and confidence.
- Boxes are in normalized `[0,1]` units → multiplied by canvas dimensions.

**TTL**: `latestBoxes[cam]` is cleared after 10 s of no new detections (`DETECTION_TTL_MS = 10_000`). Without TTL, the last boxes would persist forever. TTL matches the Pi's `CAPTURE_FPS=0.1` — see [[03-pi-edge#Frame extractor]].

## Firestore realtime listeners

Both listeners are wired in the `firestoreReady` handler (`index.html:1006-1139`). There are now **two** `onSnapshot` queries plus the capture-toggle doc listener.

### `alerts` → page-1 event log

Query: `collection("alerts") orderBy("created_at" desc) limit(50)` (`index.html:1020-1064`).

`onSnapshot` callback:
1. Filter to `change.type === "added"` only (ignores updates).
2. Dedupe via `seenAlerts` Set (Firestore can replay docs across reconnections).
3. Skip docs older than `pageLoadTime` — fresh sessions don't show historical events.
4. Map `detections[]` into the page-1 canvas overlay state (`latestBoxes[cam]`).
5. For each detection, append a row to the event log with `addEventLine()`.
6. If any detection's label is in the HIGH_SEV set `{spagetti, not_sticking, layer_shift, warping}`:
   - Flash a red border on the body (`flashGlobalAlert(3000)`)
   - Show a toast (`showToast(...)`)

The HIGH_SEV set is duplicated client-side at `index.html:749` — keep in sync with `terraform_v2/services/alert-manager/main.py:22`.

### `inferences` → page-2 frame tiles + table

Query: `collection("inferences") orderBy("created_at" desc) limit(100)` (`index.html:1085-1111`).

- **First snapshot** (`infInitialized` guard): back-fills the table with the last 100 inferences (reversed so newest ends up on top) and seeds each camera's frame tile.
- **Subsequent snapshots**: for each newly-`added` doc, update the matching frame tile (if `ts` is newer) and prepend a table row, scrolling the log back to top.
- `extractInfData()` pulls `{camera_id, detections, timestamp, seq, frame_url}` off each doc. Zero-detection frames show a `clean` badge and a `—` in the detections column.

### `system_state/extraction` → capture toggle

A third `onSnapshot` (`index.html:1125-1128`) on the single `system_state/extraction` doc keeps the **"Capture: ON/OFF"** button in sync with the live `enabled` flag. Clicking the button `setDoc`s `{enabled: !current}` (`index.html:1130-1138`); the Pi extractor polls the same doc to pause/resume ([[03-pi-edge#Remote capture kill-switch]]). This is the only place the dashboard *writes* to Firestore — `firestore.rules` permits it because `system_state` is public read+write.

## Coordinate fallback

`index.html:1041-1044` (page-1 alerts listener) — if `det.x/y/w/h` are missing but `det.bbox` is present, the dashboard divides by a hard-coded `640×480` to normalize. This is a **historical compatibility shim** for an older detection format and assumes 640×480 source frames; current frames are 1280×720 so this branch produces incorrect overlays.

Modern judge output always includes normalized fields (`judge/main.py:126-129`), so this branch never fires in production. Safe to delete if no historical Firestore docs need rendering.

## Service worker (`firebase-messaging-sw.js`)

Background FCM handler. When the browser tab is **not** focused, `onBackgroundMessage` fires for any push and calls `self.registration.showNotification`:
- title: from `payload.notification.title` (server-set)
- body: from `payload.notification.body`
- `tag: "printermonitor-alert"` — replaces previous notifications (so the OS notification tray doesn't pile up identical alerts)
- `renotify: true` — re-buzzes the device on each new alert with the same tag

> **The currently-deployed `alert-manager` does not send FCM.** The previous version (preserved as dead code in `services/budget-notifier/main.py`) did. So in practice the SW is wired up but receives nothing.

## Visual state machine

| Body class | Trigger | Visual effect |
|---|---|---|
| (none) | default | normal dark theme |
| `alerting` | HIGH_SEV detection in last 2.5 s | red outline around grid, red header border |
| `budget-alert` | (unused — was for FCM budget pushes) | yellow header border |

The `alerting` class auto-clears after 2.5 s via `setTimeout`. `budget-alert` would auto-clear similarly but isn't triggered anywhere in the current code.

## Browser → service worker registration

Not visible in `index.html` — there's no `navigator.serviceWorker.register()` call. The Firebase SDK registers the SW automatically when `getMessaging(app)` is called, but **the SDK is not initialized** in the current `index.html` (only `firebase-app` and `firebase-firestore` are imported). So FCM is **off** on this build, even though the SW file exists. To re-enable, import `firebase-messaging.js` and call `getToken(messaging, {vapidKey, serviceWorkerRegistration})`.

## Known issues / smells

- HIGH_SEV duplicated in JS + Python (see [[04-ml-model#Severity tiers]])
- FCM half-wired (SW present, SDK not initialized in page)
- 640×480 bbox fallback branch (page-1 alerts listener) is dead code
- `system_state` is **publicly writable** — anyone with the URL can stop frame capture. Fine for a thesis demo; document it as a known limitation. See [[feedback]].
- `inferences` query is `limit(100)` with no time-window filter; on a long-lived collection the first paint reads the 100 most recent docs regardless of age.

See [[01-architecture#Component diagram]] for where this fits in the broader flow.
