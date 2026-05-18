---
title: 06 — Dashboard
tags: [dashboard, firebase, webrtc, fcm, frontend, project/printermonitor]
aliases: [frontend, web app, browser UI]
type: frontend
---

# Dashboard (Firebase Hosting + WebRTC + Firestore + FCM)

A single-page vanilla-JS dashboard. **No build step, no framework.** Two files in `dashboard/`:
- `index.html` — UI + WebRTC + Firestore listener + alert UX (508 lines)
- `firebase-messaging-sw.js` — service worker for FCM background push (26 lines)

Served from Firebase Hosting at https://printermonitor-488112.web.app.

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

`firebase deploy --only hosting` pushes the contents of `dashboard/` to the CDN. There's no CI job for this — manual deploys only.

## Layout

CSS-grid 2×2:
- Top-left: cam1 tile (`<video>` + overlay `<canvas>` over a black stage)
- Top-right: cam2 tile
- Bottom-left: cam3 tile
- Bottom-right: dashboard tile (event log + clear button)

Header: title pill, MediaMTX host pill, Firestore status pill, "Reconnect all" button.

Responsive: < 1100 px → single column.

## WebRTC playback (WHEP)

`index.html:265-330`. For each cam:
1. `new RTCPeerConnection()` and `pc.addTransceiver("video", { direction: "recvonly" })`.
2. `pc.createOffer()` → `pc.setLocalDescription(offer)`.
3. `POST https://<HOST>/<cam>/whep` with `Content-Type: application/sdp` and the offer SDP as body.
4. The response is the SDP answer → `pc.setRemoteDescription({type:"answer", sdp: …})`.
5. `pc.ontrack` → wire the incoming stream into the `<video>` element.

`HOST` comes from `window.MEDIAMTX_HOST` with a hard-coded fallback (`hire-measures-ink-buf.trycloudflare.com`). The fallback is for the ephemeral Cloudflare quick-tunnel mode. When a Named Tunnel is configured, the fallback must be updated (or `MEDIAMTX_HOST` set somewhere earlier in the page).

Reconnect strategy: connection state goes `failed`/`disconnected` → status dot turns red, user clicks "Reconnect all". No automatic retry.

## Bounding box overlay

Each tile has a `<canvas>` absolutely positioned over its `<video>`. `requestAnimationFrame` loop calls `drawBoxes(cam, boxes)`:
- If video has loaded, set `canvas.width = video.videoWidth`, etc. — so the bbox coordinates are in the natural resolution of the stream.
- Clear, then for each box: stroke rectangle in a per-class color (`palette` object at `index.html:346-351`), then a black 65% alpha label background with the class name and confidence.
- Boxes are in normalized `[0,1]` units → multiplied by canvas dimensions.

**TTL**: `latestBoxes[cam]` is cleared after 10 s of no new detections (`DETECTION_TTL_MS = 10_000`). Without TTL, the last boxes would persist forever. TTL matches the Pi's `CAPTURE_FPS=0.1` — see [[03-pi-edge#Frame extractor]].

## Firestore realtime listener

`index.html:438-503`. Query: `collection("alerts") orderBy("created_at" desc) limit(50)`.

`onSnapshot` callback:
1. Filter to `change.type === "added"` only (ignores updates).
2. Dedupe via `seen` Set (Firestore can replay docs across reconnections).
3. Skip docs older than `pageLoadTime` — fresh sessions don't show historical events.
4. Map `detections[]` into the canvas overlay state.
5. For each detection, append a row to the event log with `addEventLine()`.
6. If any detection's label is in the HIGH_SEV set `{spagetti, not_sticking, layer_shift, warping}`:
   - Flash a red border on the body (`flashGlobalAlert(3000)`)
   - Show a 5 s toast (`showToast(...)`)

The HIGH_SEV set is duplicated client-side at `index.html:469-471` — keep in sync with `terraform_v2/services/alert-manager/main.py:23`.

## Coordinate fallback

`index.html:474-481` — if `det.x/y/w/h` are missing but `det.bbox` is present, the dashboard divides by a hard-coded `640×480` to normalize. This is a **historical compatibility shim** for an older detection format and assumes 640×480 source frames; current frames are 1280×720 so this branch produces incorrect overlays.

Modern judge output always includes normalized fields (`judge/main.py:104-108`), so this branch never fires in production. Safe to delete if no historical Firestore docs need rendering.

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

- WebRTC reconnect is fully manual — `pc.connectionState === "failed"` doesn't auto-reattempt
- HIGH_SEV duplicated in JS + Python (see [[04-ml-model#Severity tiers]])
- FCM half-wired (SW present, SDK not initialized in page)
- Hard-coded fallback Cloudflare hostname will rot
- 640×480 bbox fallback branch is dead code

See [[01-architecture#Component diagram]] for where this fits in the broader flow.
