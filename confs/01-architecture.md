---
title: 01 — Architecture
tags: [architecture, dataflow, project/printermonitor]
aliases: [data flow, system overview]
type: architecture
---

# Architecture & Data Flow

End-to-end view of how a frame becomes an alert. See [[02-cloud-services]] for per-service detail and [[05-infrastructure]] for the IAM and Pub/Sub plumbing.

## Component diagram

```
┌──────────────────── Raspberry Pi ────────────────────┐
│                                                       │
│  USB cam1 ─┐                                          │
│  USB cam2 ─┼─► GStreamer (start_3cams_rtsp.sh) ─► MediaMTX (RTSP :8554, WebRTC :8889)
│  USB cam3 ─┘                                  │       │
│                                               │       │
│  frame_extractor.py  ◄────RTSP localhost──────┘       │
│       │ (cv2.VideoCapture, 0.1 fps,                   │
│       │  jpeg q=70, ≤1280×720)                        │
│       ▼                                               │
│  Pub/Sub: frames-in (gRPC via sa-frame-extractor key) │
│       │                                               │
│       └──── WebRTC WHEP ◄── Cloudflare Tunnel ◄────── ┘
│                                                  │
└──────────────────────────────────────────────────│────┘
                                                   │
┌──────────────── Google Cloud Platform ───────────│────┐
│                                                   │   │
│  Pub/Sub: frames-in  ─────────────────────────────┘   │
│       │                                               │
│       ▼ Eventarc                                      │
│  Cloud Function (Gen2): dispatcher (Python 3.12)      │
│       │ google-auth → Bearer token                    │
│       │ HTTPS POST {"instances": [...]}               │
│       ▼                                               │
│  Vertex AI Endpoint (id 6900414029643120640)          │
│   └─► Judge container (n1-standard-4 + NVIDIA T4)     │
│       PyTorch 2.2 + CUDA 12.1, YOLOv8x best.pt        │
│       Streak filter (≥2 consecutive frames)           │
│       │ publishes (via Pub/Sub publisher client)      │
│       ▼                                               │
│  Pub/Sub: detections-out                              │
│       │                                               │
│       ├─► Eventarc → CF: alert-manager                │
│       │     ├─ Firestore: alerts (add doc)            │
│       │     └─ Gmail SMTP (cooldown 300s, HIGH_SEV)   │
│       │                                               │
│       └─► (dashboard listens via onSnapshot)          │
│                                                       │
│  Pub/Sub: budget-notifications (from billing budget)  │
│       └─► Eventarc → CF: budget-notifier              │
│              (entry handle_budget_alert, same source) │
│              ├─ Firestore: budget_alerts              │
│              └─ Gmail SMTP                            │
│                                                       │
│  Firestore (NATIVE, eur3): alerts, alert_cooldowns,   │
│                            budget_alerts              │
│                                                       │
└────────────────────│──────────────────────────────────┘
                     │
┌──────── Browser (Firebase Hosting) ─────────│─────────┐
│                                              │        │
│  index.html (vanilla JS + Firebase SDK)      │        │
│   ├─ WebRTC RTCPeerConnection (WHEP POST) ◄──┘        │
│   ├─ Firestore onSnapshot('alerts')                   │
│   ├─ Canvas overlay (bbox drawn from normalized x/y/w/h)
│   └─ Service Worker: firebase-messaging-sw.js (FCM)   │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## Message contracts

### `frames-in` payload (published by Pi frame extractor)
JSON, UTF-8, base64-wrapped by Pub/Sub:
```json
{
  "camera_id": "cam1",
  "seq": 42,
  "ts": "2026-05-18T12:34:56.789012+00:00",
  "data_b64": "<base64 JPEG>"
}
```

### Vertex AI request (dispatcher → judge)
```json
{
  "instances": [{
    "data_b64": "...",
    "camera_id": "cam1",
    "seq": 42,
    "ts": "2026-05-18T12:34:56+00:00"
  }]
}
```
Judge also accepts the raw Pub/Sub push envelope `{"message":{"data": ...}}` — see `terraform_v2/services/judge/main.py:70-84`.

### `detections-out` payload (published by judge)
```json
{
  "ts": "2026-05-18T12:34:56+00:00",
  "camera_id": "cam1",
  "seq": 42,
  "detections": [{
    "label": "spagetti",
    "confidence": 0.8721,
    "bbox": [120, 80, 540, 460],
    "x": 0.1875, "y": 0.1667, "w": 0.6563, "h": 0.7917
  }]
}
```
Both pixel `bbox` and normalized `x,y,w,h` are emitted so the dashboard can overlay without knowing the original frame size.

### Vertex AI response (judge → dispatcher)
```json
{"predictions": [<same as detections-out payload>]}
```

### `budget-notifications` payload (GCP-emitted)
```json
{
  "costAmount": 4.50,
  "budgetAmount": 5.00,
  "budgetDisplayName": "monthly-cap"
}
```

## Hop summary

| # | Hop | Protocol | Auth |
|---|---|---|---|
| 1 | Cam → MediaMTX | RTSP via GStreamer `v4l2src ! v4l2h264enc ! rtspclientsink` | none (loopback) |
| 2 | MediaMTX → Pi frame_extractor | RTSP `localhost:8554/camN` | none |
| 3 | Pi → `frames-in` | gRPC | service account key file (`sa-frame-extractor`) on Pi |
| 4 | `frames-in` → dispatcher | Eventarc push (CloudEvent) | Pub/Sub SA token (impersonates dispatcher SA) |
| 5 | Dispatcher → Vertex AI | HTTPS, `Authorization: Bearer` | google-auth ADC; dispatcher SA has `roles/aiplatform.user` |
| 6 | Judge → `detections-out` | gRPC | judge container runs as `judge-svc` (manual binding for Vertex AI) |
| 7 | `detections-out` → alert-manager | Eventarc push | alert-manager SA |
| 8 | Alert manager → Firestore | gRPC | `roles/datastore.user` |
| 9 | Alert manager → Gmail SMTP | TLS to `smtp.gmail.com:465` | Gmail app password (env var) |
| 10 | Browser → MediaMTX | HTTPS WHEP → WebRTC (UDP) | Cloudflare Tunnel hostname; no app auth |
| 11 | Browser → Firestore | gRPC over HTTPS | Firebase web SDK config (public API key) |

## Latency budget (observed during testing, very rough)

| Stage | ms |
|---|---|
| Pi capture + JPEG encode | 50–150 |
| Pi → Pub/Sub publish ack | 100–300 |
| Eventarc dispatch latency | 200–500 |
| Vertex AI inference (T4, YOLOv8x, 720p) | 300–600 |
| `detections-out` → alert-manager → Firestore | 200–400 |
| Firestore → browser onSnapshot | 200–500 |
| **End-to-end frame→bbox in dashboard** | ≈ 1–2 s |

## Failure modes

- **Frames-in backlog**: Pi keeps capturing while judge is undeployed → Pub/Sub stockpiles, then crashes judge on redeploy. Mitigation: see [[09-deployment-ops#Purge frames-in backlog]].
- **Cooldown leak**: alert_cooldowns docs persist after a test run; new HIGH_SEV detections get suppressed for 5 min. Flush: [[09-deployment-ops#Flush Firestore]].
- **Streak filter on noisy classes**: cables / wire bundles get misclassified as `spagetti` once; streak ≥2 mostly suppresses this — but a sustained mis-detection still fires.
- **Vertex AI cold start**: container takes ~30 s to load YOLOv8x; first POST after deploy will timeout (`60 s` budget in dispatcher). Pub/Sub retries handle it.

See [[12-glossary]] for any term used above.
