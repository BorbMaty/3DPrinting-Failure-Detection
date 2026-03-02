import asyncio
import json
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque

import cv2
import websockets
from ultralytics import YOLO


# -----------------------
# CONFIG
# -----------------------
WS_HOST = "0.0.0.0"
WS_PORT = 8765

MODEL_PATH = "best.pt"  # <-- ha máshol van: "/abs/path/to/best.pt"

RTSP_URLS = {
    "cam1": "rtsp://192.168.0.118:8554/cam1",
    "cam2": "rtsp://192.168.0.118:8554/cam2",
    "cam3": "rtsp://192.168.0.118:8554/cam3",
}

# Inference settings
CONF_THRES = 0.35
IOU_THRES = 0.45
INFER_FPS_PER_CAM = 3.0          # mennyi YOLO infer/cam (állítsd lejjebb ha laggol)
MAX_FPS_CAPTURE = 30.0           # cap a decode-nak (OpenCV amúgy is best-effort)
SEND_CLEAR_EVERY_S = 1.5         # ha nincs detection, ennyi időnként küld üres boxot (hogy eltűnjön overlay)
MIN_SEND_INTERVAL_S = 0.15       # ne spameld a UI-t

# Majority alert settings (2/3)
ALERT_WINDOW_S = 2.0             # ennyi időn belül számít együtt a több kamera jelzése
ALERT_MIN_CAMS = 2               # 2 kamera kell a 3-ból
ALERT_COOLDOWN_S = 6.0           # ennyi ideig nem küld új alertet (ne spam)
REQUIRE_SAME_CLASS = False       # indulásnak inkább False (külön szögek néha mást klasztereznek)


# -----------------------
# STATE
# -----------------------
clients_lock = threading.Lock()
clients: set = set()

latest_lock = threading.Lock()
latest_msgs: Dict[str, Dict[str, Any]] = {}  # key -> last msg
# kulcsok: "cam1"/"cam2"/"cam3" + "ALERT"


@dataclass
class RateLimiter:
    period_s: float
    next_t: float = 0.0

    def wait(self):
        now = time.time()
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t = time.time() + self.period_s


def normalize_box_xyxy(x1, y1, x2, y2, w, h):
    # clamp
    x1 = max(0.0, min(float(x1), float(w)))
    x2 = max(0.0, min(float(x2), float(w)))
    y1 = max(0.0, min(float(y1), float(h)))
    y2 = max(0.0, min(float(y2), float(h)))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return {
        "x": x1 / w,
        "y": y1 / h,
        "w": bw / w,
        "h": bh / h,
    }


class AlertManager:
    """
    Majority vote across cameras within a sliding window.
    Register per-camera "failure events". When >= ALERT_MIN_CAMS cameras have events
    inside ALERT_WINDOW_S (and cooldown passes), emit an alert payload.

    Optionally require a common class across the participating cameras.
    """

    def __init__(
        self,
        window_s: float,
        min_cams: int,
        cooldown_s: float,
        require_same_class: bool = False,
    ):
        self.window_s = float(window_s)
        self.min_cams = int(min_cams)
        self.cooldown_s = float(cooldown_s)
        self.require_same_class = bool(require_same_class)

        self._lock = threading.Lock()
        # cam -> deque[(ts, cls)]
        self._events: Dict[str, deque[Tuple[float, str]]] = defaultdict(deque)
        self._last_alert_ts = 0.0

    def register(self, cam: str, cls_name: str, ts: float) -> Optional[dict]:
        now = float(ts)

        with self._lock:
            self._events[cam].append((now, cls_name))
            self._cleanup(now)

            active_cams = self._active_cams(now)
            if len(active_cams) < self.min_cams:
                return None

            common_classes = None
            if self.require_same_class:
                common_classes = self._common_classes(active_cams)
                if not common_classes:
                    return None

            if now - self._last_alert_ts < self.cooldown_s:
                return None

            self._last_alert_ts = now

            payload = {
                "type": "ALERT",
                "cam": "ALERT",          # UI-ban könnyű felismerni
                "ts": now,
                "cameras": sorted(active_cams),
                "reason": "majority_failure",
                "window_s": self.window_s,
                "min_cams": self.min_cams,
            }
            if self.require_same_class:
                payload["classes"] = sorted(common_classes)
            return payload

    def _cleanup(self, now: float) -> None:
        cutoff = now - self.window_s
        for cam in list(self._events.keys()):
            dq = self._events[cam]
            while dq and dq[0][0] < cutoff:
                dq.popleft()
            if not dq:
                self._events.pop(cam, None)

    def _active_cams(self, now: float) -> List[str]:
        cutoff = now - self.window_s
        out = []
        for cam, dq in self._events.items():
            # at least one event still in window
            if dq and dq[-1][0] >= cutoff:
                out.append(cam)
        return out

    def _common_classes(self, cams: List[str]) -> set:
        sets = []
        for cam in cams:
            sets.append({cls for _, cls in self._events.get(cam, [])})
        if not sets:
            return set()
        common = sets[0]
        for s in sets[1:]:
            common &= s
        return common


# single shared instance
alert_mgr = AlertManager(
    window_s=ALERT_WINDOW_S,
    min_cams=ALERT_MIN_CAMS,
    cooldown_s=ALERT_COOLDOWN_S,
    require_same_class=REQUIRE_SAME_CLASS,
)


def pick_dominant_class(boxes_out: List[Dict[str, Any]]) -> str:
    """
    Pick a representative class for the frame:
    choose the highest-confidence box's class.
    """
    if not boxes_out:
        return "unknown"
    best = max(boxes_out, key=lambda b: float(b.get("conf", 0.0)))
    return str(best.get("cls", "unknown"))


def cam_worker(cam: str, url: str, model: YOLO):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[{cam}] ERROR: cannot open RTSP: {url}")
        return

    infer_rl = RateLimiter(period_s=1.0 / max(0.1, INFER_FPS_PER_CAM))
    send_rl = RateLimiter(period_s=MIN_SEND_INTERVAL_S)

    last_clear_sent = 0.0

    print(f"[{cam}] capture started: {url}")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{cam}] WARN: frame read failed; retrying...")
            time.sleep(0.25)
            continue

        infer_rl.wait()

        h, w = frame.shape[:2]

        res = model.predict(
            source=frame,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )[0]

        boxes_out: List[Dict[str, Any]] = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            names = model.names  # id -> name

            for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
                bb = normalize_box_xyxy(x1, y1, x2, y2, w, h)
                bb["conf"] = float(conf)
                bb["cls"] = str(names.get(int(cls_id), str(cls_id)))
                boxes_out.append(bb)

        now = time.time()

        # throttle WS update rate (avoid spam)
        send_rl.wait()

        if boxes_out:
            # 1) update per-camera detection msg
            msg = {"cam": cam, "ts": now, "boxes": boxes_out}
            with latest_lock:
                latest_msgs[cam] = msg

            # 2) register failure for majority vote
            dominant_cls = pick_dominant_class(boxes_out)
            alert = alert_mgr.register(cam=cam, cls_name=dominant_cls, ts=now)

            if alert:
                # publish ALERT as another "stream" message
                with latest_lock:
                    latest_msgs["ALERT"] = alert

        else:
            # send clear periodically so overlay disappears
            if now - last_clear_sent >= SEND_CLEAR_EVERY_S:
                msg = {"cam": cam, "ts": now, "boxes": []}
                with latest_lock:
                    latest_msgs[cam] = msg
                last_clear_sent = now


async def ws_handler(ws):
    with clients_lock:
        clients.add(ws)
    try:
        # on connect, push current state once (cams + possible ALERT)
        with latest_lock:
            snapshot = list(latest_msgs.values())
        for msg in snapshot:
            await ws.send(json.dumps(msg))

        await ws.wait_closed()
    finally:
        with clients_lock:
            clients.discard(ws)


async def broadcaster_loop():
    last_sent: Dict[str, float] = {}
    while True:
        await asyncio.sleep(0.05)

        with latest_lock:
            msgs = dict(latest_msgs)

        to_send = []
        for key, msg in msgs.items():
            ts = float(msg.get("ts", 0.0))
            if last_sent.get(key, 0.0) < ts:
                to_send.append((key, msg))

        if not to_send:
            continue

        payloads = [json.dumps(m) for _, m in to_send]

        with clients_lock:
            conns = list(clients)

        if not conns:
            continue

        for p in payloads:
            await asyncio.gather(*(c.send(p) for c in conns), return_exceptions=True)

        for key, msg in to_send:
            last_sent[key] = float(msg.get("ts", time.time()))


async def main():
    print(f"[WS] serving on ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(ws_handler, WS_HOST, WS_PORT):
        await broadcaster_loop()


if __name__ == "__main__":
    # Load model once, share across threads
    model = YOLO(MODEL_PATH)

    # Start one thread per camera
    for cam, url in RTSP_URLS.items():
        t = threading.Thread(target=cam_worker, args=(cam, url, model), daemon=True)
        t.start()

    asyncio.run(main())