
import asyncio
import json
import random
import time
import websockets

# WebSocket server: ws://0.0.0.0:8765
# Dummy detections: random box sometimes, per camera

CAMS = ["cam1", "cam2", "cam3"]

clients = set()

async def handler(ws):
    clients.add(ws)
    try:
        # keep connection open
        await ws.wait_closed()
    finally:
        clients.discard(ws)

def make_box():
    # normalized bbox
    x = random.uniform(0.05, 0.70)
    y = random.uniform(0.05, 0.70)
    w = random.uniform(0.10, 0.25)
    h = random.uniform(0.10, 0.25)
    return {"x": x, "y": y, "w": w, "h": h, "cls": "fail", "conf": round(random.uniform(0.6, 0.95), 2)}

async def broadcaster():
    last_clear = {c: 0 for c in CAMS}

    while True:
        await asyncio.sleep(0.25)  # 4 Hz tick

        cam = random.choice(CAMS)

        # 25% chance to emit a detection, otherwise maybe emit clear every ~2s
        if random.random() < 0.25:
            msg = {"cam": cam, "ts": time.time(), "boxes": [make_box()]}
        else:
            # send an occasional clear so overlays can disappear
            if time.time() - last_clear[cam] < 2.0:
                continue
            last_clear[cam] = time.time()
            msg = {"cam": cam, "ts": time.time(), "boxes": []}

        if clients:
            payload = json.dumps(msg)
            await asyncio.gather(*(c.send(payload) for c in list(clients)), return_exceptions=True)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await broadcaster()

if __name__ == "__main__":
    asyncio.run(main())