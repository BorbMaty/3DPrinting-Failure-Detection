from __future__ import annotations
import cv2
from pathlib import Path
from typing import Iterator


class VideoReader:
    def __init__(self, source: str | int):
        try:
            src_int = int(source)
            self.cap = cv2.VideoCapture(src_int)
        except (ValueError, TypeError):
            self.cap = cv2.VideoCapture(str(source))

        if not self.cap.isOpened():
            raise RuntimeError(f"Nem tudom megnyitni a videót/kamerát: {source}")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            raise StopIteration
        return frame

    def show(self, frame, win_name: str = "printerfail"):
        cv2.imshow(win_name, frame)

    def key_pressed(self, key_code: int) -> bool:
        return (cv2.waitKey(1) & 0xFF) == key_code

    def release(self):
        self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


class VideoWriter:
    def __init__(self, path: str, fps: float, width: int, height: int):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # MP4 (H264) vagy AVI fallback
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if p.suffix.lower() == ".mp4" else "XVID"))
        self.out = cv2.VideoWriter(str(p), fourcc, fps, (width, height))
        if not self.out.isOpened():
            raise RuntimeError(f"Nem tudom megnyitni a kimeneti videót: {path}")

    def write(self, frame):
        self.out.write(frame)

    def release(self):
        self.out.release()
