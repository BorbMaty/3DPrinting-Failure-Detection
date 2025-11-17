from __future__ import annotations
import cv2
import numpy as np

# Ultralytics Results -> annotált frame
def draw_detections(img: np.ndarray, result) -> np.ndarray:
    """
    result: ultralytics.engine.results.Results
    """
    out = img.copy()
    boxes = result.boxes  # Boxes object
    names = result.names  # id->class név

    if boxes is None or len(boxes) == 0:
        return out

    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c, p in zip(xyxy, clss, conf):
        label = f"{names.get(c, str(c))} {p:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # címke háttér
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return out
