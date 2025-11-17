import mss
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
sct = mss.mss()

monitor = sct.monitors[2]  # 1=all, 2=second monitor

while True:
    img = np.array(sct.grab(monitor))
    results = model(img)
    annotated = results[0].plot()

    cv2.imshow("Processed Monitor Feed", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
