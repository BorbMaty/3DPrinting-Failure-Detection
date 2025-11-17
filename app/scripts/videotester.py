from ultralytics import YOLO
import os

# --- 1️⃣ Modell betöltése ---
model = YOLO('/home/lucy/Desktop/yolov8x_finetune_highmap/weights/best.pt')  # saját YOLOv8 modell

# --- 2️⃣ Videó forrás megadása ---
video_path = '/home/lucy/Desktop/datasetmaybefinal/testvideo/testvideo.mp4'

# --- 3️⃣ Detektálás és mentés ---
results = model.predict(
    source=video_path,   # bemeneti videó
    save=True,           # mentse az eredményt runs/detect/exp mappába
    conf=0.25,           # confidence threshold
    show=False,          # ne nyisson ablakot
    stream=False         # teljes videó feldolgozása (nem képkockánként)
)

print(f"🎥 Video processed: {os.path.basename(video_path)}")
print("✅ Eredmény mentve a runs/detect/exp mappába")
