from ultralytics import YOLO
from pathlib import Path

# --- 1️⃣ Modell betöltése ---
model = YOLO('/home/lucy/Desktop/runs/ba_yolov8l_first/weights/best.pt')  # saját YOLOv8 modell

# --- 2️⃣ Tesztképek mappa ---
test_folder = '/home/lucy/Desktop/datasetmaybefinal/testPhotos'
image_paths = []
for ext in ("*.jpg", "*.png", "*.jpeg", "*.webp"):
    image_paths.extend(Path(test_folder).glob(ext))

# --- 3️⃣ Képek feldolgozása ---
for image_path in image_paths:
    results = model.predict(
        source=str(image_path),
        save=True,         # mentse az eredményt runs/detect/exp mappába
        conf=0.25,         # confidence threshold
        show=False         # ne nyisson ablakot
    )

    print(f"📸 Processed {image_path.name}")
    # Részletes eredmények
    for r in results:
        # r.show()  # interaktív megjelenítéshez
        print(r.boxes)  # bounding box adatok
