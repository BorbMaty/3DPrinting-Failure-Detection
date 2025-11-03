# 🧠 3D Printing Failure Detection  
**Bachelor Thesis Project – Sapientia EMTE**

---

## 📘 Overview

This repository contains the complete workflow for a **real-time failure detection system for 3D printers**.  
The project leverages **deep learning** and **computer vision** to identify printing defects automatically from live or recorded video feeds, reducing wasted time and material.

The system is designed as part of a **Bachelor’s Thesis** at Sapientia Hungarian University of Transylvania, focusing on building a reproducible, AI-powered detection pipeline using **Docker**, **TensorFlow**, and **YOLOv8**.

---

## 🎯 Objectives

- Detect common 3D printing errors automatically in real-time  
- Train a robust deep learning model on a **custom-annotated dataset**  
- Achieve **high accuracy (mAP ≥ 0.95)** across failure types  
- Build a **Dockerized and reproducible environment**  
- Document the methodology and results for academic purposes  

---

## 🧰 Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming | Python 3.10+, Jupyter Notebook |
| Deep Learning | TensorFlow, Ultralytics YOLOv8 (m / l / x) |
| Computer Vision | OpenCV |
| Data Analysis | Pandas, SciPy |
| Containerization | Docker |
| Version Control | Git & GitHub |
| Annotation | CVAT (Computer Vision Annotation Tool) |
| Documentation | LaTeX (Bachelor Thesis Report) |

---

## 🗂️ Project Structure

```
3DPrinting-Failure-Detection/
├── scripts/
│   ├── teacher.py          # Training pipeline for YOLOv8
│   ├── tester.py           # Model inference and visualization
│   ├── dataset_splitter.py # Dataset preparation utilities
│   ├── metrics.json        # Performance metrics (mAP, precision, recall)
│   └── cfg/                # Configurations and YAMLs
│
├── dataset/
│   ├── train/              # Training images
│   ├── val/                # Validation images
│   ├── testPhotos/         # Sample test images
│   └── data.yaml           # YOLO dataset configuration
│
├── docker/
│   └── Dockerfile          # Dockerized environment setup
│
├── runs/
│   ├── train_YYYY-MM-DD_HH-MM-SS/   # Timestamped YOLO training logs
│   └── detect/                      # Inference results
│
├── report/
│   └── thesis_latex/       # LaTeX source for Bachelor Thesis
│
└── README.md
```

---

## 🧾 Dataset and Labels

The dataset was built using **real-world 3D printing scenarios**, collected from various printers and environments.  
Annotations were performed manually in **CVAT**, and exported in YOLO-compatible format.

### Current Failure Categories (9):
1. `stringing`  
2. `warping`  
3. `layer_shift`  
4. `under_extrusion`  
5. `over_extrusion`  
6. `nozzle_clog`  
7. `foreign_object_on_print_area`  
8. `not_sticking`  
9. `spaghetti`

---

## ⚙️ Training and Validation

Training is handled by the `teacher.py` script, which supports automatic data validation and experiment tracking.

### Example Training Command:
```bash
python3 teacher.py --data dataset/data.yaml --model yolov8m.pt --epochs 100 --batch 16
```

### Validation Command:
```bash
python3 tester.py --weights runs/train_*/weights/best.pt --source dataset/testPhotos/
```

### Models Used
- ✅ YOLOv8m  
- ✅ YOLOv8l  
- ⏳ Planned: YOLOv8x (for extended experimentation)

---

## 📊 Model Performance

--TODO

All metrics are exported to `metrics.json` for reproducibility.  
Visual results (bounding boxes, confusion matrix) are stored in the `runs/detect/` directory.

---

## 🧪 Experimental Pipeline

### 1️⃣ Dataset Collection
- Collected from multiple 3D prints under varying lighting conditions  
- Each defect type photographed across multiple materials and layers  

### 2️⃣ Annotation
- Manually labeled using CVAT  
- Exported to YOLO format (`.txt` per image)

### 3️⃣ Preprocessing
- Data augmentation with OpenCV (rotation, contrast, noise, blur)  
- Stratified split into `train`, `val`, and `test` sets  

### 4️⃣ Model Training
- Conducted on local GPU (RTX 3060) and Google Colab (T4) Also on the University Computer NVDIA A500 
- Trained YOLOv8m and YOLOv8l variants  
- Early stopping and learning rate scheduling applied  

### 5️⃣ Validation
- Confusion matrix generation  
- Precision-recall curve evaluation  
- Metric logging in JSON  

### 6️⃣ Dockerized Deployment (in progress)
- All dependencies defined in `Dockerfile`  
- Environment reproducible across machines  

---

## 🧭 Achieved Milestones

| Status | Milestone |
|:-------|:-----------|
| ✅ | Dataset fully collected and annotated |
| ✅ | CVAT YOLO-format export finalized |
| ✅ | YOLOv8m & YOLOv8l trained successfully |
| ✅ | Automatic metric export implemented |
| ✅ | Docker environment built |
| ✅ | LaTeX documentation started for thesis |
| ⏳ | Raspberry Pi live inference integration |
| ⏳ | YOLOv8x training and comparison |
| ⏳ | Real-time dashboard (planned) |

---

## 🧩 Future Work

- Integration with **Raspberry Pi** for real-time monitoring  
- Optimization for **low-latency video stream inference**  
- Expansion to **YOLOv8x** for higher capacity and comparison  
- Improved dataset balance (augment rare failure types)  
- Final thesis documentation and defense preparation  

---

## 🧠 Academic Context

This project forms the practical and research component of the **Bachelor’s Thesis** at  
**Sapientia Hungarian University of Transylvania – Faculty of Electrical Engineering**.

It demonstrates the use of **AI-based defect detection** in additive manufacturing,  
bridging **machine learning** and **industrial automation** applications.

---



---

## 🧾 License

This repository is provided for **academic and research purposes only**.  
The dataset and trained models are proprietary and not intended for commercial redistribution.

If used in related research, please cite this project appropriately.

---

## 🧱 How to Run (Quick Start)

```bash
# Clone the repository
git clone https://github.com/BorbMaty/3DPrinting-Failure-Detection.git
cd 3DPrinting-Failure-Detection

# Build the Docker container
docker build -t printer-failure-detector .

# Run the container
docker run -it --gpus all printer-failure-detector
```
