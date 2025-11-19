# 🧩 3D Printing Failure Detection

**YOLOv8-Based Real-Time Failure Detection System with Dockerized
Training Pipeline**

This repository contains a fully modular, GPU-accelerated YOLOv8 system
for detecting failure modes in FDM 3D printing.\
It includes a refactored Python package, Dockerized training
environment, image/video inference tools, and a fully reproducible
workflow designed for both local PCs and remote GPU servers (including
SSH Windows hosts).

## 📁 Project Structure

    project_root/
    ├─ src/
    │  └─ printerfail/
    │     ├─ cli/
    │     ├─ io/
    │     ├─ model/
    │     ├─ config/
    │     └─ utils/
    ├─ dataset/
    ├─ runs/
    ├─ docker-compose.yml
    ├─ Dockerfile
    ├─ requirements.txt
    └─ pyproject.toml

## 🐳 Docker Features

-   CUDA-accelerated PyTorch\
-   Ultralytics YOLOv8\
-   OpenCV + FFmpeg\
-   Live-mounted code\
-   Reproducible execution\
-   Bind-mounted dataset and runs folders

## 🚀 Getting Started

### 1. Clone

    git clone https://github.com/<yourname>/<repo>.git
    cd <repo>

### 2. Build

    docker compose build

### 3. Start shell

    docker compose run --rm yolo

Inside container:

    pip install -e .
    python -c "import torch; print('cuda?', torch.cuda.is_available())"

## 🏋️ Training Example

    printerfail-train --override   train.data_yaml=/workspace/dataset/data.yaml   train.model=/workspace/dataset/yolov8x.pt   train.imgsz=896   train.epochs=120   train.batch=6   train.device=0   train.project=/workspace/runs   train.name=yolov8x_finetune_highmap   train.patience=50   train.lr0=0.001   train.lrf=0.05   train.mosaic=0.5   train.mixup=0.1   train.degrees=5   train.scale=0.1   train.translate=0.05   train.fliplr=0.5

## 📸 Prediction

    printerfail-predict --weights /workspace/runs/.../best.pt   --source /workspace/dataset/val/images   --out /workspace/runs/predict_vis

## 🎞️ Video

    printerfail-video --weights ... --source video.avi --out output.mp4

## 💾 Persistence

Your dataset and trained models always stay on the host due to bind
mounts.

## 🗑 Cleanup

    docker compose down
    docker image prune -f
