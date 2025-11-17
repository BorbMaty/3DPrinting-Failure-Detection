# split_dataset.py
# YOLO dataset szétosztás train/val mappákra (80/20 arány)
# by Selene ✨

import os
import shutil
import random
from pathlib import Path

def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Mappák létrehozása
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Összes kép kigyűjtése
    image_files = [
        f for f in images_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]
    random.shuffle(image_files)

    # Szétosztás
    split_idx = int(len(image_files) * (1 - val_ratio))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def copy_files(files, split):
        for img in files:
            # kép másolása
            shutil.copy(img, output_dir / "images" / split / img.name)

            # annotáció másolása, ha van
            label = labels_dir / (img.stem + ".txt")
            if label.exists():
                shutil.copy(label, output_dir / "labels" / split / label.name)

    copy_files(train_files, "train")
    copy_files(val_files, "val")

    print(f"Összes kép: {len(image_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    print(f"Train/val mappák ide készültek: {output_dir.resolve()}")

if __name__ == "__main__":
    # Példa használat:
    split_dataset(
        images_dir="/home/lucy/Desktop/Dolgaim/Egyetem/Államvizsga/BA_Dataset/YoloDataset/merged/images/train",   # kepek mappája
        labels_dir="/home/lucy/Desktop/Dolgaim/Egyetem/Államvizsga/BA_Dataset/YoloDataset/merged/labels/train",   # annotációk mappája
        output_dir="/home/lucy/Desktop/Dolgaim/Egyetem/Államvizsga/BA_Dataset/YoloDataset/splittedDataset",    # ide készül az új struktúra
        val_ratio=0.2,                 # 20% validáció
        seed=42,
    )
