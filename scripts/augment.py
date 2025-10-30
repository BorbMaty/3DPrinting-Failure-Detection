#!/usr/bin/env python3
"""
Lazy YOLO Dataset Augmenter (final fixed)
-----------------------------------------
- Nem dob el képet, mindenből készít augmentált példányt
- Minden bbox visszavágva (clamp [0..1])
- Smart MixUp (~20%), Mosaic (~10%)
- YOLOv8-kompatibilis címkék

Használat:
  python3 augment_yolo_dataset_lazy_fixed.py \
    --input /path/to/images/train \
    --labels /path/to/labels/train \
    --output /path/to/output \
    --multiplier 4
"""

import os, glob, cv2, random, argparse, numpy as np
import albumentations as A

# ----- CLI -----
p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="Képek mappa (pl. dataset/images/train)")
p.add_argument("--labels", required=True, help="Labels mappa (pl. dataset/labels/train)")
p.add_argument("--output", required=True, help="Kimeneti gyökérmappa (pl. dataset_aug_final)")
p.add_argument("--multiplier", type=int, default=4, help="Minden képből ennyi augmentált verzió")
args = p.parse_args()

# ----- Paths -----
os.makedirs(args.output, exist_ok=True)
out_img_dir = os.path.join(args.output, "images")
out_lbl_dir = os.path.join(args.output, "labels")
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lbl_dir, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".bmp", ".tif", ".tiff"]

# ----- Smart MixUp párok -----
MIXUP_ALLOWED = {(0,1),(3,4),(6,7),(0,8),(5,0)}

# ----- Albumentations augment pipeline -----
base_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.Rotate(limit=15, p=0.5),
], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], clip=True, min_visibility=0.0))

# ----- Helper függvények -----
def clamp01(x): return max(0.0, min(1.0, float(x)))

def lazy_clamp_boxes(boxes, labels):
    out_b, out_l = [], []
    for (x, y, w, h), c in zip(boxes, labels):
        x, y, w, h = clamp01(x), clamp01(y), clamp01(w), clamp01(h)
        out_b.append([x, y, w, h])
        out_l.append(c)
    return out_b, out_l

def load_image_and_labels(img_path, lbl_dir):
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(lbl_dir, base + ".txt")
    img = cv2.imread(img_path)
    if img is None: return None, [], []
    bboxes, labels = [], []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(parts[0]); x, y, w, h = map(float, parts[1:5])
                bboxes.append([x, y, w, h]); labels.append(cls)
    return img, bboxes, labels

def mixup_allowed(labels_a, labels_b):
    for a in set(labels_a):
        for b in set(labels_b):
            if (a,b) in MIXUP_ALLOWED or (b,a) in MIXUP_ALLOWED:
                return True
    return False

def mixup_augment(img1, boxes1, labels1, img2, boxes2, labels2):
    lam = random.uniform(0.4, 0.6)
    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    img1r = cv2.resize(img1, (w, h))
    img2r = cv2.resize(img2, (w, h))
    mix = cv2.addWeighted(img1r, lam, img2r, 1 - lam, 0)
    boxes = boxes1 + boxes2
    labels = labels1 + labels2
    return mix, boxes, labels

def mosaic_augment(imgs, lbls):
    size = 640
    out = np.zeros((size, size, 3), dtype=np.uint8)
    yc, xc = size // 2, size // 2
    idx = [(0,0),(0,1),(1,0),(1,1)]
    new_boxes, new_labels = [], []
    for i, (r, c) in enumerate(idx):
        img = imgs[i]
        h, w = img.shape[:2]
        scale = random.uniform(0.4, 0.6)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
        ih, iw = img.shape[:2]
        y1, x1 = r * yc, c * xc
        out[y1:y1+ih, x1:x1+iw] = img[:min(ih, size-y1), :min(iw, size-x1)]
        for (x, y, bw, bh), lab in lbls[i]:
            nx = (x * iw + x1) / size
            ny = (y * ih + y1) / size
            nbw, nbh = (bw * iw) / size, (bh * ih) / size
            new_boxes.append([nx, ny, nbw, nbh])
            new_labels.append(lab)
    return out, new_boxes, new_labels

# ----- Képek beolvasása -----
images = [f for f in glob.glob(os.path.join(args.input, "*")) if os.path.splitext(f)[1] in IMG_EXTS]
print(f"Talált képek: {len(images)}")

made = 0

for img_path in images:
    img, boxes, labels = load_image_and_labels(img_path, args.labels)
    if img is None or not labels: continue
    base = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(args.multiplier):
        r = random.random()
        mode = "basic"
        if r < 0.1:
            mode = "mosaic"
        elif r < 0.3:
            mode = "mixup"

        if mode == "mosaic" and len(images) >= 4:
            chosen = random.sample(images, 4)
            imgs, lbls = [], []
            for c in chosen:
                img_c, bb, lb = load_image_and_labels(c, args.labels)
                if img_c is None: continue
                imgs.append(img_c)
                lbls.append(list(zip(bb, lb)))
            if len(imgs) == 4:
                aug_img, aug_boxes, aug_labels = mosaic_augment(imgs, lbls)
            else:
                aug_img, aug_boxes, aug_labels = img, boxes, labels
        elif mode == "mixup":
            img2 = random.choice(images)
            img_b, b2, l2 = load_image_and_labels(img2, args.labels)
            if img_b is not None and mixup_allowed(labels, l2):
                aug_img, aug_boxes, aug_labels = mixup_augment(img, boxes, labels, img_b, b2, l2)
            else:
                aug_img, aug_boxes, aug_labels = img, boxes, labels
        else:
            t = base_transform(image=img, bboxes=boxes, class_labels=labels)
            aug_img = t["image"]
            aug_boxes = t["bboxes"]
            aug_labels = t["class_labels"]

        aug_boxes, aug_labels = lazy_clamp_boxes(aug_boxes, aug_labels)

        new_name = f"{base}_aug{i+1}"
        cv2.imwrite(os.path.join(out_img_dir, new_name + ".jpg"), aug_img)
        with open(os.path.join(out_lbl_dir, new_name + ".txt"), "w") as f:
            for c, (x, y, w, h) in zip(aug_labels, aug_boxes):
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        made += 1

print(f"✅ Lazy augmentáció kész ({made} augmentált kép mentve).")
