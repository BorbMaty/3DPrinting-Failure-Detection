#!/usr/bin/env python3
# split_train_val_stratified.py
import os, glob, argparse, random
from collections import defaultdict, Counter
from shutil import copy2

p = argparse.ArgumentParser()
p.add_argument("--labels", required=True)
p.add_argument("--images", required=True)
p.add_argument("--output", required=True)
p.add_argument("--val_ratio", type=float, default=0.2)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

random.seed(args.seed)

# --- 1) Beolvasás: minden kép -> osztálykészlet ---
label_files = glob.glob(os.path.join(args.labels, "*.txt"))
img2classes = {}
all_classes = set()

for lbl in label_files:
    base = os.path.splitext(os.path.basename(lbl))[0]
    clsset = set()
    with open(lbl) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: 
                continue
            try:
                c = int(float(parts[0]))
            except ValueError:
                continue
            clsset.add(c)
            all_classes.add(c)
    if clsset:
        img2classes[base] = clsset

bases = list(img2classes.keys())
classes = sorted(all_classes)

# --- 2) Cél darabszám classonként train/val ---
total_class_counts = Counter()
for s in img2classes.values():
    total_class_counts.update(s)

target_val = {c: int(round(total_class_counts[c] * args.val_ratio)) for c in classes}
target_train = {c: total_class_counts[c] - target_val[c] for c in classes}

# --- 3) Iterative stratification (Sechidis) egyszerűsített megvalósítás ---
unassigned = set(bases)
val_set, train_set = set(), set()
cur_val = Counter()
cur_train = Counter()

# ritka osztályok előre
def rarity_key(b):
    # min gyakoriság a kép osztályai közül
    return min(total_class_counts[c] for c in img2classes[b])

# Képeket ritkaság szerint, majd véletlenítve rendezzük
ordered = sorted(unassigned, key=rarity_key)
random.shuffle(ordered)

for b in ordered:
    need_val = sum(max(0, target_val[c] - cur_val[c]) for c in img2classes[b])
    need_train = sum(max(0, target_train[c] - cur_train[c]) for c in img2classes[b])

    # ha mindkettő tele lenne, ahova kisebb a túltöltés
    if need_val > need_train:
        dest = "val"
    elif need_train > need_val:
        dest = "train"
    else:
        # tie-breaker: amelyikben kevesebb az összes osztály kitöltése
        sum_val_gap = sum(max(0, target_val[c] - cur_val[c]) for c in classes)
        sum_train_gap = sum(max(0, target_train[c] - cur_train[c]) for c in classes)
        dest = "val" if sum_val_gap >= sum_train_gap else "train"

    if dest == "val":
        val_set.add(b)
        cur_val.update(img2classes[b])
    else:
        train_set.add(b)
        cur_train.update(img2classes[b])

# --- 4) Mappák és másolás ---
def ensure_dirs(root):
    for sub in ["train/images","train/labels","val/images","val/labels"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)

def find_image(base):
    for ext in [".jpg",".jpeg",".png",".JPG",".PNG",".bmp",".tif",".tiff"]:
        p = os.path.join(args.images, base + ext)
        if os.path.exists(p): return p
    return None

ensure_dirs(args.output)

def copy_one(base, split):
    img = find_image(base)
    lbl = os.path.join(args.labels, base + ".txt")
    if not img or not os.path.exists(lbl): 
        return False
    dst_i = os.path.join(args.output, f"{split}/images", os.path.basename(img))
    dst_l = os.path.join(args.output, f"{split}/labels", os.path.basename(lbl))
    copy2(img, dst_i); copy2(lbl, dst_l)
    return True

for b in train_set: copy_one(b, "train")
for b in val_set:   copy_one(b, "val")

# --- 5) Riport ---
def count_split(split_set):
    cc = Counter()
    for b in split_set:
        cc.update(img2classes[b])
    return cc

train_cc = count_split(train_set)
val_cc   = count_split(val_set)

print("\n📊 Összes (címke nélküli) képfájl:", len(bases))
print("🎯 Cél val class-darabok:", target_val)
print("\nTrain class-count:", dict(sorted(train_cc.items())))
print("Val   class-count:", dict(sorted(val_cc.items())))
print(f"\n✅ Kész. Train képek: {len(train_set)} | Val képek: {len(val_set)}")
print(f"Eredmény: {args.output}/train és {args.output}/val")
