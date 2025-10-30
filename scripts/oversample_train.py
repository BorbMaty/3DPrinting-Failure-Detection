#!/usr/bin/env python3
import os, glob, argparse, shutil
from collections import Counter, defaultdict
import random

p = argparse.ArgumentParser()
p.add_argument("--train_images", required=True)
p.add_argument("--train_labels", required=True)
p.add_argument("--target_perc_of_max", type=float, default=0.6,
              help="A ritka osztály cél szintje a leggyakoriból (pl. 0.6 = 60%)")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
random.seed(args.seed)

label_paths = glob.glob(os.path.join(args.train_labels, "*.txt"))
cls_counts = Counter()
img_classes = {}

for lp in label_paths:
    base = os.path.splitext(os.path.basename(lp))[0]
    s = set()
    with open(lp) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 5: continue
            c = int(float(parts[0])); s.add(c); cls_counts[c] += 1
    if s: img_classes[base] = s

if not cls_counts:
    print("No labels found."); exit(0)

max_cls = max(cls_counts.values())
target = int(max_cls * args.target_perc_of_max)

print("\n📊 Train label counts:")
for c in sorted(cls_counts):
    print(f"Class {c}: {cls_counts[c]}")
print(f"\n🎯 Target per class: ≥ {target} labels")

# képek listája osztályonként
cls2images = defaultdict(list)
for base, s in img_classes.items():
    for c in s:
        cls2images[c].append(base)

dups = 0
for c in sorted(cls_counts):
    need = max(0, target - cls_counts[c])
    if need == 0:
        continue
    candidates = cls2images[c][:]
    if not candidates:
        continue
    print(f"Class {c}: need +{need} labels via duplication")
    i = 0
    while need > 0:
        base = random.choice(candidates)
        img_src = None
        for ext in [".jpg",".jpeg",".png",".JPG",".PNG",".bmp",".tif",".tiff"]:
            pth = os.path.join(args.train_images, base + ext)
            if os.path.exists(pth):
                img_src = pth; img_ext = ext; break
        if not img_src:
            # nincs kép ehhez a labelhez
            continue
        lbl_src = os.path.join(args.train_labels, base + ".txt")

        dup_name = f"{base}__dup{dups}"
        img_dst = os.path.join(args.train_images, dup_name + img_ext)
        lbl_dst = os.path.join(args.train_labels, dup_name + ".txt")

        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)

        dups += 1
        # növeljük a számlálót annyival, ahány c osztály szerepel a fájlban
        added = 0
        with open(lbl_src) as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) < 5: continue
                if int(float(parts[0])) == c:
                    added += 1
        cls_counts[c] += added
        need = max(0, target - cls_counts[c])

print(f"\n✅ Oversampling kész. Duplikált fájlok: {dups}")
print("Új train label counts:")
for c in sorted(cls_counts):
    print(f"Class {c}: {cls_counts[c]}")
