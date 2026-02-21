#!/usr/bin/env python3
import os
import glob
import argparse
import random
from collections import Counter, defaultdict
from shutil import copy2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_images", required=True, help="train/images mappa")
    p.add_argument("--train_labels", required=True, help="train/labels mappa")
    p.add_argument("--target_per_class", type=int, default=500,
                   help="minimum cél instance/class (csak aki ez alatt van, azt húzzuk fel)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # --- 1) Beolvasás: melyik képben milyen classok vannak, és class count ---
    label_files = glob.glob(os.path.join(args.train_labels, "*.txt"))
    img2classes = {}
    class_counts = Counter()
    class2images = defaultdict(list)

    def find_image(base):
        for ext in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".PNG"]:
            p = os.path.join(args.train_images, base + ext)
            if os.path.exists(p):
                return p
        return None

    for lbl in label_files:
        base = os.path.splitext(os.path.basename(lbl))[0]
        classes_here = set()
        with open(lbl, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    c = int(float(parts[0]))
                except ValueError:
                    continue
                classes_here.add(c)

        if not classes_here:
            continue

        img_path = find_image(base)
        if not img_path:
            continue

        img2classes[base] = classes_here
        for c in classes_here:
            class_counts[c] += 1
            class2images[c].append(base)

    print("📊 Jelenlegi class count (train):", dict(sorted(class_counts.items())))
    print("🎯 Cél minimum instance/class:", args.target_per_class)

    # --- 2) Döntés: kit kell oversample-elni? ---
    targets = {}
    for c, cnt in class_counts.items():
        if cnt >= args.target_per_class:
            targets[c] = cnt      # ehhez NEM nyúlunk, marad
        else:
            targets[c] = args.target_per_class  # ezt felhúzzuk eddig

    print("🎯 Target instance/class:", targets)

    # --- 3) Oversample ciklus ---
    # új fájlneveknél suffix: _ov1, _ov2, ...
    duplicate_index = 1

    # Hogy ne lépjük túl brutálisan, adunk egy egyszerű limitet:
    max_copies_per_image = 10  # safeguard
    used_copies = Counter()     # hányszor duplikáltunk már egy base-t

    # addig megyünk classonként, míg el nem érjük a targetet
    # több fordulóban, hogy minden class esélyt kapjon
    progress = True
    while progress:
        progress = False
        for c in sorted(targets.keys()):
            current = class_counts[c]
            target = targets[c]
            if current >= target:
                continue  # kész

            candidates = class2images[c]
            if not candidates:
                continue

            # véletlenül választunk egy képet, ami tartalmazza ezt az osztályt
            base = random.choice(candidates)
            if used_copies[base] >= max_copies_per_image:
                # ezt már elégszer másoltuk
                continue

            img_path = find_image(base)
            lbl_path = os.path.join(args.train_labels, base + ".txt")
            if not img_path or not os.path.exists(lbl_path):
                continue

            # új név
            stem, ext = os.path.splitext(os.path.basename(img_path))
            new_stem = f"{stem}_ov{duplicate_index}"
            new_img = os.path.join(args.train_images, new_stem + ext)
            new_lbl = os.path.join(args.train_labels, new_stem + ".txt")

            copy2(img_path, new_img)
            copy2(lbl_path, new_lbl)

            # frissítjük a számlálókat
            duplicate_index += 1
            used_copies[base] += 1

            # ugyanazok a classok lesznek az új képen, mint az eredetin
            for cc in img2classes[base]:
                class_counts[cc] += 1
                class2images[cc].append(new_stem)
            img2classes[new_stem] = set(img2classes[base])

            progress = True  # történt valami ebben a körben

    print("\n✅ Oversample kész.")
    print("📊 Új class count (train):", dict(sorted(class_counts.items())))

if __name__ == "__main__":
    main()
