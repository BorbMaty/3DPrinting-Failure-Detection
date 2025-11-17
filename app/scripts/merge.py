import os
import shutil
from glob import glob

# Paths
root_dir = '/home/lucy/Desktop/datasetmaybefinal'
output_images = os.path.join(root_dir, 'merged/images/train')
output_labels = os.path.join(root_dir, 'merged/labels/train')

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

total_images = 0
total_labels = 0

for i in range(1, 14):
    dataset_path = os.path.join(root_dir, f'final{i}')
    image_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')

    if not os.path.exists(image_dir):
        print(f"⚠️  Skipping dataset{i}: no 'images' folder")
        continue
    if not os.path.exists(label_dir):
        print(f"⚠️  Skipping dataset{i}: no 'labels' folder")
        continue

    image_files = glob(os.path.join(image_dir, '*'))
    label_files = glob(os.path.join(label_dir, '*'))

    print(f"📦 dataset{i}: Found {len(image_files)} images, {len(label_files)} labels")

    for image_path in image_files:
        base_name = os.path.basename(image_path)
        new_name = f'ds{i}_{base_name}'
        shutil.copy(image_path, os.path.join(output_images, new_name))
        total_images += 1

    for label_path in label_files:
        base_name = os.path.basename(label_path)
        new_name = f'ds{i}_{base_name}'
        shutil.copy(label_path, os.path.join(output_labels, new_name))
        total_labels += 1

print(f"\n✅ Merge complete! Total images: {total_images}, Total labels: {total_labels}")