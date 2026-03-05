import json
import os
import cv2
import numpy as np
import math
from sklearn.model_selection import train_test_split
import shutil

dataset_root = "sign_dataset/train"
json_path = os.path.join(dataset_root, "via_region_data.json")
output_dir = "yolo_dataset"
val_split = 0.2
random_seed = 42

class_field = "name"

def shape_to_polygon(shape_attrs):
    shape_name = shape_attrs.get('name', '')
    if shape_name == 'polygon':
        xs = shape_attrs.get('all_points_x', [])
        ys = shape_attrs.get('all_points_y', [])
        if len(xs) < 3 or len(ys) < 3:
            return None
        return list(zip(xs, ys))

    elif shape_name == 'rect':
        x = shape_attrs.get('x')
        y = shape_attrs.get('y')
        w = shape_attrs.get('width')
        h = shape_attrs.get('height')
        if None in (x, y, w, h):
            return None
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    elif shape_name == 'circle':
        cx = shape_attrs.get('cx')
        cy = shape_attrs.get('cy')
        r = shape_attrs.get('r')
        if None in (cx, cy, r):
            return None
        num_points = 24
        points = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            points.append((x, y))
        return points
    return None

def normalize_class_name(name):
    if not name:
        return "road_sign"
    name = name.strip().lower()
    name = name.replace(' ', '_')
    return name

with open(json_path, 'r') as f:
    data = json.load(f)

class_mapping = {}
next_id = 0
samples = []

for filename, info in data.items():
    if 'regions' not in info or not info['regions']:
        continue

    img_path = os.path.join(dataset_root, info['filename'])
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    polygons = []
    class_ids = []
    for region in info['regions'].values():
        shape_attrs = region.get('shape_attributes', {})
        region_attrs = region.get('region_attributes', {})

        abs_poly = shape_to_polygon(shape_attrs)
        if abs_poly is None or len(abs_poly) < 3:
            continue

        raw_class = region_attrs.get(class_field, '')
        class_name = normalize_class_name(raw_class)

        if class_name not in class_mapping:
            class_mapping[class_name] = next_id
            next_id += 1
        class_id = class_mapping[class_name]

        pts = [(x / w, y / h) for x, y in abs_poly]
        pts_flat = np.array(pts, dtype=np.float32).flatten()
        polygons.append(pts_flat)
        class_ids.append(class_id)

    if polygons:
        samples.append({
            'img_path': img_path,
            'h': h, 'w': w,
            'polygons': polygons,
            'class_ids': class_ids
        })


train_samples, val_samples = train_test_split(samples, test_size=val_split, random_state=random_seed)

def write_labels(samples, subset):
    subset_dir = os.path.join(output_dir, subset)
    os.makedirs(os.path.join(subset_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(subset_dir, 'labels'), exist_ok=True)
    for sample in samples:
        img_path = sample['img_path']
        img_dst = os.path.join(subset_dir, 'images', os.path.basename(img_path))
        if not os.path.exists(img_dst):
            shutil.copy(img_path, img_dst)

        label_path = os.path.join(subset_dir, 'labels',
                                  os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        with open(label_path, 'w') as f:
            for class_id, poly in zip(sample['class_ids'], sample['polygons']):
                line = f"{class_id} " + " ".join(f"{x:.6f}" for x in poly)
                f.write(line + '\n')

write_labels(train_samples, 'train')
write_labels(val_samples, 'val')

sorted_classes = sorted(class_mapping.items(), key=lambda x: x[1])
with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
    for name, idx in sorted_classes:
        f.write(f"{name}\n")

print(f"saved in{output_dir}")
print("saved in classes.txt:")
with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
    print(f.read())