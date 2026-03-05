import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

def polygon_to_mask(polygon, img_shape):
    h, w = img_shape[:2]
    pts = np.array(polygon, dtype=np.float32)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def read_yolo_seg_label(txt_path):
    objects = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
            objects.append((class_id, polygon))
    return objects

if __name__ == "__main__":
    model_path = r'C:\Users\shama\CV\hw3\runs\segment\road_signs4\weights\best.pt'
    dataset_yaml = 'yolo_dataset/dataset.yaml'

    model = YOLO(model_path)

    with open(dataset_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = os.path.dirname(dataset_yaml)
    val_img_dir = os.path.join(yaml_dir, data_cfg.get('val', 'val/images'))
    val_label_dir = os.path.join(yaml_dir, 'val', 'labels')

    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(val_img_dir) if f.lower().endswith(img_extensions)]

    iou_thresholds = [0.5, 0.75, 0.9]
    counts = {th: 0 for th in iou_thresholds}
    total_images = 0

    for img_file in img_files:
        total_images += 1
        img_path = os.path.join(val_img_dir, img_file)
        label_path = os.path.join(val_label_dir, os.path.splitext(img_file)[0] + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(img, conf=0.25, iou=0.45, verbose=False)
        result = results[0]

        if not os.path.exists(label_path):
            continue
        gt_objects = read_yolo_seg_label(label_path)

        orig_h, orig_w = result.orig_shape
        gt_masks = []
        for _, polygon in gt_objects:
            mask = polygon_to_mask(polygon, (orig_h, orig_w))
            gt_masks.append(mask)

        if result.masks is None:
            pred_masks = []
        else:
            pred_masks = result.masks.data.cpu().numpy()

        best_iou = 0.0
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            for pred_mask in pred_masks:
                pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                pred_bin = pred_mask_resized > 0.5
                for gt_mask in gt_masks:
                    gt_bin = gt_mask > 0
                    intersection = np.logical_and(pred_bin, gt_bin).sum()
                    union = np.logical_or(pred_bin, gt_bin).sum()
                    iou = intersection / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou

        for th in iou_thresholds:
            if best_iou >= th:
                counts[th] += 1

        if total_images % 100 == 0:
            print(f"{total_images}")

    print(f"{total_images}\n")
    for th in iou_thresholds:
        percent = counts[th] / total_images * 100 if total_images > 0 else 0
        print(f"IoU >= {th}: {percent:.2f}%")