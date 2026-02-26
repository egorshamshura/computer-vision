import os
import shutil
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

base_input = 'svhn'
base_output = 'SVHN'

h5_root = 'digitStruct'
h5_name_key = 'name'
h5_bbox_key = 'bbox'
bbox_fields = ['left', 'top', 'width', 'height', 'label']
dataset_parts = ['train', 'test']
mapping = {'train': 'train', 'test': 'val'}

label_for_zero = 10
class_labels = ['0','1','2','3','4','5','6','7','8','9']

def read_h5_attr(handle, ref):
    if hasattr(ref, '__len__') and len(ref) > 1:
        values = []
        for single_ref in ref:
            r = single_ref[0] if isinstance(single_ref, h5py.Reference) else single_ref
            if isinstance(r, h5py.h5r.Reference):
                values.append(handle[r][0][0])
            else:
                values.append(float(r))
        return np.array(values, dtype=float)
    else:
        r = ref[0] if hasattr(ref, '__getitem__') else ref
        if isinstance(r, h5py.h5r.Reference):
            return np.array([handle[r][0][0]], dtype=float)
        else:
            return np.array([float(r)], dtype=float)

def parse_h5_annotation(file_path):
    records = []
    with h5py.File(file_path, 'r') as h5:
        group = h5[h5_root]
        name_entries = group[h5_name_key]
        bbox_entries = group[h5_bbox_key]
        for idx in range(len(name_entries)):
            name_ref = name_entries[idx][0]
            if isinstance(name_ref, h5py.h5r.Reference):
                image_name = ''.join(chr(c[0]) for c in h5[name_ref][:])
            else:
                image_name = str(name_ref)
            bbox_group = h5[bbox_entries[idx][0]]
            bbox_dict = {}
            for field in bbox_fields:
                if field in bbox_group:
                    bbox_dict[field] = read_h5_attr(h5, bbox_group[field])
                else:
                    bbox_dict[field] = np.array([])
            records.append({'name': image_name, 'boxes': bbox_dict})
    return records

def convert_to_yolo(l, t, w, h, img_w, img_h):
    x_center = (l + w / 2) / img_w
    y_center = (t + h / 2) / img_h
    return x_center, y_center, w / img_w, h / img_h

def write_annotation_file(img_name, img_dir, out_img_dir, out_lbl_dir, boxes):
    src_path = os.path.join(img_dir, img_name)
    if not os.path.exists(src_path):
        return
    shutil.copy2(src_path, os.path.join(out_img_dir, img_name))
    with Image.open(src_path) as pil_img:
        width, height = pil_img.size
    label_path = os.path.join(out_lbl_dir, os.path.splitext(img_name)[0] + '.txt')
    with open(label_path, 'w') as lbl_file:
        num_boxes = len(boxes[bbox_fields[0]])
        for j in range(num_boxes):
            left = boxes[bbox_fields[0]][j]
            top = boxes[bbox_fields[1]][j]
            box_w = boxes[bbox_fields[2]][j]
            box_h = boxes[bbox_fields[3]][j]
            raw_label = int(boxes[bbox_fields[4]][j])
            if raw_label == label_for_zero:
                class_id = 0
            else:
                class_id = raw_label
            xc, yc, nw, nh = convert_to_yolo(left, top, box_w, box_h, width, height)
            lbl_file.write(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

def process_partition(partition):
    source_dir = os.path.join(base_input, partition)
    mat_file = os.path.join(source_dir, 'digitStruct.mat')
    if not os.path.isfile(mat_file):
        print(f"Skipping {partition}: {mat_file} not found")
        return
    target_part = mapping.get(partition, partition)
    image_out = os.path.join(base_output, 'images', target_part)
    label_out = os.path.join(base_output, 'labels', target_part)
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)
    annotations = parse_h5_annotation(mat_file)
    for entry in tqdm(annotations, desc=f'Processing {partition}'):
        write_annotation_file(entry['name'], source_dir, image_out, label_out, entry['boxes'])

def generate_yaml():
    if class_labels is None:
        num_classes = 10
        class_names = [str(i) for i in range(num_classes)]
    else:
        num_classes = len(class_labels)
        class_names = class_labels
    config = {
        'path': os.path.abspath(base_output),
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names
    }
    with open(os.path.join(base_output, 'data.yaml'), 'w') as stream:
        yaml.dump(config, stream, sort_keys=False)

if __name__ == '__main__':
    os.makedirs(base_output, exist_ok=True)
    for part in dataset_parts:
        process_partition(part)
    generate_yaml()
