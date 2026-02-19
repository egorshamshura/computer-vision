import os
from collections import Counter
import random

from imagedataset import ImageDataset


def extract_color_from_filename(filename):
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    parts = name_without_ext.split('$$')
    if len(parts) >= 4:
        return parts[3]
    return None

def _stratified_split(data_by_class, train_ratio, random_seed):
    random.seed(random_seed)
    train_data = []
    test_data = []
    for cls, items in data_by_class.items():
        random.shuffle(items)
        split_idx = int(len(items) * train_ratio)
        train_data.extend(items[:split_idx])
        test_data.extend(items[split_idx:])
    random.shuffle(train_data)
    random.shuffle(test_data)
    return train_data, test_data


def create_dataset_from_folder_recursive(
    dataset_root,
    transform=lambda x: x,
    min_count=500,
    mode=None,
    train_ratio=0.9,
    random_seed=42
):
    img_data = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.lower().endswith('.jpg'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dataset_root)
                color = extract_color_from_filename(file)
                if color not in [None, 'Unlisted']:
                    img_data.append((rel_path, color))
    counter = Counter(color for _, color in img_data)
    valid_colors = {color for color, cnt in counter.items() if cnt >= min_count}
    filtered_data = [(rel_path, color) for rel_path, color in img_data if color in valid_colors]

    if mode is None:
        selected_data = filtered_data
    else:
        data_by_class = {color: [] for color in valid_colors}
        for rel_path, color in filtered_data:
            data_by_class[color].append((rel_path, color))

        train_data, test_data = _stratified_split(data_by_class, train_ratio, random_seed)

        if mode == 'train':
            selected_data = train_data
        elif mode == 'test':
            selected_data = test_data

    classes = sorted(valid_colors)
    img_names = [rel_path for rel_path, _ in selected_data]
    img_name_to_label = {rel_path.replace('.jpg', ''): color for rel_path, color in selected_data}

    dataset = ImageDataset(
        dataset_root=dataset_root,
        classes=classes,
        img_name_to_label=img_name_to_label,
        img_names=img_names,
        transform=transform
    )
    return dataset