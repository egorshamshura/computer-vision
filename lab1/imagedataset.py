from collections import Counter
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class ImageDataset(Dataset):
    def __init__(self, dataset_root, classes, img_name_to_label,
                 img_names, transform):
        self.dataset_root = dataset_root
        self.img_names = img_names
        self.img_name_to_label = img_name_to_label
        self.label_to_idx = {val: idx for idx, val in enumerate(classes)}
        self.transform = transform
        self.color_counter = Counter(img_name_to_label.values())

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.dataset_root, img_name)
        key = img_name.replace('.jpg', '')
        label = self.img_name_to_label[key]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)
        image.close()
        return image_tensor, self.label_to_idx[label]
