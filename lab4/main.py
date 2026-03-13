import os
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

DATA_ROOT = './celeba'
ZIP_PATH = 'C:\\Users\\shama\\CV\\hw4\\archive.zip'
ATTR_CSV = './celeba/list_attr_celeba.csv'
PARTITION_CSV = './celeba/list_eval_partition.csv'

def extract_images(zip_path, extract_to):
    image_dir = os.path.join(extract_to, 'img_align_celeba', 'img_align_celeba')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return image_dir

def load_annotations(attr_csv, partition_csv):
    attrs = pd.read_csv(attr_csv)
    partition = pd.read_csv(partition_csv)
    df = attrs.merge(partition, on='image_id')
    return df

def prepare_gender_label(df):
    df['Gender'] = (df['Male'] == 1).astype(int)
    return df

class CelebADataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        label = self.df.iloc[idx]['Gender']

        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def prepare_dataloaders(batch_size=128, img_size=64, num_workers=4):
    image_dir = extract_images(ZIP_PATH, DATA_ROOT)

    df = load_annotations(ATTR_CSV, PARTITION_CSV)
    df = prepare_gender_label(df)

    train_df = df[df['partition'] == 0].reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = CelebADataset(train_df, image_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_loader

if __name__ == "__main__":
    loader = prepare_dataloaders(batch_size=64, img_size=128)
    for images, labels in loader:
        print(f"Пакет изображений: {images.shape}, метки: {labels.shape}")
        break
    