import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = 128
Z_DIM = 128
NUM_CLASSES = 2
LR = 0.0002
BETAS = (0.5, 0.999)
N_CRITIC = 5
LAMBDA_GP = 10
EPOCHS = 10
SAMPLE_INTERVAL = 500
OUTPUT_DIR = "output_conditional"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_ROOT = './celeba'
IMAGE_DIR = os.path.join(DATA_ROOT, 'img_align_celeba', 'img_align_celeba')
ATTR_CSV = os.path.join(DATA_ROOT, 'list_attr_celeba.csv')а
PARTITION_CSV = os.path.join(DATA_ROOT, 'list_eval_partition.csv')

class CelebADataset(Dataset):
    """Возвращает (изображение, метка пола) для train/val/test."""
    def __init__(self, image_dir, attr_csv, partition_csv, split='train', transform=None):
        self.image_dir = image_dir
        self.transform = transform

        attrs = pd.read_csv(attr_csv)
        partition = pd.read_csv(partition_csv)
        df = attrs.merge(partition, on='image_id')

        split_map = {'train': 0, 'val': 1, 'test': 2}
        self.df = df[df['partition'] == split_map[split]].reset_index(drop=True)
        self.df['Gender'] = (self.df['Male'] == 1).astype(int)   # 1 – мужчина, 0 – женщина

        print(f"Загружено {len(self.df)} изображений для split '{split}'")

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

def get_dataloader(split='train'):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = CelebADataset(IMAGE_DIR, ATTR_CSV, PARTITION_CSV, split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split=='train'),
                        num_workers=4, pin_memory=True, drop_last=(split=='train'))
    return loader

class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.fc = nn.Linear(z_dim + embed_dim, 512 * 8 * 8)
        self.main = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, labels):
        label_emb = self.embed(labels)
        z = torch.cat([z, label_emb], dim=1)
        out = self.fc(z)
        out = out.view(z.size(0), 512, 8, 8)
        out = self.main(out)
        return out

class Critic(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(1024 * 4 * 4, 1)

    def forward(self, img, labels):
        label_map = labels.view(-1, 1, 1, 1).expand(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([img, label_map], dim=1)
        features = self.main(x)
        features = features.view(features.size(0), -1)
        out = self.fc(features)
        return out

def compute_gradient_penalty(critic, real_imgs, fake_imgs, labels):
    batch_size = real_imgs.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=DEVICE)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated.requires_grad_(True)

    critic_out = critic(interpolated, labels)
    grad = torch.autograd.grad(
        outputs=critic_out,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_out),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(batch_size, -1)
    grad_norm = grad.norm(2, dim=1)
    gp = ((grad_norm - 1) ** 2).mean()
    return gp

def main():
    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val')

    netG = Generator(Z_DIM, NUM_CLASSES).to(DEVICE)
    netD = Critic(NUM_CLASSES).to(DEVICE)

    if DEVICE.type == 'cuda' and torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=BETAS)
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=BETAS)

    fixed_z = torch.randn(64, Z_DIM, device=DEVICE)
    fixed_labels = torch.tensor([i % 2 for i in range(64)], device=DEVICE)

    fid = FrechetInceptionDistance(feature=64, normalize=True).to(DEVICE)
    is_metric = InceptionScore(normalize=True).to(DEVICE)

    real_images_for_fid = []
    for i, (imgs, _) in enumerate(val_loader):
        if i >= 10:
            break
        real_images_for_fid.append(imgs)
    real_images_for_fid = torch.cat(real_images_for_fid)[:1000].to(DEVICE)

    epoch_losses_d = []
    epoch_losses_g = []
    epoch_fid_scores = []
    epoch_is_means = []
    epoch_is_stds = []

    step = 0
    for epoch in range(EPOCHS):
        running_loss_d = 0.0
        running_loss_g = 0.0
        n_batches = 0

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            real_imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = real_imgs.size(0)

            for _ in range(N_CRITIC):
                z = torch.randn(batch_size, Z_DIM, device=DEVICE)
                fake_imgs = netG(z, labels).detach()

                d_real = netD(real_imgs, labels)
                d_fake = netD(fake_imgs, labels)
                lossD = -(d_real.mean() - d_fake.mean())
                gp = compute_gradient_penalty(netD, real_imgs, fake_imgs, labels)
                lossD += LAMBDA_GP * gp

                optimizerD.zero_grad()
                lossD.backward()
                optimizerD.step()

            z = torch.randn(batch_size, Z_DIM, device=DEVICE)
            fake_imgs = netG(z, labels)
            g_loss = -netD(fake_imgs, labels).mean()

            optimizerG.zero_grad()
            g_loss.backward()
            optimizerG.step()

            running_loss_d += lossD.item()
            running_loss_g += g_loss.item()
            n_batches += 1

            if step % 100 == 0:
                print(f"[Epoch {epoch+1}/{EPOCHS}] [Batch {i}/{len(train_loader)}] "
                      f"Loss D: {lossD.item():.4f}, Loss G: {g_loss.item():.4f}")

            if step % SAMPLE_INTERVAL == 0:
                with torch.no_grad():
                    fake = netG(fixed_z, fixed_labels)
                    save_image(fake, os.path.join(OUTPUT_DIR, f"epoch_{epoch}_step_{step}.png"),
                               nrow=8, normalize=True, value_range=(-1,1))
            step += 1

        avg_loss_d = running_loss_d / n_batches
        avg_loss_g = running_loss_g / n_batches
        epoch_losses_d.append(avg_loss_d)
        epoch_losses_g.append(avg_loss_g)

        netG.eval()
        torch.cuda.empty_cache()

        # Сброс метрик
        fid.reset()
        is_metric.reset()

        chunk_size = 8

        with torch.no_grad():
            real_norm = (real_images_for_fid + 1) / 2
            for i in range(0, real_norm.size(0), chunk_size):
                chunk = real_norm[i:i+chunk_size]
                fid.update(chunk, real=True)

            num_fake_samples = 5000
            fake_labels = torch.randint(0, NUM_CLASSES, (num_fake_samples,), device=DEVICE)

            for j in range(0, num_fake_samples, BATCH_SIZE):
                current_batch = min(BATCH_SIZE, num_fake_samples - j)
                z = torch.randn(current_batch, Z_DIM, device=DEVICE)
                lbl = fake_labels[j:j+current_batch]
                fake = netG(z, lbl)
                fake_norm = (fake + 1) / 2

                for k in range(0, current_batch, chunk_size):
                    chunk = fake_norm[k:k+chunk_size]
                    fid.update(chunk, real=False)

                for k in range(0, current_batch, chunk_size):
                    chunk = fake_norm[k:k+chunk_size]
                    is_metric.update(chunk)

            fid_score = fid.compute().item()
            print(f"Epoch {epoch+1} FID: {fid_score:.4f}")
            epoch_fid_scores.append(fid_score)

            is_mean, is_std = is_metric.compute()
            print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
            epoch_is_means.append(is_mean.item())
            epoch_is_stds.append(is_std.item())

        torch.cuda.empty_cache()
        netG.train()

        torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, f"netG_epoch{epoch+1}.pth"))
        torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, f"netD_epoch{epoch+1}.pth"))

    print("Обучение завершено!")

    metrics_df = pd.DataFrame({
        'epoch': list(range(1, EPOCHS+1)),
        'loss_d': epoch_losses_d,
        'loss_g': epoch_losses_g,
        'fid': epoch_fid_scores,
        'is_mean': epoch_is_means,
        'is_std': epoch_is_stds
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'training_metrics.csv'), index=False)
    print(f"Метрики сохранены в {os.path.join(OUTPUT_DIR, 'training_metrics.csv')}")

    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(metrics_df['epoch'], metrics_df['loss_d'], label='Loss D', marker='o')
        ax1.plot(metrics_df['epoch'], metrics_df['loss_g'], label='Loss G', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Потери дискриминатора и генератора')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(metrics_df['epoch'], metrics_df['fid'], label='FID', marker='o', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('FID', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        ax3 = ax2.twinx()
        ax3.errorbar(metrics_df['epoch'], metrics_df['is_mean'], yerr=metrics_df['is_std'],
                     label='IS', marker='s', color='blue', capsize=3)
        ax3.set_ylabel('Inception Score', color='blue')
        ax3.tick_params(axis='y', labelcolor='blue')

        ax2.set_title('FID и Inception Score')
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150)
        plt.show()
        print(f"Графики сохранены в {os.path.join(OUTPUT_DIR, 'learning_curves.png')}")
    except Exception as e:
        print(f"Не удалось построить графики: {e}")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    main()
