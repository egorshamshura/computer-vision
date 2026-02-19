import torch
from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageEnhance
import math
import os
import time
from torch import nn, optim
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


from CustomResNet18 import CustomResNet18
from dataset import create_dataset_from_folder_recursive

normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_sz = 150


test_data_transforms = T.Compose(
    [
        T.Resize((img_sz, img_sz)),
        T.ToTensor(),
        normalize,
    ]
)

train_data_transforms = T.Compose(
    [
        T.RandomResizedCrop(img_sz, scale=(0.8, 1.2), ratio=(0.9, 1.2)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.01),
        T.RandomAutocontrast(p=0.4),
        T.RandomRotation(degrees=20),
        T.ToTensor(),
        normalize,
    ]
)
from torch.utils.data import DataLoader

from sklearn import metrics

def run_epoch(phase, dataloader):
  if phase == 'train':
      model.train()
  else:
      model.eval()

  running_loss = 0.0
  running_corrects = 0
  y_test = []
  y_pred = []
  all_elems_count = 0
  cur_tqdm = tqdm(dataloader)
  for inputs, labels in cur_tqdm:
      bz = inputs.shape[0]
      all_elems_count += bz

      inputs = inputs.to(device, non_blocking=True)
      labels = labels.to(device, non_blocking=True)
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      if phase == 'train':
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      _, preds = torch.max(outputs, 1)
      y_test.extend(labels.detach().cpu().numpy())
      y_pred.extend(preds.detach().cpu().numpy())
      running_loss += loss.item() * bz
      corrects_cnt = torch.sum(preds == labels.detach())
      running_corrects += corrects_cnt
      show_dict = {'Loss': f'{loss.item():.6f}',
                    'Corrects': f'{corrects_cnt.item()}/{bz}',
                    'Accuracy': f'{(corrects_cnt * 100 / bz).item():.3f}%'}
      cur_tqdm.set_postfix(show_dict)

  conf_matrix = metrics.confusion_matrix(y_test, y_pred)

  print("Calculating metrics...")
  f05_macro = metrics.fbeta_score(y_test, y_pred, average="macro", beta=0.5)
  f1_macro = metrics.f1_score(y_test, y_pred, average="macro")
  epoch_loss = running_loss / all_elems_count
  epoch_acc = running_corrects.float().item() / all_elems_count
  return epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix

def test_epoch(dataloader):
    with torch.inference_mode():
      return run_epoch('test', dataloader)

def train_epoch(dataloader):
    return run_epoch('train', dataloader)

def train_model(dataloaders, num_epochs=5):
  print(f"Training model with params:")
  print(f"Optim: {optimizer}")
  print(f"Criterion: {criterion}")

  phases = ['train', 'test']
  for phase in dataloaders:
      if phase not in phases:
          phases.append(phase)

  saved_epoch_losses = {phase: [] for phase in phases}
  saved_epoch_accuracies = {phase: [] for phase in phases}
  saved_epoch_f1_macros = {phase: [] for phase in phases}

  for epoch in range(1, num_epochs + 1):
      start_time = time.time()

      print("=" * 100)
      print(f'Epoch {epoch}/{num_epochs}')
      print('-' * 10)

      for phase in phases:
          print("--- Cur phase:", phase)
          epoch_loss, epoch_acc, f05_macro, f1_macro, conf_matrix = \
              train_epoch(dataloaders[phase]) if phase == 'train' \
                  else test_epoch(dataloaders[phase])
          saved_epoch_losses[phase].append(epoch_loss)
          saved_epoch_accuracies[phase].append(epoch_acc)
          saved_epoch_f1_macros[phase].append(f1_macro)
          print(f'{phase} loss: {epoch_loss:.6f}, '
                f'acc: {epoch_acc:.6f}, '
                f'f05_macro: {f05_macro:.6f}, '
                f'f1_macro: {f1_macro:.6f}')
          print("Confusion matrix:")
          print(conf_matrix)

      model.eval()
      if epoch > 1:
        plt.title(f'Losses during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_losses['train'], label='Train Loss')
        plt.plot(range(1, epoch + 1), saved_epoch_losses['test'], label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel(criterion.__class__.__name__)
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/loss_graph_epoch{epoch + 1}.png')
        plt.close('all')

        plt.title(f'Accuracies during training. Epoch {epoch}/{num_epochs}.')
        plt.plot(range(1, epoch + 1), saved_epoch_accuracies['train'], label='Train Acc')
        plt.plot(range(1, epoch + 1), saved_epoch_accuracies['test'], label='Test Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.savefig(f'{log_folder}/acc_graph_epoch{epoch + 1}.png')
        plt.close('all')

      end_time = time.time()
      epoch_time = end_time - start_time
      print("-" * 10)
      print(f"Epoch Time: {math.floor(epoch_time // 60)}:{math.floor(epoch_time % 60):02d}")

  print("*** Training Completed ***")

  return saved_epoch_losses, saved_epoch_accuracies, saved_epoch_f1_macros

def denormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tensor = tensor.clone().detach()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_pic(dataset_dirpath):
    example_img_path = None
    for root, dirs, files in os.walk(dataset_dirpath):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                example_img_path = os.path.join(root, file)
                break
        if example_img_path:
            break

    if example_img_path:
        img_pil = Image.open(example_img_path).convert('RGB')
        orig_resized = T.Resize((img_sz, img_sz))(img_pil)

        num_samples = 8
        transformed_tensors = [train_data_transforms(img_pil) for _ in range(num_samples)]

        transformed_imgs = [denormalize(t).permute(1, 2, 0).numpy() for t in transformed_tensors]

        fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 4))
        axes[0].imshow(orig_resized)
        axes[0].set_title('Original')
        axes[0].axis('off')
        for i, img in enumerate(transformed_imgs):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f'Augmented {i+1}')
            axes[i + 1].axis('off')
        plt.tight_layout()
        plt.savefig(f'{log_folder}/example_augmentations.png')
        plt.show()
    

if __name__ == '__main__':
    log_folder = 'logs'
    os.makedirs(log_folder, exist_ok=True)
    dataset_dirpath = 'confirmed_fronts'

    train_dataset = create_dataset_from_folder_recursive(
        dataset_dirpath, mode='train', train_ratio=0.9, transform=train_data_transforms
    )
    test_dataset = create_dataset_from_folder_recursive(
        dataset_dirpath, mode='test', train_ratio=0.9, transform=test_data_transforms
    )

    batch_size = 32
    num_workers = 4

    show_pic(dataset_dirpath)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(test_dataset.color_counter)

    model = CustomResNet18(num_classes=len(test_dataset.color_counter)).to(device)

    # model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # in_features = model.classifier[-1].in_features
    # model.classifier[-1] = nn.Linear(in_features, num_classes)

    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.features[-5:].parameters():
    #     param.requires_grad = True

    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    # model = model.to(device)

    # model = models.mobilenet_v2(weights='IMAGENET1K_V1')

    # in_features = model.classifier[1].in_features

    # model.classifier[1] = nn.Linear(in_features, num_classes)

    # for param in model.parameters():
    #     param.requires_grad = False

    # for param in model.features[-5:].parameters():
    #     param.requires_grad = True

    # for param in model.classifier.parameters():
    #     param.requires_grad = True

    # model = model.to(device)

    num_epochs = 10
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(dataloaders, num_epochs)