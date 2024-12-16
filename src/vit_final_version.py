"""
ViT training on 7K non-cancerous images and 1590 malignant
"""
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time
from torch.optim import AdamW
from torchvision import transforms


def load_and_preprocess_data(root_dir, external_train_dir):
    """
    Load and preprocess the data, including balancing the dataset and merging with external datasets.
    """
    data = pd.read_csv(f'{root_dir}/train-metadata.csv')
    selected_data = data[['isic_id', 'target']]
    data_target_1 = selected_data[selected_data['target'] == 1]

    num_samples_target_0 = 7000
    data_target_0 = selected_data[selected_data['target'] == 0].sample(n=num_samples_target_0, random_state=42)
    balanced_data = pd.concat([data_target_1, data_target_0], axis=0).reset_index(drop=True)

    file_list = os.listdir(external_train_dir)
    isic_2020_dataset = pd.DataFrame({
        'isic_id': ['ISIC2020_' + file for file in file_list],
        'target': 1
    })

    merged_dataset = pd.concat([balanced_data, isic_2020_dataset], ignore_index=True)

    train_val_data, test_data = train_test_split(
        merged_dataset, test_size=0.2, stratify=merged_dataset['target'], random_state=42
    )

    train_data, val_data = train_test_split(
        train_val_data, test_size=0.2, stratify=train_val_data['target'], random_state=42
    )

    return train_data, val_data, test_data


def remove_hair(image):
    """
    Remove hair artifacts from an image using the DullRazor approach.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 12, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(image, thresh, inpaintRadius=1, flags=cv2.INPAINT_TELEA)

    return inpainted


class CassavaDataset(torch.utils.data.Dataset):
    """
    Helper class for DataLoader
    """

    def __init__(self, df, data_path='isic-2024-challenge/train-image/image/',
                 secondary_data_path="isic-2020/data/malignant", mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.secondary_data_path = secondary_data_path
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        if img_name.startswith("ISIC2020_"):
            stripped_name = img_name.replace("ISIC2020_", "")
            img_path = os.path.join(self.secondary_data_path, f"{stripped_name}")
        else:
            img_path = os.path.join(self.data_path, f"{img_name}.jpg")
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class HairRemovalTransform:
    def __call__(self, img):
        img_np = np.array(img)
        img_np = remove_hair(img_np)
        img = Image.fromarray(img_np)
        return img


def create_datasets(train_data, val_data, test_data, data_path, transforms_train, transforms_valid):
    """
    Create dataset objects for training, validation, and testing.
    """
    train_dataset = CassavaDataset(train_data, data_path=data_path, transforms=transforms_train)
    valid_dataset = CassavaDataset(val_data, data_path=data_path, transforms=transforms_valid)
    test_dataset = CassavaDataset(test_data, data_path=data_path, transforms=transforms_valid)
    return train_dataset, valid_dataset, test_dataset


def initialize_dataloaders(train_dataset, valid_dataset, test_dataset, batch_size, num_workers=8):
    """
    Creates DataLoader objects for training, validation, and testing.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader, test_loader

# ViT model initialization
class ViTBase16(nn.Module):
    def __init__(self, n_classes, pretrained=False, model_path=None):
        super(ViTBase16, self).__init__()
        self.model = timm.create_model("vit_large_patch16_224", pretrained=True)

        if pretrained and model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.head = nn.Linear(self.model.head.in_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def train_one_epoch(self, train_loader, criterion, optimizer, device):
        self.model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            accuracy = (output.argmax(dim=1) == target).float().mean()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

            optimizer.step()
            if device.type == "xla" and i % 20 == 0:
                xm.master_print(f"\tBATCH {i + 1}/{len(train_loader)} - LOSS: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        avg_accuracy = epoch_accuracy / len(train_loader)
        return avg_loss, avg_accuracy

    def validate_one_epoch(self, valid_loader, criterion, device):
        self.model.eval()
        valid_loss = 0.0
        valid_accuracy = 0.0

        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                accuracy = (output.argmax(dim=1) == target).float().mean()
                valid_loss += loss.item()
                valid_accuracy += accuracy.item()

        avg_loss = valid_loss / len(valid_loader)
        avg_accuracy = valid_accuracy / len(valid_loader)
        return avg_loss, avg_accuracy


def train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    """
    Train and evaluate the model over multiple epochs.
    """
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)
        train_loss, train_acc = model.train_one_epoch(train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

        valid_loss, valid_acc = model.validate_one_epoch(valid_loader, criterion, device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            print(f"Validation accuracy improved from {best_valid_acc:.4f} to {valid_acc:.4f}. Saving model...")
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), "best_model.pth")

        epoch_time = time.time() - start_time
        print(f"Time for epoch {epoch + 1}: {epoch_time:.2f} seconds")

    return train_losses, valid_losses, train_accuracies, valid_accuracies


def visualize_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs):
    """
    Visualize training and validation metrics.
    """
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o', linestyle='-', linewidth=2, color='#9f86c0')
    plt.plot(epochs, valid_losses, label='Validation Loss', marker='s', linestyle='--', linewidth=2, color='#231942')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o', linestyle='-', linewidth=2, color='#9f86c0')
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy', marker='s', linestyle='--', linewidth=2,
             color='#231942')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('train_validation_metrics.png', dpi=300, bbox_inches='tight')


def perform_cross_validation(train_data, labels, model_class, data_path, transforms_train, transforms_valid, num_epochs,
                             batch_size, k_folds, learning_rate):
    """
    Perform k-fold cross-validation.
    """
    splitter = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.2, random_state=42)

    cv_train_losses = []
    cv_valid_losses = []
    cv_train_accuracies = []
    cv_valid_accuracies = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_data, labels)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")
        train_subset = Subset(
            CassavaDataset(train_data.iloc[train_idx].reset_index(drop=True),
                           data_path=data_path,
                           transforms=transforms_train),
            range(len(train_idx))
        )
        valid_subset = Subset(
            CassavaDataset(train_data.iloc[valid_idx].reset_index(drop=True),
                           data_path=data_path,
                           transforms=transforms_valid),
            range(len(valid_idx))
        )

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=8)

        model = model_class(n_classes=2, pretrained=True)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        fold_train_losses = []
        fold_valid_losses = []
        fold_train_accuracies = []
        fold_valid_accuracies = []

        best_valid_acc = 0.0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            train_loss, train_acc = model.train_one_epoch(train_loader, criterion, optimizer, device)
            fold_train_losses.append(train_loss)
            fold_train_accuracies.append(train_acc)
            print(f"Training: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")

            valid_loss, valid_acc = model.validate_one_epoch(valid_loader, criterion, device)
            fold_valid_losses.append(valid_loss)
            fold_valid_accuracies.append(valid_acc)
            print(f"Validation: Loss = {valid_loss:.4f}, Accuracy = {valid_acc:.4f}")

            if valid_acc > best_valid_acc:
                print(f"Validation accuracy improved from {best_valid_acc:.4f} to {valid_acc:.4f}. Saving model...")
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), f"best_model_fold_{fold + 1}.pth")
        cv_train_losses.append(np.mean(fold_train_losses))
        cv_valid_losses.append(np.mean(fold_valid_losses))
        cv_train_accuracies.append(np.mean(fold_train_accuracies))
        cv_valid_accuracies.append(np.mean(fold_valid_accuracies))

    return cv_train_losses, cv_valid_losses, cv_train_accuracies, cv_valid_accuracies


IMG_SIZE = 224
NUM_WORKERS = 8
# Transformations applied to train images set
transforms_train = transforms.Compose(
    [
        HairRemovalTransform(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Transformations applied to validation images set
transforms_valid = transforms.Compose(
    [
        HairRemovalTransform(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# Example Usage:
# 1. Load and preprocess data
# train_data, val_data, test_data = load_and_preprocess_data('ROOT_DIR', 'EXTERNAL_TRAIN_DIR')

# 2. Create datasets train_dataset, valid_dataset, test_dataset = CassavaDataset(train_data, val_data, train_data,
# data_path='isic-2024-challenge/train-image/image/', transforms_train, transforms_valid)

# 3. Initialize DataLoaders train_loader, valid_loader, test_loader = initialize_dataloaders(train_dataset,
# valid_dataset, test_dataset, batch_size=32, NUM_WORKERS)

# 4. Train and evaluate model model = ViTBase16(n_classes=2, pretrained=True) optimizer = AdamW(model.parameters(),
# lr=1e-4, weight_decay=0.01) criterion = nn.CrossEntropyLoss() device = torch.device("cuda" if
# torch.cuda.is_available() else "cpu") train_losses, valid_losses, train_accuracies, valid_accuracies =
# train_and_evaluate(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=20)

# 5. Visualize metrics
# visualize_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs=20)

# 6. Perform cross-validation cv_train_losses, cv_valid_losses, cv_train_accuracies, cv_valid_accuracies =
# perform_cross_validation(train_data, train_data['target'], ViTBase16, 'DATA_PATH', transforms_train,
# transforms_valid, num_epochs=20, batch_size=32, k_folds=5, learning_rate=1e-4)
