import os
import torch
import open_clip
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
import pandas as pd
import shutil
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import freeze_support

# Environment settings for performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
torch.backends.cudnn.benchmark = True  # Optimize cuDNN kernel selection

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        try:
            torch.cuda.empty_cache()  # Clear cache to avoid memory fragmentation
        except Exception as e:
            print(f"Error with GPU: {e}. Switching to CPU...")
            device = torch.device("cpu")
    return device

def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained_weights, device=device
    )
    print("\nModel structure:")
    print(model)
    return model, preprocess_train, preprocess_val

class FullFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]
        classifier_layers = []
        for i in range(num_layers):
            classifier_layers.append(
                nn.Linear(output_size, output_size if i < num_layers - 1 else num_classes)
            )
            if i < num_layers - 1:
                classifier_layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)

class FashionDataset(Dataset):
    def __init__(self, data, subcategories, transform=None, augment=False):
        self.data = data
        self.subcategories = subcategories
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        self.augment = augment
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.data[idx]
        image = item['image']
        if self.augment:
            image = self.augmentation_transforms(image)
        image = self.transform(image)
        label = self.subcategories.index(item['subCategory'])
        return image, label

def load_fashion_dataset():
    ds = load_dataset("ceyda/fashion-products-small")
    dataset = ds['train']
    subcategories = sorted(list(set(dataset['subCategory'])))
    return dataset, subcategories

def compute_class_weights(dataset, subcategories):
    counts = Counter(dataset['subCategory'])
    total = sum(counts.values())
    weights = [total / counts[subcat] for subcat in subcategories]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

def compute_metrics(true_labels, predictions, subcategories, fold=None, save_dir=None):
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nMetrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    # Save confusion matrix if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"confusion_matrix.png" if fold is None else f"confusion_matrix_fold_{fold}.png"
        plt.savefig(os.path.join(save_dir, fname))
    plt.close()
    return precision, recall, f1, conf_matrix

def evaluate_model_with_metrics(model, loader, criterion, subcategories, device, display_predictions=False, fold=None, save_dir=None):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predictions = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
    avg_loss = total_loss / len(loader)
    precision, recall, f1, conf_matrix = compute_metrics(
        true_labels, predictions, subcategories, fold=fold, save_dir=save_dir
    )
    return avg_loss, precision, recall, f1, conf_matrix

class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, f1_score):
        score = f1_score
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def save_logs_as_zip(log_dir, zip_name):
    try:
        os.makedirs(os.path.dirname(zip_name), exist_ok=True)
        shutil.make_archive(zip_name, 'zip', log_dir)
        print(f"Logs saved as {zip_name}.zip")
    except Exception as e:
        print(f"Error saving logs as zip: {e}")

def save_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

def plot_fold_metrics(metrics, k_folds, save_dir):
    folds = range(1, k_folds + 1)
    # Check all lists have length k_folds
    for key in ["val_loss", "precision", "recall", "f1"]:
        if len(metrics[key]) != k_folds:
            print(f"Skipping plot_fold_metrics: '{key}' has length {len(metrics[key])}, expected {k_folds}.")
            return
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(folds, metrics["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Folds")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(folds, metrics["precision"], label="Precision", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title("Precision Across Folds")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(folds, metrics["recall"], label="Recall", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title("Recall Across Folds")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(folds, metrics["f1"], label="F1-Score", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Across Folds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fold_metrics.png"))
    plt.close()

def plot_loss_accuracy_across_folds(fold_metrics, k_folds, save_dir):
    folds = range(1, k_folds + 1)
    # Check all lists have length k_folds
    for key in ["fold_train_loss", "fold_val_loss", "test_loss", "fold_train_acc", "fold_val_acc", "test_acc"]:
        if len(fold_metrics[key]) != k_folds:
            print(f"Skipping plot_loss_accuracy_across_folds: '{key}' has length {len(fold_metrics[key])}, expected {k_folds}.")
            return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(folds, fold_metrics["fold_train_loss"], label="Train Loss", marker="o")
    plt.plot(folds, fold_metrics["fold_val_loss"], label="Validation Loss", marker="o")
    plt.plot(folds, fold_metrics["test_loss"], label="Test Loss", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Loss")
    plt.title("Loss Across Folds")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(folds, fold_metrics["fold_train_acc"], label="Train Accuracy", marker="o")
    plt.plot(folds, fold_metrics["fold_val_acc"], label="Validation Accuracy", marker="o")
    plt.plot(folds, fold_metrics["test_acc"], label="Test Accuracy", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Across Folds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_accuracy_across_folds.png"))
    plt.close()

def train_full_finetune(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, subcategories, device, log_dir,
    accumulation_steps=4, validate_every=2
):
    writer = SummaryWriter(log_dir)
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=3, delta=0.001)
    best_f1 = 0.0
    metrics_dict = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': []
    }
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_loss /= len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase (less frequent to save time)
        if (epoch + 1) % validate_every == 0:
            val_loss, precision, recall, f1, conf_matrix = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, display_predictions=False,
                fold=None, save_dir=log_dir
            )
            metrics_dict['train_loss'].append(train_loss)
            metrics_dict['val_loss'].append(val_loss)
            metrics_dict['train_acc'].append(train_acc)
            val_acc = np.trace(conf_matrix) / np.sum(conf_matrix) * 100
            metrics_dict['val_acc'].append(val_acc)
            metrics_dict['precision'].append(precision)
            metrics_dict['recall'].append(recall)
            metrics_dict['f1'].append(f1)
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Precision/Validation", precision, epoch)
            writer.add_scalar("Recall/Validation", recall, epoch)
            writer.add_scalar("F1-Score/Validation", f1, epoch)
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_full_finetune_clip.pth'))
            early_stopping(f1)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        else:
            metrics_dict['train_loss'].append(train_loss)
            metrics_dict['val_loss'].append(metrics_dict['val_loss'][-1] if metrics_dict['val_loss'] else 0)
            metrics_dict['train_acc'].append(train_acc)
            metrics_dict['val_acc'].append(metrics_dict['val_acc'][-1] if metrics_dict['val_acc'] else 0)
            metrics_dict['precision'].append(metrics_dict['precision'][-1] if metrics_dict['precision'] else 0)
            metrics_dict['recall'].append(metrics_dict['recall'][-1] if metrics_dict['recall'] else 0)
            metrics_dict['f1'].append(metrics_dict['f1'][-1] if metrics_dict['f1'] else 0)
    writer.close()

    # Plot and save per-fold loss and accuracy curves
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_dict['train_loss'], label="Train Loss")
    plt.plot(metrics_dict['val_loss'], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(metrics_dict['train_acc'], label="Train Accuracy")
    plt.plot(metrics_dict['val_acc'], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig(os.path.join(log_dir, "accuracy_curve.png"))
    plt.close()

    return metrics_dict

if __name__ == '__main__':
    freeze_support()
    logs_base_dir = "logs/3-1b"
    os.makedirs(logs_base_dir, exist_ok=True)

    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device, model_name="ViT-B-32", pretrained_weights="openai"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    # K-Fold Cross-Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_metrics = {
        "val_loss": [], "precision": [], "recall": [], "f1": [],
        "test_loss": [], "test_acc": [],
        "fold_train_loss": [], "fold_val_loss": [],
        "fold_train_acc": [], "fold_val_acc": []
    }

    num_workers = min(8, os.cpu_count())
    print(f"Using {num_workers} workers for data loading.")

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset = FashionDataset(val_subset, subcategories)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Model: always fine-tune all layers
        model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)

        fold_log_dir = os.path.join(logs_base_dir, f"kfold_finetune/fold_{fold+1}/")
        os.makedirs(fold_log_dir, exist_ok=True)

        metrics_dict = train_full_finetune(
            model, train_loader, val_loader, criterion, optimizer, num_epochs=4,
            subcategories=subcategories, device=device, log_dir=fold_log_dir,
            accumulation_steps=4, validate_every=2
        )

        # Load best model for final validation and save confusion matrix
        model.load_state_dict(torch.load(os.path.join(fold_log_dir, 'best_full_finetune_clip.pth')))
        val_loss, precision, recall, f1, conf_matrix = evaluate_model_with_metrics(
            model, val_loader, criterion, subcategories, device, display_predictions=False,
            fold=fold + 1, save_dir=fold_log_dir
        )
        val_acc = np.trace(conf_matrix) / np.sum(conf_matrix) * 100

        # For test, here we use the val set as test set (unless you have a separate test set)
        test_loss, test_precision, test_recall, test_f1, test_conf_matrix = val_loss, precision, recall, f1, conf_matrix
        test_acc = val_acc

        fold_metrics["val_loss"].append(val_loss)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        fold_metrics["f1"].append(f1)
        fold_metrics["test_loss"].append(test_loss)
        fold_metrics["test_acc"].append(test_acc)
        fold_metrics["fold_train_loss"].append(metrics_dict['train_loss'][-1] if metrics_dict['train_loss'] else 0)
        fold_metrics["fold_val_loss"].append(metrics_dict['val_loss'][-1] if metrics_dict['val_loss'] else 0)
        fold_metrics["fold_train_acc"].append(metrics_dict['train_acc'][-1] if metrics_dict['train_acc'] else 0)
        fold_metrics["fold_val_acc"].append(metrics_dict['val_acc'][-1] if metrics_dict['val_acc'] else 0)

        # Save per-fold metrics as CSV
        save_metrics_to_csv({
            "epoch": list(range(1, len(metrics_dict['train_loss']) + 1)),
            "train_loss": metrics_dict['train_loss'],
            "val_loss": metrics_dict['val_loss'],
            "train_acc": metrics_dict['train_acc'],
            "val_acc": metrics_dict['val_acc'],
            "precision": metrics_dict['precision'],
            "recall": metrics_dict['recall'],
            "f1": metrics_dict['f1'],
        }, os.path.join(fold_log_dir, f"metrics_fold_{fold + 1}.csv"))

        save_logs_as_zip(fold_log_dir, os.path.join(logs_base_dir, f"kfold_finetune/fold_{fold + 1}_logs"))
        torch.cuda.empty_cache()  # Clear cache after each fold to manage memory

        # Print metric lengths for debugging before plotting
    print("val_loss:", len(fold_metrics["val_loss"]))
    print("precision:", len(fold_metrics["precision"]))
    print("recall:", len(fold_metrics["recall"]))
    print("f1:", len(fold_metrics["f1"]))
    print("test_loss:", len(fold_metrics["test_loss"]))
    print("test_acc:", len(fold_metrics["test_acc"]))
    print("fold_train_loss:", len(fold_metrics["fold_train_loss"]))
    print("fold_val_loss:", len(fold_metrics["fold_val_loss"]))
    print("fold_train_acc:", len(fold_metrics["fold_train_acc"]))
    print("fold_val_acc:", len(fold_metrics["fold_val_acc"]))

    # Plot and save overall fold metrics and loss/accuracy graphs
    plot_fold_metrics(fold_metrics, k_folds, logs_base_dir)
    plot_loss_accuracy_across_folds(fold_metrics, k_folds, logs_base_dir)
    save_metrics_to_csv(fold_metrics, os.path.join(logs_base_dir, "kfold_full_finetune_metrics.csv"))
    print("\nK-Fold fine-tuning complete. Fold results and graphs saved for analysis.")