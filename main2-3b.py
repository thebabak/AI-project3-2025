import os
import torch
import open_clip
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import freeze_support

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Enable CuDNN benchmarking for GPU optimization
torch.backends.cudnn.benchmark = True

# Setup environment
def setup_environment():
    """Initialize device and verify environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device

# Load CLIP Model
def load_clip_model(device, model_name="convnext_base_w", pretrained_weights=None):
    """
    Load CLIP model and preprocessors.
    """
    try:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_weights,
            device=device
        )
        print("\nModel structure:")
        print(model)
        return model, preprocess_train, preprocess_val
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise

# Custom CLIP Fine-Tuner
class CustomCLIPFineTuner(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        layers = list(self.visual_encoder.children())
        for layer in layers[:-2]:
            for param in layer.parameters():
                param.requires_grad = False
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]
        classifier_layers = []
        for i in range(num_layers):
            classifier_layers.append(nn.Linear(output_size, output_size if i < num_layers - 1 else num_classes))
            if i < num_layers - 1:
                classifier_layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)

# Dataset and Preprocessing
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
        item = self.data[idx]
        image = item['image']
        if self.augment:
            image = self.augmentation_transforms(image)
        image = self.transform(image)
        label = self.subcategories.index(item['subCategory'])
        return image, label

# Load Fashion Dataset
def load_fashion_dataset():
    ds = load_dataset("ceyda/fashion-products-small")
    dataset = ds['train']
    subcategories = sorted(list(set(dataset['subCategory'])))
    return dataset, subcategories

# Compute Class Weights
def compute_class_weights(dataset, subcategories):
    counts = Counter(dataset['subCategory'])
    total = sum(counts.values())
    weights = [total / counts[subcat] for subcat in subcategories]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

# Compute Metrics
def compute_metrics(true_labels, predictions, subcategories):
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(true_labels, predictions)
    print("\nMetrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    return precision, recall, f1, conf_matrix

# Evaluate Model
def evaluate_model_with_metrics(model, loader, criterion, subcategories):
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
    precision, recall, f1, conf_matrix = compute_metrics(true_labels, predictions, subcategories)
    return avg_loss, precision, recall, f1, conf_matrix

# Train Model with TensorBoard
def train_model_with_tensorboard(
    model, train_loader, val_loader, optimizer_type, num_epochs=3, lr=1e-4, log_dir="logs/", fold_index=0
):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir=f"{log_dir}{optimizer_type}_fold_{fold_index}/")
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f'{optimizer_type} Fold {fold_index}, Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})
        train_loss /= len(train_loader)
        train_acc = 100 * correct_train / total_train
        val_loss, precision, recall, f1, conf_matrix = evaluate_model_with_metrics(
            model, val_loader, criterion, subcategories
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Precision/Validation", precision, epoch)
        writer.add_scalar("Recall/Validation", recall, epoch)
        writer.add_scalar("F1-Score/Validation", f1, epoch)
        if f1 > best_acc:
            best_acc = f1
            torch.save(model.state_dict(), f'best_clip_model_{optimizer_type}_fold_{fold_index}.pth')
    writer.close()
    return val_loss, precision, recall, f1

# Plot Metrics
def plot_metrics(metrics, k_folds):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(folds, metrics["adam"]["val_loss"], label="Adam", marker="o")
    plt.plot(folds, metrics["adamw"]["val_loss"], label="AdamW", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Folds")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(folds, metrics["adam"]["precision"], label="Adam", marker="o")
    plt.plot(folds, metrics["adamw"]["precision"], label="AdamW", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title("Precision Across Folds")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(folds, metrics["adam"]["recall"], label="Adam", marker="o")
    plt.plot(folds, metrics["adamw"]["recall"], label="AdamW", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title("Recall Across Folds")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(folds, metrics["adam"]["f1"], label="Adam", marker="o")
    plt.plot(folds, metrics["adamw"]["f1"], label="AdamW", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.title("F1-Score Across Folds")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main Workflow
if __name__ == '__main__':
    freeze_support()
    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device,
        model_name="convnext_base_w",
        pretrained_weights="laion2b_s13b_b82k_augreg"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories)
    k_folds = 3
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {"adam": {"val_loss": [], "precision": [], "recall": [], "f1": []},
               "adamw": {"val_loss": [], "precision": [], "recall": [], "f1": []}}
    for optimizer_type in ["adam", "adamw"]:
        for fold_index, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            print(f"\nStarting {optimizer_type.upper()} Fold {fold_index + 1}/{k_folds}...")
            train_dataset = FashionDataset([dataset[int(i)] for i in train_indices], subcategories, augment=True)
            val_dataset = FashionDataset([dataset[int(i)] for i in val_indices], subcategories)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            model = CustomCLIPFineTuner(clip_model, len(subcategories), num_layers=1).to(device)
            val_loss, precision, recall, f1 = train_model_with_tensorboard(
                model, train_loader, val_loader, optimizer_type=optimizer_type, num_epochs=3, lr=1e-4, log_dir="logs/", fold_index=fold_index
            )
            metrics[optimizer_type]["val_loss"].append(val_loss)
            metrics[optimizer_type]["precision"].append(precision)
            metrics[optimizer_type]["recall"].append(recall)
            metrics[optimizer_type]["f1"].append(f1)
    plot_metrics(metrics, k_folds)