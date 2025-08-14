import os
import torch
import open_clip
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Environment setup
def setup_environment():
    """Configure hardware settings for optimal performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


device = setup_environment()


# Load CLIP model with caching
def load_clip_model(device):
    """Load CLIP model with optimized settings"""
    try:
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name="convnext_base_w",
            pretrained="laion2b_s13b_b82k_augreg",
            device=device,
            jit=False
        )
        return model, preprocess_val
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise


clip_model, preprocess_val = load_clip_model(device)


def compute_class_weights(dataset, subcategories):
    """Compute balanced class weights to handle imbalanced datasets"""
    # Convert dataset to list to avoid index issues
    dataset_list = list(dataset)
    label_counts = Counter(item['subCategory'] for item in dataset_list)
    counts = torch.tensor([label_counts[cat] for cat in subcategories], dtype=torch.float32)
    weights = 1.0 / counts
    weights = weights / weights.sum()  # Normalize
    return weights.to(device)


# Optimized model architecture
class EfficientCLIPFineTuner(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual

        with torch.no_grad():
            for param in self.visual_encoder.parameters():
                param.requires_grad = False

            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            output_size = self.visual_encoder(dummy_input).shape[1]

        layers = []
        hidden_size = min(512, output_size)

        for i in range(num_layers):
            in_dim = output_size if i == 0 else hidden_size
            out_dim = num_classes if i == num_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.extend([
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])

        self.classifier = nn.Sequential(*layers)
        self.classifier = self.classifier.to(memory_format=torch.channels_last)

    def forward(self, x):
        with torch.no_grad():
            x = self.visual_encoder(x)
        return self.classifier(x)


class CachedFashionDataset(Dataset):
    def __init__(self, data, subcategories, transform=None, augment=False):
        # Convert data to list to avoid index issues
        self.data = list(data)
        self.subcategories = subcategories
        self.transform = transform or preprocess_val
        self.augment = augment
        self.cache = {}
        self.augmenter = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.RandomAffine(15, translate=(0.1, 0.1)),
            transforms.TrivialAugmentWide()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]

        item = self.data[idx]
        image = item['image']

        if self.augment:
            image = self.augmenter(image)

        image = self.transform(image)
        label = self.subcategories.index(item['subCategory'])

        self.cache[idx] = (image, label)
        return image, label


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0

    for i, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / 2

        scaler.scale(loss).backward()

        if (i + 1) % 2 == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    metrics = {
        'accuracy': accuracy,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }

    return metrics


def train_and_evaluate(model, train_loader, val_loader, num_epochs=3, lr=1e-4):
    # Get the actual dataset from the loader's dataset
    train_dataset = train_loader.dataset
    class_weights = compute_class_weights(train_dataset.data, train_dataset.subcategories)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Handle GradScaler initialization based on PyTorch version
    if hasattr(torch.amp, 'GradScaler'):
        # PyTorch 2.1+ (new API)
        scaler = torch.amp.GradScaler(device_type='cuda')
    else:
        # Older PyTorch versions (legacy API)
        scaler = torch.cuda.amp.GradScaler()

    best_metrics = {'accuracy': 0}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = evaluate(model, val_loader)
        print(f"Val Accuracy: {val_metrics['accuracy']:.2f}%")

        if val_metrics['accuracy'] > best_metrics['accuracy']:
            best_metrics = val_metrics
            torch.save(model.state_dict(), f"best_model_{len(model.classifier) // 3}_layers.pt")

    return best_metrics


def compare_models(dataset, subcategories, num_layers_list=[1, 2, 3], k_folds=3):
    results = {}

    # Convert dataset to list to avoid index issues
    dataset_list = list(dataset)

    for num_layers in num_layers_list:
        fold_metrics = []
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset_list)):
            print(f"\nTraining {num_layers}-layer model (Fold {fold + 1}/{k_folds})")

            # Convert indices to Python integers
            train_idx = [int(i) for i in train_idx]
            val_idx = [int(i) for i in val_idx]

            train_loader = DataLoader(
                CachedFashionDataset(Subset(dataset_list, train_idx), subcategories, augment=True),
                batch_size=64,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

            val_loader = DataLoader(
                CachedFashionDataset(Subset(dataset_list, val_idx), subcategories),
                batch_size=128,
                num_workers=4,
                pin_memory=True
            )

            model = EfficientCLIPFineTuner(clip_model, len(subcategories), num_layers).to(device)
            metrics = train_and_evaluate(model, train_loader, val_loader)
            fold_metrics.append(metrics['accuracy'])

            del model
            torch.cuda.empty_cache()

        avg_acc = np.mean(fold_metrics)
        results[num_layers] = avg_acc
        print(f"\n{num_layers}-layer model average accuracy: {avg_acc:.2f}%")

    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Number of Linear Layers')
    plt.ylabel('Average Validation Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(min(results.values()) - 5, max(results.values()) + 5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    return results


if __name__ == "__main__":
    dataset = load_dataset("ceyda/fashion-products-small")['train']
    subcategories = sorted(list(set(dataset['subCategory'])))

    results = compare_models(dataset, subcategories)

    print("\nFinal Comparison Results:")
    for layers, acc in results.items():
        print(f"{layers}-layer model: {acc:.2f}% accuracy")