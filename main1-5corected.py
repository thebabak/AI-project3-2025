# %% [markdown]
## Image Classification with ConvNeXt: Full Workflow with GPU, Batch Visualization, Confusion Matrix, and Cosine Similarity

# %%
# Import required libraries
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
from torch.cuda.amp import GradScaler, autocast

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


# %% [markdown]
### 1. Enhanced Environment Setup

def setup_environment():
    """Initialize device with CUDA optimizations"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        # Enable TF32 for faster math on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return device


device = setup_environment()


# %% [markdown]
### 2. Robust Model Loading

def load_convnext_model(device, model_name="convnext_base_w", pretrained_weights="laion2b_s13b_b82k_augreg",
                        local_weights_path=None):
    """
    Load ConvNeXt model with:
    - Local weight verification
    - Automatic fallback
    - Memory reporting
    """
    try:
        if local_weights_path:
            if not os.path.exists(local_weights_path):
                raise FileNotFoundError(f"Weights file not found: {local_weights_path}")
            print(f"Loading local weights from: {local_weights_path}")

            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=None,
                device='cpu'  # Load on CPU first for safety
            )
            state_dict = torch.load(local_weights_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.to(device)
        else:
            print("Loading pretrained weights from online source")
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_weights,
                device=device
            )

        # Memory reporting
        if device.type == 'cuda':
            print(f"Model memory usage: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")

        return model, preprocess_train, preprocess_val

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Fallback to smaller model
        print("Falling back to convnext_small_w")
        return open_clip.create_model_and_transforms(
            model_name="convnext_small_w",
            pretrained="laion2b_s13b_b82k_augreg",
            device=device
        )


# Load model
convnext_model, preprocess_train, preprocess_val = load_convnext_model(
    device,
    local_weights_path=None  # Set path if needed
)


# %% [markdown]
### 3. Enhanced Dataset with Advanced Augmentations

class FashionDataset(Dataset):
    """Custom dataset with advanced augmentations and class balancing"""

    def __init__(self, data, subcategories, transform=None, augment=False):
        self.data = data
        self.subcategories = subcategories
        self.augment = augment

        # Base transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # Advanced augmentations
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3)
        ])

        # Class weights with smoothing
        class_counts = Counter(item['subCategory'] for item in data)
        self.class_weights = torch.tensor([
            sum(class_counts.values()) / (class_counts[cat] + 1e-6)
            for cat in subcategories
        ], dtype=torch.float32).to(device)

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


# Load dataset
ds = load_dataset("ceyda/fashion-products-small")
dataset = ds['train']
subcategories = sorted(list(set(dataset['subCategory'])))


# %% [markdown]
### 4. Enhanced Model with AMP Support

class ConvNeXtFineTuner(nn.Module):
    """Fine-tuning wrapper with AMP support"""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.convnext = base_model
        for param in self.convnext.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(640, num_classes)

    def forward(self, images):
        with autocast():
            features = self.convnext.encode_image(images)
            return self.classifier(features)


# %% [markdown]
### 5. Cosine Similarity Computation

def compute_cosine_similarity(model, preprocess, images, descriptions, device):
    """Compute similarity between images and text with AMP"""
    model.eval()
    preprocessed_images = torch.stack([preprocess(image) for image in images]).to(device)

    with torch.no_grad(), autocast():
        # Image features
        image_features = model.encode_image(preprocessed_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Text features
        text_inputs = open_clip.tokenize(descriptions).to(device)
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return (image_features @ text_features.T).cpu().numpy()


# %% [markdown]
### 6. Training with AMP and Early Stopping

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    """Training with AMP and early stopping"""
    criterion = nn.CrossEntropyLoss(weight=model.class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    best_acc = 0.0
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})

        # Validation
        val_acc = evaluate_model(model, val_loader, subcategories)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience += 1
            if patience >= 2:
                print("Early stopping triggered")
                break

    return model


# %% [markdown]
### 7. Enhanced Evaluation

def evaluate_model(model, loader, subcategories):
    """Comprehensive evaluation with metrics"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=subcategories, yticklabels=subcategories)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds)


# %% [markdown]
### 8. K-Fold Cross Validation

def k_fold_cross_validation(model, dataset, subcategories, k_folds=5, batch_size=64):
    """Enhanced k-fold with AMP and early stopping"""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n{'=' * 40}\nFold {fold + 1}/{k_folds}\n{'=' * 40}")

        # Create datasets
        train_subset = Subset(dataset, [int(i) for i in train_idx])
        val_subset = Subset(dataset, [int(i) for i in val_idx])

        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset = FashionDataset(val_subset, subcategories)

        # DataLoaders with optimized settings
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, persistent_workers=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True
        )

        # Initialize model
        model_ft = ConvNeXtFineTuner(model, len(subcategories)).to(device)
        model_ft.class_weights = train_dataset.class_weights

        # Train
        model_ft = train_model(model_ft, train_loader, val_loader)

        # Evaluate
        val_acc = evaluate_model(model_ft, val_loader, subcategories)
        results.append(val_acc)
        print(f"Fold {fold + 1} Val Acc: {val_acc:.2f}%")

    print(f"\nAverage Accuracy: {np.mean(results):.2f}% ± {np.std(results):.2f}")
    return results


# %% [markdown]
### 9. Similarity Visualization

def visualize_similarity(dataset, subcategories, num_samples=8):
    """Compute and visualize cosine similarity"""
    selected_indices = [int(i) for i in np.random.choice(len(dataset), num_samples, replace=False)]
    selected_samples = [dataset[i] for i in selected_indices]

    images = [sample['image'] for sample in selected_samples]
    descriptions = [f"A high-quality {sample['subCategory']} product" for sample in selected_samples]

    similarities = compute_cosine_similarity(
        convnext_model, preprocess_val, images, descriptions, device
    )

    # Plot matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarities, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=[f"Desc {i + 1}" for i in range(num_samples)],
                yticklabels=[f"Img {i + 1}" for i in range(num_samples)])
    plt.title("Image-Text Similarity Matrix")
    plt.show()

    # Show sample matches
    for i in range(num_samples):
        plt.figure(figsize=(5, 5))
        plt.imshow(images[i])
        plt.title(f"{descriptions[i]}\nSimilarity: {similarities[i, i]:.2f}")
        plt.axis('off')
        plt.show()


# %% [markdown]
### 10. Main Workflow

if __name__ == "__main__":
    # K-Fold Cross Validation
    results = k_fold_cross_validation(convnext_model, dataset, subcategories)

    # Similarity Analysis
    visualize_similarity(dataset, subcategories)