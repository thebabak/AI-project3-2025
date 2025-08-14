# %% [markdown]
## Optimized Image Classification Workflow with CLIP Visual Encoder

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
from multiprocessing import freeze_support

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Enable CuDNN benchmarking for GPU optimization
torch.backends.cudnn.benchmark = True


# %% [markdown]
### 1. Setup Environment

def setup_environment():
    """Initialize device and verify environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device


# %% [markdown]
### 2. Load CLIP Model

def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
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


# %% [markdown]
### 3. Custom CLIP Model with Last Two Layers for Fine-Tuning

class CustomCLIPFineTuner(nn.Module):
    """
    Fine-tuning wrapper for CLIP's visual encoder with last two layers trainable
    and a customizable classifier.
    """

    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual

        # Freeze all layers except the last two
        layers = list(self.visual_encoder.children())
        for layer in layers[:-2]:
            for param in layer.parameters():
                param.requires_grad = False

        # Determine the output size dynamically
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]

        # Replace the last projection layer with a custom classifier
        classifier_layers = []
        for i in range(num_layers):
            classifier_layers.append(nn.Linear(output_size, output_size if i < num_layers - 1 else num_classes))
            if i < num_layers - 1:
                classifier_layers.append(nn.ReLU())

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)


# %% [markdown]
### 4. Load Dataset

def load_fashion_dataset():
    """Load fashion dataset."""
    ds = load_dataset("ceyda/fashion-products-small")
    dataset = ds['train']
    subcategories = sorted(list(set(dataset['subCategory'])))
    return dataset, subcategories


# %% [markdown]
### 5. Compute Class Weights

def compute_class_weights(dataset, subcategories):
    """Compute class weights based on the dataset distribution."""
    labels = [item['subCategory'] for item in dataset]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[subcategory] for subcategory in subcategories]
    return torch.tensor(class_weights, dtype=torch.float).to(device)


# %% [markdown]
### 6. Custom PyTorch Dataset with Augmentations

class FashionDataset(Dataset):
    """Custom dataset for fashion products with preprocessing and augmentations."""

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


# %% [markdown]
### 7. Training Function with Optimizer Selection

def train_model_with_optimizer(model, train_loader, val_loader, optimizer_type="adam", num_epochs=3, lr=1e-4):
    """
    Train the model using the specified optimizer type: 'adam' or 'adamw'.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Select optimizer based on user input
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    scaler = torch.amp.GradScaler()
    best_acc = 0.0

    print(f"\nTraining with {optimizer_type.upper()} optimizer...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Mixed precision forward and backward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})

        val_acc = evaluate_model(model, val_loader, subcategories)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_clip_model_{optimizer_type}.pth')

    return best_acc


# %% [markdown]
### 8. Evaluation Function

def evaluate_model(model, loader, subcategories):
    """Evaluate model performance and visualize confusion matrix."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return 100 * correct / total


# %% [markdown]
### 9. Compare Optimizers

def compare_optimizers(dataset, subcategories, clip_model, batch_size=32, num_epochs=3, lr=1e-4):
    """
    Train the model using Adam and AdamW optimizers and compare their performance.
    """
    results = {}

    # Perform k-fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for optimizer_type in ["adam", "adamw"]:
        print(f"\n--- Training with {optimizer_type.upper()} Optimizer ---")
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"\n--- Fold {fold + 1}/5 ---")
            train_subset = Subset(dataset, [int(i) for i in train_idx])
            val_subset = Subset(dataset, [int(i) for i in val_idx])
            train_dataset = FashionDataset(train_subset, subcategories, augment=True)
            val_dataset = FashionDataset(val_subset, subcategories)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

            # Initialize the model
            model = CustomCLIPFineTuner(clip_model, len(subcategories), num_layers=1).to(device)

            # Train and evaluate the model
            val_acc = train_model_with_optimizer(
                model, train_loader, val_loader, optimizer_type=optimizer_type, num_epochs=num_epochs, lr=lr
            )
            fold_results.append(val_acc)

            print(f"Fold {fold + 1} Validation Accuracy with {optimizer_type.upper()}: {val_acc:.2f}%")

        # Compute average accuracy across folds
        avg_acc = sum(fold_results) / len(fold_results)
        results[optimizer_type] = avg_acc
        print(f"Average Validation Accuracy with {optimizer_type.upper()}: {avg_acc:.2f}%")

    # Compare results
    print("\n--- Optimizer Comparison ---")
    for optimizer_type, avg_acc in results.items():
        print(f"{optimizer_type.upper()} Optimizer: Average Validation Accuracy = {avg_acc:.2f}%")

    # Visualize results
    plt.bar(results.keys(), results.values(), color=['blue', 'green'])
    plt.xlabel('Optimizer Type')
    plt.ylabel('Average Validation Accuracy (%)')
    plt.title('Comparison of Optimizer Performance')
    plt.show()

    return results


# %% [markdown]
### 10. Main Workflow

if __name__ == '__main__':
    freeze_support()  # Ensure multiprocessing works safely on Windows

    # Setup environment
    device = setup_environment()

    # Load the CLIP model
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device,
        model_name="ViT-B-32",  # Using ViT-B-32 model
        pretrained_weights="openai"  # Pretrained weights from OpenAI
    )

    # Load the dataset and subcategories
    dataset, subcategories = load_fashion_dataset()

    # Compute class weights
    class_weights = compute_class_weights(dataset, subcategories)

    # Compare Adam and AdamW optimizers
    optimizer_results = compare_optimizers(dataset, subcategories, clip_model, batch_size=32, num_epochs=3, lr=1e-4)