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

### Setup Environment

def setup_environment():
    """Initialize device and verify environment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device

### Load CLIP Model

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

### Custom CLIP Model with ReLU Activation

class CustomCLIPFineTuner(nn.Module):
    """
    Fine-tuning wrapper for CLIP's visual encoder with last two layers trainable
    and a customizable classifier using ReLU activation.
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

        # Replace the last projection layer with a custom classifier using ReLU
        classifier_layers = []
        for i in range(num_layers):
            classifier_layers.append(nn.Linear(output_size, output_size if i < num_layers - 1 else num_classes))
            if i < num_layers - 1:
                classifier_layers.append(nn.ReLU())  # Use ReLU activation

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)

### Dataset and Preprocessing

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

### Load Fashion Dataset

def load_fashion_dataset():
    """Load fashion dataset."""
    ds = load_dataset("ceyda/fashion-products-small")
    dataset = ds['train']
    subcategories = sorted(list(set(dataset['subCategory'])))
    return dataset, subcategories

### Compute Class Weights

def compute_class_weights(dataset, subcategories):
    """Compute class weights based on dataset distribution."""
    counts = Counter(dataset['subCategory'])
    total = sum(counts.values())
    weights = [total / counts[subcat] for subcat in subcategories]
    weights = torch.tensor(weights, dtype=torch.float32)
    return weights

### Compute Metrics

def compute_metrics(true_labels, predictions, subcategories):
    """Compute precision, recall, F1-score, and confusion matrix."""
    precision = precision_score(true_labels, predictions, average="weighted")
    recall = recall_score(true_labels, predictions, average="weighted")
    f1 = f1_score(true_labels, predictions, average="weighted")
    conf_matrix = confusion_matrix(true_labels, predictions)

    print("\nMetrics:")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    return precision, recall, f1, conf_matrix

### Evaluation with Metrics

def evaluate_model_with_metrics(model, loader, criterion, subcategories):
    """Evaluate model performance and compute metrics."""
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

### Main Workflow with Metrics

if __name__ == '__main__':
    freeze_support()  # Ensure multiprocessing works safely on Windows

    # Setup environment
    device = setup_environment()

    # Load the CLIP model
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device,
        model_name="ViT-B-32",
        pretrained_weights="openai"
    )

    # Load the dataset and subcategories
    dataset, subcategories = load_fashion_dataset()

    # Compute class weights
    class_weights = compute_class_weights(dataset, subcategories)

    # Prepare for k-fold cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize TensorBoard SummaryWriter with specific log directory
    log_dir = "logs/2-5-3"
    writer = SummaryWriter(log_dir)

    # Lists to store losses for plotting
    train_losses, val_losses = [], []

    for fold_index, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"\nStarting Fold {fold_index + 1}/{k_folds}...")

        # Convert indices to int for compatibility
        train_dataset = FashionDataset([dataset[int(i)] for i in train_indices], subcategories, augment=True)
        val_dataset = FashionDataset([dataset[int(i)] for i in val_indices], subcategories)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Initialize model
        model = CustomCLIPFineTuner(clip_model, len(subcategories), num_layers=1).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        # Train the model
        for epoch in range(3):  # Example: train for 3 epochs
            model.train()
            epoch_train_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}'):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Evaluate on validation set
            val_loss, precision, recall, f1, conf_matrix = evaluate_model_with_metrics(model, val_loader, criterion, subcategories)
            val_losses.append(val_loss)

            # Log training and validation losses to TensorBoard
            writer.add_scalar('Loss/train', avg_train_loss, epoch + fold_index * 3)  # Adjust step for TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch + fold_index * 3)

            # Log metrics to TensorBoard
            writer.add_scalar('Metrics/precision', precision, epoch + fold_index * 3)
            writer.add_scalar('Metrics/recall', recall, epoch + fold_index * 3)
            writer.add_scalar('Metrics/f1', f1, epoch + fold_index * 3)

            print(f"Fold {fold_index + 1}, Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Validation Loss = {val_loss:.4f}")

    # Close the TensorBoard writer
    writer.close()

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.title('Losses Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()