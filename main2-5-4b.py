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

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Enable CuDNN benchmarking for GPU optimization
torch.backends.cudnn.benchmark = True


# Setup environment with GPU to CPU fallback
def setup_environment():
    """Initialize device and verify environment with fallback to CPU if GPU fails."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error with GPU: {e}. Switching to CPU...")
            device = torch.device("cpu")
    return device


# Load CLIP Model
def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    """Load CLIP model and preprocessors."""
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


# Custom CLIP Fine-Tuner (Fine-Tuned Model)
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


# Non-Fine-Tuned Model (Base CLIP with Simple Classifier)
class NonFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.visual_encoder = base_model.visual
        # Freeze all parameters for non-fine-tuned model
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]
        self.classifier = nn.Linear(output_size, num_classes)

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
        # Convert index to int to avoid TypeError with Hugging Face datasets
        idx = int(idx)
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
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
    return precision, recall, f1, conf_matrix


# Visualize Predictions
def visualize_predictions(images_list, subcategories):
    """Display predicted vs actual labels for validation images."""
    num_samples_to_display = min(8, len(images_list))
    plt.figure(figsize=(15, 10))
    for i in range(num_samples_to_display):
        images, labels, predicted = images_list[i]
        for j in range(min(len(images), num_samples_to_display)):
            plt.subplot(2, 4, j + 1)
            image = images[j].permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())
            actual_label = subcategories[labels[j].item()]
            predicted_label = subcategories[predicted[j].item()]
            plt.imshow(image)
            plt.title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


# Evaluate Model
def evaluate_model_with_metrics(model, loader, criterion, subcategories, display_predictions=False):
    """Evaluate model performance and compute metrics."""
    model.eval()
    total_loss = 0.0
    true_labels = []
    predictions = []
    images_list = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            if display_predictions:
                images_list.append((images.cpu(), labels.cpu(), predicted.cpu()))
    avg_loss = total_loss / len(loader)
    precision, recall, f1, conf_matrix = compute_metrics(true_labels, predictions, subcategories)
    if display_predictions:
        visualize_predictions(images_list, subcategories)
    return avg_loss, precision, recall, f1, conf_matrix


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=3, delta=0):
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


# Save Logs as Zip
def save_logs_as_zip(log_dir, zip_name):
    """Save logs directory as a zip file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(zip_name), exist_ok=True)
        shutil.make_archive(zip_name, 'zip', log_dir)
        print(f"Logs saved as {zip_name}.zip")
    except Exception as e:
        print(f"Error saving logs as zip: {e}")


# Save Metrics to CSV
def save_metrics_to_csv(metrics, filename):
    """Save metrics to a CSV file for easy reference."""
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


# Train Model with TensorBoard and Early Stopping
def train_model_with_tensorboard(
        model, train_loader, val_loader, optimizer_type, model_type, num_epochs=3, lr=1e-4, log_dir="logs/part2-5-4/", fold_index=0
):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir=f"{log_dir}{model_type}_{optimizer_type}_fold_{fold_index}/")
    early_stopping = EarlyStopping(patience=3)
    best_f1 = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        progress_bar = tqdm(train_loader, desc=f'{model_type} {optimizer_type} Fold {fold_index}, Epoch {epoch + 1}/{num_epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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
            model, val_loader, criterion, subcategories, display_predictions=True
        )
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Precision/Validation", precision, epoch)
        writer.add_scalar("Recall/Validation", recall, epoch)
        writer.add_scalar("F1-Score/Validation", f1, epoch)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f'best_clip_model_{model_type}_{optimizer_type}_fold_{fold_index}.pth')
        early_stopping(f1)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    writer.close()
    # Save logs as zip after training
    save_logs_as_zip(
        f"{log_dir}{model_type}_{optimizer_type}_fold_{fold_index}/",
        f"logs/part2-5-4/{model_type}_{optimizer_type}_fold_{fold_index}_logs"
    )
    return val_loss, precision, recall, f1


# Plot Metrics for Comparison
def plot_metrics(metrics, k_folds):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(folds, metrics["adam_fine_tuned"]["val_loss"], label="Adam (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adamw_fine_tuned"]["val_loss"], label="AdamW (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adam_non_fine_tuned"]["val_loss"], label="Adam (Non-Fine-Tuned)", marker="x")
    plt.plot(folds, metrics["adamw_non_fine_tuned"]["val_loss"], label="AdamW (Non-Fine-Tuned)", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Across Folds")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(folds, metrics["adam_fine_tuned"]["precision"], label="Adam (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adamw_fine_tuned"]["precision"], label="AdamW (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adam_non_fine_tuned"]["precision"], label="Adam (Non-Fine-Tuned)", marker="x")
    plt.plot(folds, metrics["adamw_non_fine_tuned"]["precision"], label="AdamW (Non-Fine-Tuned)", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title("Precision Across Folds")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(folds, metrics["adam_fine_tuned"]["recall"], label="Adam (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adamw_fine_tuned"]["recall"], label="AdamW (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adam_non_fine_tuned"]["recall"], label="Adam (Non-Fine-Tuned)", marker="x")
    plt.plot(folds, metrics["adamw_non_fine_tuned"]["recall"], label="AdamW (Non-Fine-Tuned)", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title("Recall Across Folds")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(folds, metrics["adam_fine_tuned"]["f1"], label="Adam (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adamw_fine_tuned"]["f1"], label="AdamW (Fine-Tuned)", marker="o")
    plt.plot(folds, metrics["adam_non_fine_tuned"]["f1"], label="Adam (Non-Fine-Tuned)", marker="x")
    plt.plot(folds, metrics["adamw_non_fine_tuned"]["f1"], label="AdamW (Non-Fine-Tuned)", marker="x")
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
        model_name="ViT-B-32",
        pretrained_weights="openai"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories)
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    metrics = {
        "adam_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adamw_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adam_non_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adamw_non_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []}
    }
    for optimizer_type in ["adam", "adamw"]:
        for fold_index, (train_indices, val_indices) in enumerate(kf.split(dataset)):
            print(f"\nStarting {optimizer_type.upper()} Fold {fold_index + 1}/{k_folds}...")
            # Convert indices to list of integers to avoid numpy.int64 issue
            train_indices = train_indices.tolist()
            val_indices = val_indices.tolist()
            # Use Subset for memory efficiency
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            train_dataset = FashionDataset(train_subset, subcategories, augment=True)
            val_dataset = FashionDataset(val_subset, subcategories)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Fine-Tuned Model
            print(f"Training Fine-Tuned Model with {optimizer_type}...")
            fine_tuned_model = CustomCLIPFineTuner(clip_model, len(subcategories), num_layers=1).to(device)
            val_loss_ft, precision_ft, recall_ft, f1_ft = train_model_with_tensorboard(
                fine_tuned_model, train_loader, val_loader, optimizer_type=optimizer_type, model_type="fine_tuned",
                num_epochs=3, lr=1e-4, log_dir="logs/part2-5-4/", fold_index=fold_index
            )
            metrics[f"{optimizer_type}_fine_tuned"]["val_loss"].append(val_loss_ft)
            metrics[f"{optimizer_type}_fine_tuned"]["precision"].append(precision_ft)
            metrics[f"{optimizer_type}_fine_tuned"]["recall"].append(recall_ft)
            metrics[f"{optimizer_type}_fine_tuned"]["f1"].append(f1_ft)

            # Non-Fine-Tuned Model
            print(f"Training Non-Fine-Tuned Model with {optimizer_type}...")
            non_fine_tuned_model = NonFineTunedCLIP(clip_model, len(subcategories)).to(device)
            val_loss_nft, precision_nft, recall_nft, f1_nft = train_model_with_tensorboard(
                non_fine_tuned_model, train_loader, val_loader, optimizer_type=optimizer_type, model_type="non_fine_tuned",
                num_epochs=3, lr=1e-4, log_dir="logs/part2-5-4/", fold_index=fold_index
            )
            metrics[f"{optimizer_type}_non_fine_tuned"]["val_loss"].append(val_loss_nft)
            metrics[f"{optimizer_type}_non_fine_tuned"]["precision"].append(precision_nft)
            metrics[f"{optimizer_type}_non_fine_tuned"]["recall"].append(recall_nft)
            metrics[f"{optimizer_type}_non_fine_tuned"]["f1"].append(f1_nft)

    plot_metrics(metrics, k_folds)
    # Save metrics to CSV
    save_metrics_to_csv(metrics, "metrics_summary.csv")