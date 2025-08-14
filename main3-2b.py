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
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.is_encoder_frozen = freeze_encoder  # Store freeze state for later use
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
        # Freeze visual encoder if specified
        if freeze_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print("Visual encoder frozen for faster training.")

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)


class TextEncoderFineTunedCLIP(nn.Module):
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder = base_model.transformer  # Corrected to use 'transformer' for text encoder in open_clip
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device
        self.subcategories = subcategories

        # Freeze visual encoder
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
            print("Visual encoder frozen for text encoder fine-tuning.")

        # Ensure text encoder is trainable
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        print("Text encoder set to trainable.")

        # Tokenize subcategory labels for text embeddings
        self.text_inputs = self.tokenizer([f"a photo of {subcat}" for subcat in subcategories]).to(device)
        self.num_classes = len(subcategories)

        # Access text projection and token embedding from base model for complete text encoding
        self.text_projection = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.token_embedding = base_model.token_embedding

    def encode_text(self, text_inputs):
        # Adapted from open_clip's CLIP implementation for text encoding
        x = self.token_embedding(text_inputs)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # Take features from the eot embedding (end of text token)
        x = x[torch.arange(x.shape[0]), text_inputs.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images):
        # Get image embeddings
        image_features = self.visual_encoder(images)

        # Get text embeddings for all subcategories
        text_features = self.encode_text(self.text_inputs)

        # Compute similarity between image and text embeddings
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_features, text_features.T) * 100.0  # Scale logits as in CLIP
        return logits


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


def analyze_misclassifications(conf_matrix, subcategories):
    misclassifications = []
    print(f"Confusion Matrix Shape: {conf_matrix.shape}")
    print(f"Number of Subcategories: {len(subcategories)}")
    total_samples_per_class = np.sum(conf_matrix, axis=1)
    num_classes = min(conf_matrix.shape[0], conf_matrix.shape[1])  # Use matrix dimensions to be safe
    for true_label in range(num_classes):
        for pred_label in range(num_classes):
            if true_label != pred_label and conf_matrix[true_label, pred_label] > 0:
                count = conf_matrix[true_label, pred_label]
                percentage = (count / total_samples_per_class[true_label]) * 100 if total_samples_per_class[
                                                                                        true_label] > 0 else 0
                misclassifications.append({
                    'True Class': subcategories[true_label] if true_label < len(
                        subcategories) else f"Unknown_{true_label}",
                    'Predicted Class': subcategories[pred_label] if pred_label < len(
                        subcategories) else f"Unknown_{pred_label}",
                    'Count': count,
                    'Percentage of True Class': f"{percentage:.2f}%"
                })
    if misclassifications:
        print("\nSignificant Misclassifications:")
        misclass_df = pd.DataFrame(misclassifications)
        print(misclass_df.sort_values(by='Count', ascending=False).head(10))  # Top 10 misclassifications
    else:
        print("\nNo significant misclassifications found.")
    return misclassifications


def compute_metrics(true_labels, predictions, subcategories, fold, epoch, model_type, split_type="Validation"):
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    # Force confusion matrix to have dimensions equal to number of subcategories
    conf_matrix = confusion_matrix(true_labels, predictions, labels=list(range(len(subcategories))))
    print(f"\nMetrics for {model_type} - {split_type} (Fold {fold}, Epoch {epoch}):")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm", xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_type} - {split_type} (Fold {fold}, Epoch {epoch})")
    plt.savefig(
        f"confusion_matrix_{model_type.lower().replace(' ', '_')}_{split_type.lower()}_fold_{fold}_epoch_{epoch}.png")
    plt.close()
    misclassifications = analyze_misclassifications(conf_matrix, subcategories)
    return precision, recall, f1, conf_matrix, misclassifications


def evaluate_model_with_metrics(model, loader, criterion, subcategories, device, fold, epoch, model_type,
                                split_type="Validation", display_predictions=False):
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
    precision, recall, f1, conf_matrix, misclassifications = compute_metrics(true_labels, predictions, subcategories,
                                                                             fold, epoch, model_type, split_type)
    return avg_loss, precision, recall, f1, conf_matrix, misclassifications


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


def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(folds, metrics["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(folds, metrics["precision"], label="Precision", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title(f"Precision Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(folds, metrics["recall"], label="Recall", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title(f"Recall Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(folds, metrics["f1"], label="F1-Score", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.title(f"F1-Score Across Folds {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    plt.close()


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcategories, device,
                log_dir,
                fold, accumulation_steps=4, validate_every=2, model_type="full"):
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=3, delta=0.001)
    best_f1 = 0.0
    # Normalize model_type for filename (lowercase, replace spaces with underscores)
    model_type_normalized = model_type.lower().replace(' ', '_')
    metrics_dict = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'precision_val': [], 'recall_val': [], 'f1_val': [],
        'precision_test': [], 'recall_test': [], 'f1_test': []
    }
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
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

        metrics_dict['epoch'].append(epoch + 1)
        metrics_dict['train_loss'].append(train_loss)
        metrics_dict['train_acc'].append(train_acc)

        if (epoch + 1) % validate_every == 0:
            # Validation metrics
            val_loss, val_precision, val_recall, val_f1, val_conf_matrix, val_misclass = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, fold, epoch + 1, model_type,
                split_type="Validation"
            )
            val_acc = np.trace(val_conf_matrix) / np.sum(val_conf_matrix) * 100 if np.sum(val_conf_matrix) > 0 else 0
            metrics_dict['val_loss'].append(val_loss)
            metrics_dict['val_acc'].append(val_acc)
            metrics_dict['precision_val'].append(val_precision)
            metrics_dict['recall_val'].append(val_recall)
            metrics_dict['f1_val'].append(val_f1)

            # Test metrics
            test_loss, test_precision, test_recall, test_f1, test_conf_matrix, test_misclass = evaluate_model_with_metrics(
                model, test_loader, criterion, subcategories, device, fold, epoch + 1, model_type, split_type="Test"
            )
            test_acc = np.trace(test_conf_matrix) / np.sum(test_conf_matrix) * 100 if np.sum(
                test_conf_matrix) > 0 else 0
            metrics_dict['test_loss'].append(test_loss)
            metrics_dict['test_acc'].append(test_acc)
            metrics_dict['precision_test'].append(test_precision)
            metrics_dict['recall_test'].append(test_recall)
            metrics_dict['f1_test'].append(test_f1)

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("Accuracy/Test", test_acc, epoch)
            writer.add_scalar("Precision/Validation", val_precision, epoch)
            writer.add_scalar("Recall/Validation", val_recall, epoch)
            writer.add_scalar("F1-Score/Validation", val_f1, epoch)
            writer.add_scalar("Precision/Test", test_precision, epoch)
            writer.add_scalar("Recall/Test", test_recall, epoch)
            writer.add_scalar("F1-Score/Test", test_f1, epoch)

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_path = os.path.join(log_dir, f'best_{model_type_normalized}_clip.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
            early_stopping(val_f1)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        else:
            metrics_dict['val_loss'].append(metrics_dict['val_loss'][-1] if metrics_dict['val_loss'] else 0)
            metrics_dict['val_acc'].append(metrics_dict['val_acc'][-1] if metrics_dict['val_acc'] else 0)
            metrics_dict['precision_val'].append(
                metrics_dict['precision_val'][-1] if metrics_dict['precision_val'] else 0)
            metrics_dict['recall_val'].append(metrics_dict['recall_val'][-1] if metrics_dict['recall_val'] else 0)
            metrics_dict['f1_val'].append(metrics_dict['f1_val'][-1] if metrics_dict['f1_val'] else 0)
            metrics_dict['test_loss'].append(metrics_dict['test_loss'][-1] if metrics_dict['test_loss'] else 0)
            metrics_dict['test_acc'].append(metrics_dict['test_acc'][-1] if metrics_dict['test_acc'] else 0)
            metrics_dict['precision_test'].append(
                metrics_dict['precision_test'][-1] if metrics_dict['precision_test'] else 0)
            metrics_dict['recall_test'].append(metrics_dict['recall_test'][-1] if metrics_dict['recall_test'] else 0)
            metrics_dict['f1_test'].append(metrics_dict['f1_test'][-1] if metrics_dict['f1_test'] else 0)
    writer.close()
    return metrics_dict


def compare_metrics(full_metrics, text_metrics, k_folds):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    plt.plot(folds, full_metrics["val_loss"], label="Full Fine-Tuning Val Loss", marker="o")
    plt.plot(folds, text_metrics["val_loss"], label="Text Encoder Val Loss", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison Across Folds")
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(folds, full_metrics["test_loss"], label="Full Fine-Tuning Test Loss", marker="o")
    plt.plot(folds, text_metrics["test_loss"], label="Text Encoder Test Loss", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Comparison Across Folds")
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(folds, full_metrics["precision"], label="Full Fine-Tuning Precision", marker="o")
    plt.plot(folds, text_metrics["precision"], label="Text Encoder Precision", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Precision (Val)")
    plt.title("Precision Comparison Across Folds (Val)")
    plt.legend()
    plt.subplot(3, 2, 4)
    plt.plot(folds, full_metrics["recall"], label="Full Fine-Tuning Recall", marker="o")
    plt.plot(folds, text_metrics["recall"], label="Text Encoder Recall", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("Recall (Val)")
    plt.title("Recall Comparison Across Folds (Val)")
    plt.legend()
    plt.subplot(3, 2, 5)
    plt.plot(folds, full_metrics["f1"], label="Full Fine-Tuning F1-Score", marker="o")
    plt.plot(folds, text_metrics["f1"], label="Text Encoder F1-Score", marker="x")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score (Val)")
    plt.title("F1-Score Comparison Across Folds (Val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_metrics.png")
    plt.close()


def create_comparison_table(full_metrics, text_metrics, k_folds):
    comparison_data = []
    for fold in range(k_folds):
        comparison_data.append({
            'Fold': fold + 1,
            'Full Train Loss': full_metrics['train_loss'][fold] if full_metrics['train_loss'][fold] else 0,
            'Full Val Loss': full_metrics['val_loss'][fold],
            'Full Test Loss': full_metrics['test_loss'][fold],
            'Full Precision (Val)': full_metrics['precision'][fold],
            'Full Recall (Val)': full_metrics['recall'][fold],
            'Full F1-Score (Val)': full_metrics['f1'][fold],
            'Text Train Loss': text_metrics['train_loss'][fold] if text_metrics['train_loss'][fold] else 0,
            'Text Val Loss': text_metrics['val_loss'][fold],
            'Text Test Loss': text_metrics['test_loss'][fold],
            'Text Precision (Val)': text_metrics['precision'][fold],
            'Text Recall (Val)': text_metrics['recall'][fold],
            'Text F1-Score (Val)': text_metrics['f1'][fold]
        })
    # Add average row
    comparison_data.append({
        'Fold': 'Average',
        'Full Train Loss': np.mean(full_metrics['train_loss']),
        'Full Val Loss': np.mean(full_metrics['val_loss']),
        'Full Test Loss': np.mean(full_metrics['test_loss']),
        'Full Precision (Val)': np.mean(full_metrics['precision']),
        'Full Recall (Val)': np.mean(full_metrics['recall']),
        'Full F1-Score (Val)': np.mean(full_metrics['f1']),
        'Text Train Loss': np.mean(text_metrics['train_loss']),
        'Text Val Loss': np.mean(text_metrics['val_loss']),
        'Text Test Loss': np.mean(text_metrics['test_loss']),
        'Text Precision (Val)': np.mean(text_metrics['precision']),
        'Text Recall (Val)': np.mean(text_metrics['recall']),
        'Text F1-Score (Val)': np.mean(text_metrics['f1'])
    })
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("comparison_table.csv", index=False)
    print("\nComparison Table of Metrics Across Folds:")
    print(comparison_df)
    return comparison_df


if __name__ == '__main__':
    freeze_support()
    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device, model_name="ViT-B-32", pretrained_weights="openai"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    # K-Fold Cross-Validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Metrics storage for both approaches
    full_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}
    text_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}

    # Set number of workers based on CPU cores
    num_workers = min(8, os.cpu_count())
    print(f"Using {num_workers} workers for data loading.")

    # Store metrics per fold for detailed comparison
    all_full_metrics = []
    all_text_metrics = []

    for fold, (train_val_indices, test_indices) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        # Further split train_val_indices into train and validation (80-20 split as an example)
        train_val_size = len(train_val_indices)
        train_size = int(0.8 * train_val_size)
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        train_subset = Subset(dataset, train_indices.tolist())
        val_subset = Subset(dataset, val_indices.tolist())
        test_subset = Subset(dataset, test_indices.tolist())

        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset = FashionDataset(val_subset, subcategories)
        test_dataset = FashionDataset(test_subset, subcategories)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        # 1. Full Fine-Tuning (or with frozen visual encoder and classifier head)
        print("\nTraining Full Fine-Tuned Model...")
        full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=True).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        params_to_optimize_full = full_model.classifier.parameters() if full_model.is_encoder_frozen else full_model.parameters()
        optimizer_full = optim.AdamW(params_to_optimize_full, lr=1e-4)
        full_log_dir = f"logs/3-2/full_finetune/fold_{fold + 1}/"
        full_metrics_dict = train_model(
            full_model, train_loader, val_loader, test_loader, criterion, optimizer_full, num_epochs=4,
            subcategories=subcategories, device=device, log_dir=full_log_dir, fold=fold + 1,
            accumulation_steps=4, validate_every=2, model_type="Full Fine-Tuning"
        )
        # Check if the best model file exists before loading
        full_model_type_normalized = "full_fine-tuning"
        full_best_model_path = os.path.join(full_log_dir, f'best_{full_model_type_normalized}_clip.pth')
        if os.path.exists(full_best_model_path):
            full_model.load_state_dict(torch.load(full_best_model_path))
            print(f"Loaded best model from {full_best_model_path}")
        else:
            print(f"Best model file not found at {full_best_model_path}. Using current model weights.")
        full_val_loss, full_precision, full_recall, full_f1, full_conf_matrix, full_val_misclass = evaluate_model_with_metrics(
            full_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Full Fine-Tuning",
            split_type="Validation"
        )
        full_test_loss, full_test_precision, full_test_recall, full_test_f1, full_test_conf_matrix, full_test_misclass = evaluate_model_with_metrics(
            full_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Full Fine-Tuning",
            split_type="Test"
        )
        full_fold_metrics["train_loss"].append(full_metrics_dict['train_loss'][-1])
        full_fold_metrics["val_loss"].append(full_val_loss)
        full_fold_metrics["test_loss"].append(full_test_loss)
        full_fold_metrics["precision"].append(full_precision)
        full_fold_metrics["recall"].append(full_recall)
        full_fold_metrics["f1"].append(full_f1)
        all_full_metrics.append(full_metrics_dict)
        save_metrics_to_csv({
            "epoch": full_metrics_dict['epoch'],
            "train_loss": full_metrics_dict['train_loss'],
            "val_loss": full_metrics_dict['val_loss'],
            "test_loss": full_metrics_dict['test_loss'],
            "train_acc": full_metrics_dict['train_acc'],
            "val_acc": full_metrics_dict['val_acc'],
            "test_acc": full_metrics_dict['test_acc'],
            "precision_val": full_metrics_dict['precision_val'],
            "recall_val": full_metrics_dict['recall_val'],
            "f1_val": full_metrics_dict['f1_val'],
            "precision_test": full_metrics_dict['precision_test'],
            "recall_test": full_metrics_dict['recall_test'],
            "f1_test": full_metrics_dict['f1_test']
        }, f"metrics_full_finetune_fold_{fold + 1}.csv")
        save_logs_as_zip(full_log_dir, f"logs/3-2/full_finetune/fold_{fold + 1}_logs")
        torch.cuda.empty_cache()

        # 2. Text Encoder Fine-Tuning
        print("\nTraining Text Encoder Fine-Tuned Model...")
        text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer_text = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)
        text_log_dir = f"logs/3-2/text_encoder_finetune/fold_{fold + 1}/"
        text_metrics_dict = train_model(
            text_model, train_loader, val_loader, test_loader, criterion, optimizer_text, num_epochs=4,
            subcategories=subcategories, device=device, log_dir=text_log_dir, fold=fold + 1,
            accumulation_steps=4, validate_every=2, model_type="Text Encoder Fine-Tuning"
        )
        # Check if the best model file exists before loading
        text_model_type_normalized = "text_encoder_fine-tuning"
        text_best_model_path = os.path.join(text_log_dir, f'best_{text_model_type_normalized}_clip.pth')
        if os.path.exists(text_best_model_path):
            text_model.load_state_dict(torch.load(text_best_model_path))
            print(f"Loaded best model from {text_best_model_path}")
        else:
            print(f"Best model file not found at {text_best_model_path}. Using current model weights.")
        text_val_loss, text_precision, text_recall, text_f1, text_conf_matrix, text_val_misclass = evaluate_model_with_metrics(
            text_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Text Encoder Fine-Tuning",
            split_type="Validation"
        )
        text_test_loss, text_test_precision, text_test_recall, text_test_f1, text_test_conf_matrix, text_test_misclass = evaluate_model_with_metrics(
            text_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Text Encoder Fine-Tuning",
            split_type="Test"
        )
        text_fold_metrics["train_loss"].append(text_metrics_dict['train_loss'][-1])
        text_fold_metrics["val_loss"].append(text_val_loss)
        text_fold_metrics["test_loss"].append(text_test_loss)
        text_fold_metrics["precision"].append(text_precision)
        text_fold_metrics["recall"].append(text_recall)
        text_fold_metrics["f1"].append(text_f1)
        all_text_metrics.append(text_metrics_dict)
        save_metrics_to_csv({
            "epoch": text_metrics_dict['epoch'],
            "train_loss": text_metrics_dict['train_loss'],
            "val_loss": text_metrics_dict['val_loss'],
            "test_loss": text_metrics_dict['test_loss'],
            "train_acc": text_metrics_dict['train_acc'],
            "val_acc": text_metrics_dict['val_acc'],
            "test_acc": text_metrics_dict['test_acc'],
            "precision_val": text_metrics_dict['precision_val'],
            "recall_val": text_metrics_dict['recall_val'],
            "f1_val": text_metrics_dict['f1_val'],
            "precision_test": text_metrics_dict['precision_test'],
            "recall_test": text_metrics_dict['recall_test'],
            "f1_test": text_metrics_dict['f1_test']
        }, f"metrics_text_encoder_fold_{fold + 1}.csv")
        save_logs_as_zip(text_log_dir, f"logs/3-2/text_encoder_finetune/fold_{fold + 1}_logs")
        torch.cuda.empty_cache()

    # Plot and save metrics for both approaches
    plot_fold_metrics(full_fold_metrics, k_folds, title_suffix="Full Fine-Tuning")
    plot_fold_metrics(text_fold_metrics, k_folds, title_suffix="Text Encoder Fine-Tuning")
    compare_metrics(full_fold_metrics, text_fold_metrics, k_folds)
    save_metrics_to_csv(full_fold_metrics, "kfold_full_finetune_metrics.csv")
    save_metrics_to_csv(text_fold_metrics, "kfold_text_encoder_metrics.csv")

    # Create comparison table
    create_comparison_table(full_fold_metrics, text_fold_metrics, k_folds)

    # Summary of average metrics across folds
    print("\nSummary of Average Metrics Across Folds:")
    print("\nFull Fine-Tuning:")
    print(f"Avg Train Loss: {np.mean(full_fold_metrics['train_loss']):.4f}")
    print(f"Avg Validation Loss: {np.mean(full_fold_metrics['val_loss']):.4f}")
    print(f"Avg Test Loss: {np.mean(full_fold_metrics['test_loss']):.4f}")
    print(f"Avg Precision (Val): {np.mean(full_fold_metrics['precision']):.4f}")
    print(f"Avg Recall (Val): {np.mean(full_fold_metrics['recall']):.4f}")
    print(f"Avg F1-Score (Val): {np.mean(full_fold_metrics['f1']):.4f}")
    print("\nText Encoder Fine-Tuning:")
    print(f"Avg Train Loss: {np.mean(text_fold_metrics['train_loss']):.4f}")
    print(f"Avg Validation Loss: {np.mean(text_fold_metrics['val_loss']):.4f}")
    print(f"Avg Test Loss: {np.mean(text_fold_metrics['test_loss']):.4f}")
    print(f"Avg Precision (Val): {np.mean(text_fold_metrics['precision']):.4f}")
    print(f"Avg Recall (Val): {np.mean(text_fold_metrics['recall']):.4f}")
    print(f"Avg F1-Score (Val): {np.mean(text_fold_metrics['f1']):.4f}")
    print("\nK-Fold training complete. Results saved for analysis.")