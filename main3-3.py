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
import json


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


class VisionOnlyFineTunedCLIP(nn.Module):
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
        # Ensure visual encoder is trainable
        for param in self.visual_encoder.parameters():
            param.requires_grad = True
        print("Visual encoder set to trainable for vision-only fine-tuning.")

    def forward(self, images):
        features = self.visual_encoder(images)
        return self.classifier(features)


class TextEncoderFineTunedCLIP(nn.Module):
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder = base_model.transformer
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
        # Access text projection and token embedding
        self.text_projection = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.token_embedding = base_model.token_embedding

    def encode_text(self, text_inputs):
        x = self.token_embedding(text_inputs)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_inputs.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images):
        image_features = self.visual_encoder(images)
        text_features = self.encode_text(self.text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_features, text_features.T) * 100.0
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
    num_classes = min(conf_matrix.shape[0], conf_matrix.shape[1])
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
        print(misclass_df.sort_values(by='Count', ascending=False).head(10))
    else:
        print("\nNo significant misclassifications found.")
    return misclassifications


def compute_metrics(true_labels, predictions, subcategories, fold, epoch, model_type, split_type="Validation"):
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
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
    # Debug: Print the contents of metrics to diagnose empty lists
    print(f"Metrics for {title_suffix}:")
    for key, value in metrics.items():
        print(f"{key}: {value} (length: {len(value)})")

    # Handle empty lists by adjusting the folds range to match the shortest metric list length
    metric_lengths = [len(metrics[key]) for key in metrics if len(metrics[key]) > 0]
    if not metric_lengths:  # If all lists are empty
        print(f"Warning: No metrics data available for {title_suffix}. Skipping plot.")
        return
    num_points = min(metric_lengths) if metric_lengths else 0
    folds = range(1, num_points + 1) if num_points > 0 else range(1, k_folds + 1)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    if len(metrics["val_loss"]) > 0:
        plt.plot(folds[:len(metrics["val_loss"])], metrics["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 2)
    if len(metrics["precision"]) > 0:
        plt.plot(folds[:len(metrics["precision"])], metrics["precision"], label="Precision", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Precision")
    plt.title(f"Precision Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 3)
    if len(metrics["recall"]) > 0:
        plt.plot(folds[:len(metrics["recall"])], metrics["recall"], label="Recall", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("Recall")
    plt.title(f"Recall Across Folds {title_suffix}")
    plt.legend()
    plt.subplot(2, 2, 4)
    if len(metrics["f1"]) > 0:
        plt.plot(folds[:len(metrics["f1"])], metrics["f1"], label="F1-Score", marker="o")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score")
    plt.title(f"F1-Score Across Folds {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    plt.close()


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcategories, device,
                log_dir, fold, accumulation_steps=4, validate_every=2, model_type="full"):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    scaler = torch.amp.GradScaler()
    early_stopping = EarlyStopping(patience=3, delta=0.001)
    best_f1 = 0.0
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


def compare_metrics(full_metrics, text_metrics, vision_metrics, k_folds):
    # Debug: Print lengths of metrics for each approach
    print("Metrics lengths for comparison:")
    print(f"Full Fine-Tuning val_loss: {len(full_metrics['val_loss'])}")
    print(f"Text Encoder Fine-Tuning val_loss: {len(text_metrics['val_loss'])}")
    print(f"Vision-Only Fine-Tuning val_loss: {len(vision_metrics['val_loss'])}")

    # Determine the number of points to plot based on non-empty lists
    max_len = max(
        len(full_metrics["val_loss"]),
        len(text_metrics["val_loss"]),
        len(vision_metrics["val_loss"])
    ) if any(
        [len(full_metrics["val_loss"]), len(text_metrics["val_loss"]), len(vision_metrics["val_loss"])]) else k_folds
    folds = range(1, max_len + 1)

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 2, 1)
    if len(full_metrics["val_loss"]) > 0:
        plt.plot(folds[:len(full_metrics["val_loss"])], full_metrics["val_loss"], label="Full Fine-Tuning Val Loss",
                 marker="o")
    if len(text_metrics["val_loss"]) > 0:
        plt.plot(folds[:len(text_metrics["val_loss"])], text_metrics["val_loss"], label="Text Encoder Val Loss",
                 marker="x")
    if len(vision_metrics["val_loss"]) > 0:
        plt.plot(folds[:len(vision_metrics["val_loss"])], vision_metrics["val_loss"], label="Vision-Only Val Loss",
                 marker="s")
    plt.xlabel("Fold")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison Across Folds")
    plt.legend()
    plt.subplot(3, 2, 2)
    if len(full_metrics["test_loss"]) > 0:
        plt.plot(folds[:len(full_metrics["test_loss"])], full_metrics["test_loss"], label="Full Fine-Tuning Test Loss",
                 marker="o")
    if len(text_metrics["test_loss"]) > 0:
        plt.plot(folds[:len(text_metrics["test_loss"])], text_metrics["test_loss"], label="Text Encoder Test Loss",
                 marker="x")
    if len(vision_metrics["test_loss"]) > 0:
        plt.plot(folds[:len(vision_metrics["test_loss"])], vision_metrics["test_loss"], label="Vision-Only Test Loss",
                 marker="s")
    plt.xlabel("Fold")
    plt.ylabel("Test Loss")
    plt.title("Test Loss Comparison Across Folds")
    plt.legend()
    plt.subplot(3, 2, 3)
    if len(full_metrics["precision"]) > 0:
        plt.plot(folds[:len(full_metrics["precision"])], full_metrics["precision"], label="Full Fine-Tuning Precision",
                 marker="o")
    if len(text_metrics["precision"]) > 0:
        plt.plot(folds[:len(text_metrics["precision"])], text_metrics["precision"], label="Text Encoder Precision",
                 marker="x")
    if len(vision_metrics["precision"]) > 0:
        plt.plot(folds[:len(vision_metrics["precision"])], vision_metrics["precision"], label="Vision-Only Precision",
                 marker="s")
    plt.xlabel("Fold")
    plt.ylabel("Precision (Val)")
    plt.title("Precision Comparison Across Folds (Val)")
    plt.legend()
    plt.subplot(3, 2, 4)
    if len(full_metrics["recall"]) > 0:
        plt.plot(folds[:len(full_metrics["recall"])], full_metrics["recall"], label="Full Fine-Tuning Recall",
                 marker="o")
    if len(text_metrics["recall"]) > 0:
        plt.plot(folds[:len(text_metrics["recall"])], text_metrics["recall"], label="Text Encoder Recall", marker="x")
    if len(vision_metrics["recall"]) > 0:
        plt.plot(folds[:len(vision_metrics["recall"])], vision_metrics["recall"], label="Vision-Only Recall",
                 marker="s")
    plt.xlabel("Fold")
    plt.ylabel("Recall (Val)")
    plt.title("Recall Comparison Across Folds (Val)")
    plt.legend()
    plt.subplot(3, 2, 5)
    if len(full_metrics["f1"]) > 0:
        plt.plot(folds[:len(full_metrics["f1"])], full_metrics["f1"], label="Full Fine-Tuning F1-Score", marker="o")
    if len(text_metrics["f1"]) > 0:
        plt.plot(folds[:len(text_metrics["f1"])], text_metrics["f1"], label="Text Encoder F1-Score", marker="x")
    if len(vision_metrics["f1"]) > 0:
        plt.plot(folds[:len(vision_metrics["f1"])], vision_metrics["f1"], label="Vision-Only F1-Score", marker="s")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score (Val)")
    plt.title("F1-Score Comparison Across Folds (Val)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_metrics_all.png")
    plt.close()


def create_comparison_table(full_metrics, text_metrics, vision_metrics, k_folds):
    comparison_data = []
    for fold in range(k_folds):
        comparison_data.append({
            'Fold': fold + 1,
            'Full Train Loss': full_metrics['train_loss'][fold] if fold < len(full_metrics['train_loss']) else 0,
            'Full Val Loss': full_metrics['val_loss'][fold] if fold < len(full_metrics['val_loss']) else 0,
            'Full Test Loss': full_metrics['test_loss'][fold] if fold < len(full_metrics['test_loss']) else 0,
            'Full Precision (Val)': full_metrics['precision'][fold] if fold < len(full_metrics['precision']) else 0,
            'Full Recall (Val)': full_metrics['recall'][fold] if fold < len(full_metrics['recall']) else 0,
            'Full F1-Score (Val)': full_metrics['f1'][fold] if fold < len(full_metrics['f1']) else 0,
            'Text Train Loss': text_metrics['train_loss'][fold] if fold < len(text_metrics['train_loss']) else 0,
            'Text Val Loss': text_metrics['val_loss'][fold] if fold < len(text_metrics['val_loss']) else 0,
            'Text Test Loss': text_metrics['test_loss'][fold] if fold < len(text_metrics['test_loss']) else 0,
            'Text Precision (Val)': text_metrics['precision'][fold] if fold < len(text_metrics['precision']) else 0,
            'Text Recall (Val)': text_metrics['recall'][fold] if fold < len(text_metrics['recall']) else 0,
            'Text F1-Score (Val)': text_metrics['f1'][fold] if fold < len(text_metrics['f1']) else 0,
            'Vision Train Loss': vision_metrics['train_loss'][fold] if fold < len(vision_metrics['train_loss']) else 0,
            'Vision Val Loss': vision_metrics['val_loss'][fold] if fold < len(vision_metrics['val_loss']) else 0,
            'Vision Test Loss': vision_metrics['test_loss'][fold] if fold < len(vision_metrics['test_loss']) else 0,
            'Vision Precision (Val)': vision_metrics['precision'][fold] if fold < len(
                vision_metrics['precision']) else 0,
            'Vision Recall (Val)': vision_metrics['recall'][fold] if fold < len(vision_metrics['recall']) else 0,
            'Vision F1-Score (Val)': vision_metrics['f1'][fold] if fold < len(vision_metrics['f1']) else 0
        })
    comparison_data.append({
        'Fold': 'Average',
        'Full Train Loss': np.mean(full_metrics['train_loss']) if full_metrics['train_loss'] else 0,
        'Full Val Loss': np.mean(full_metrics['val_loss']) if full_metrics['val_loss'] else 0,
        'Full Test Loss': np.mean(full_metrics['test_loss']) if full_metrics['test_loss'] else 0,
        'Full Precision (Val)': np.mean(full_metrics['precision']) if full_metrics['precision'] else 0,
        'Full Recall (Val)': np.mean(full_metrics['recall']) if full_metrics['recall'] else 0,
        'Full F1-Score (Val)': np.mean(full_metrics['f1']) if full_metrics['f1'] else 0,
        'Text Train Loss': np.mean(text_metrics['train_loss']) if text_metrics['train_loss'] else 0,
        'Text Val Loss': np.mean(text_metrics['val_loss']) if text_metrics['val_loss'] else 0,
        'Text Test Loss': np.mean(text_metrics['test_loss']) if text_metrics['test_loss'] else 0,
        'Text Precision (Val)': np.mean(text_metrics['precision']) if text_metrics['precision'] else 0,
        'Text Recall (Val)': np.mean(text_metrics['recall']) if text_metrics['recall'] else 0,
        'Text F1-Score (Val)': np.mean(text_metrics['f1']) if text_metrics['f1'] else 0,
        'Vision Train Loss': np.mean(vision_metrics['train_loss']) if vision_metrics['train_loss'] else 0,
        'Vision Val Loss': np.mean(vision_metrics['val_loss']) if vision_metrics['val_loss'] else 0,
        'Vision Test Loss': np.mean(vision_metrics['test_loss']) if vision_metrics['test_loss'] else 0,
        'Vision Precision (Val)': np.mean(vision_metrics['precision']) if vision_metrics['precision'] else 0,
        'Vision Recall (Val)': np.mean(vision_metrics['recall']) if vision_metrics['recall'] else 0,
        'Vision F1-Score (Val)': np.mean(vision_metrics['f1']) if vision_metrics['f1'] else 0
    })
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("comparison_table_all.csv", index=False)
    print("\nComparison Table of Metrics Across Folds (All Approaches):")
    print(comparison_df)
    return comparison_df


import json

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

    # Metrics storage for all approaches
    full_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}
    text_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}
    vision_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}

    # Checkpoint file path
    checkpoint_file = "checkpoint.json"

    # Load checkpoint if it exists
    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: Resuming from Fold {checkpoint['current_fold'] + 1}")
            # Load metrics from checkpoint if available
            if 'full_fold_metrics' in checkpoint:
                full_fold_metrics = checkpoint['full_fold_metrics']
            if 'text_fold_metrics' in checkpoint:
                text_fold_metrics = checkpoint['text_fold_metrics']
            if 'vision_fold_metrics' in checkpoint:
                vision_fold_metrics = checkpoint['vision_fold_metrics']
            print("Metrics loaded from checkpoint:")
            print("Full Fine-Tuning Metrics:", {k: len(v) for k, v in full_fold_metrics.items()})
            print("Text Encoder Fine-Tuning Metrics:", {k: len(v) for k, v in text_fold_metrics.items()})
            print("Vision-Only Fine-Tuning Metrics:", {k: len(v) for k, v in vision_fold_metrics.items()})
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    # Set number of workers based on CPU cores
    num_workers = min(8, os.cpu_count())
    print(f"Using {num_workers} workers for data loading.")

    # Process folds, starting from the checkpoint's current fold
    for fold, (train_val_indices, test_indices) in enumerate(kf.split(dataset)):
        if fold < checkpoint['current_fold']:
            print(f"\nSkipping Fold {fold + 1}/{k_folds} - Already completed (from checkpoint)")
            continue

        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
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

        # Initialize completed approaches for this fold if not in checkpoint
        if str(fold) not in checkpoint['completed_approaches']:
            checkpoint['completed_approaches'][str(fold)] = []

        # 1. Full Fine-Tuning (with frozen visual encoder and classifier head)
        if 'full' not in checkpoint['completed_approaches'][str(fold)]:
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
            full_best_model_path = os.path.join(full_log_dir, f'best_full_fine-tuning_clip.pth')
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
            # Append metrics only if not already loaded or present
            if len(full_fold_metrics["val_loss"]) <= fold:
                full_fold_metrics["train_loss"].append(full_metrics_dict['train_loss'][-1])
                full_fold_metrics["val_loss"].append(full_val_loss)
                full_fold_metrics["test_loss"].append(full_test_loss)
                full_fold_metrics["precision"].append(full_precision)
                full_fold_metrics["recall"].append(full_recall)
                full_fold_metrics["f1"].append(full_f1)
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
            # Update checkpoint after completing Full Fine-Tuning
            checkpoint['completed_approaches'][str(fold)].append('full')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Full Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Full Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

        # 2. Text Encoder Fine-Tuning
        if 'text' not in checkpoint['completed_approaches'][str(fold)]:
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
            text_best_model_path = os.path.join(text_log_dir, f'best_text_encoder_fine-tuning_clip.pth')
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
                text_model, test_loader, criterion, subcategories, device, fold + 1, "Final",
                "Text Encoder Fine-Tuning",
                split_type="Test"
            )
            # Append metrics only if not already loaded or present
            if len(text_fold_metrics["val_loss"]) <= fold:
                text_fold_metrics["train_loss"].append(text_metrics_dict['train_loss'][-1])
                text_fold_metrics["val_loss"].append(text_val_loss)
                text_fold_metrics["test_loss"].append(text_test_loss)
                text_fold_metrics["precision"].append(text_precision)
                text_fold_metrics["recall"].append(text_recall)
                text_fold_metrics["f1"].append(text_f1)
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
            # Update checkpoint after completing Text Encoder Fine-Tuning
            checkpoint['completed_approaches'][str(fold)].append('text')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Text Encoder Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Text Encoder Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

        # 3. Vision-Only Fine-Tuning (Train visual encoder and classifier)
        if 'vision' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Vision-Only Fine-Tuned Model...")
            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer_vision = optim.AdamW(vision_model.parameters(), lr=1e-4)
            vision_log_dir = f"logs/3-2/vision_only_finetune/fold_{fold + 1}/"
            vision_metrics_dict = train_model(
                vision_model, train_loader, val_loader, test_loader, criterion, optimizer_vision, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=vision_log_dir, fold=fold + 1,
                accumulation_steps=4, validate_every=2, model_type="Vision-Only Fine-Tuning"
            )
            vision_best_model_path = os.path.join(vision_log_dir, f'best_vision-only_fine-tuning_clip.pth')
            if os.path.exists(vision_best_model_path):
                vision_model.load_state_dict(torch.load(vision_best_model_path))
                print(f"Loaded best model from {vision_best_model_path}")
            else:
                print(f"Best model file not found at {vision_best_model_path}. Using current model weights.")
            vision_val_loss, vision_precision, vision_recall, vision_f1, vision_conf_matrix, vision_val_misclass = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold + 1, "Final",
                "Vision-Only Fine-Tuning",
                split_type="Validation"
            )
            vision_test_loss, vision_test_precision, vision_test_recall, vision_test_f1, vision_test_conf_matrix, vision_test_misclass = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold + 1, "Final",
                "Vision-Only Fine-Tuning",
                split_type="Test"
            )
            # Append metrics only if not already loaded or present
            if len(vision_fold_metrics["val_loss"]) <= fold:
                vision_fold_metrics["train_loss"].append(vision_metrics_dict['train_loss'][-1])
                vision_fold_metrics["val_loss"].append(vision_val_loss)
                vision_fold_metrics["test_loss"].append(vision_test_loss)
                vision_fold_metrics["precision"].append(vision_precision)
                vision_fold_metrics["recall"].append(vision_recall)
                vision_fold_metrics["f1"].append(vision_f1)
            save_metrics_to_csv({
                "epoch": vision_metrics_dict['epoch'],
                "train_loss": vision_metrics_dict['train_loss'],
                "val_loss": vision_metrics_dict['val_loss'],
                "test_loss": vision_metrics_dict['test_loss'],
                "train_acc": vision_metrics_dict['train_acc'],
                "val_acc": vision_metrics_dict['val_acc'],
                "test_acc": vision_metrics_dict['test_acc'],
                "precision_val": vision_metrics_dict['precision_val'],
                "recall_val": vision_metrics_dict['recall_val'],
                "f1_val": vision_metrics_dict['f1_val'],
                "precision_test": vision_metrics_dict['precision_test'],
                "recall_test": vision_metrics_dict['recall_test'],
                "f1_test": vision_metrics_dict['f1_test']
            }, f"metrics_vision_only_fold_{fold + 1}.csv")
            save_logs_as_zip(vision_log_dir, f"logs/3-2/vision_only_finetune/fold_{fold + 1}_logs")
            torch.cuda.empty_cache()
            # Update checkpoint after completing Vision-Only Fine-Tuning
            checkpoint['completed_approaches'][str(fold)].append('vision')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Vision-Only Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Vision-Only Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

    # Debug: Check contents of metrics dictionaries before plotting
    print("\nDebug: Contents of metrics dictionaries before plotting:")
    print("Full Fine-Tuning Metrics:", {k: len(v) for k, v in full_fold_metrics.items()})
    print("Text Encoder Fine-Tuning Metrics:", {k: len(v) for k, v in text_fold_metrics.items()})
    print("Vision-Only Fine-Tuning Metrics:", {k: len(v) for k, v in vision_fold_metrics.items()})

    # Plot and save metrics for all approaches
    plot_fold_metrics(full_fold_metrics, k_folds, title_suffix="Full Fine-Tuning")
    plot_fold_metrics(text_fold_metrics, k_folds, title_suffix="Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold_metrics, k_folds, title_suffix="Vision-Only Fine-Tuning")
    compare_metrics(full_fold_metrics, text_fold_metrics, vision_fold_metrics, k_folds)
    save_metrics_to_csv(full_fold_metrics, "kfold_full_finetune_metrics.csv")
    save_metrics_to_csv(text_fold_metrics, "kfold_text_encoder_metrics.csv")
    save_metrics_to_csv(vision_fold_metrics, "kfold_vision_only_metrics.csv")

    # Create comparison table for all approaches
    create_comparison_table(full_fold_metrics, text_fold_metrics, vision_fold_metrics, k_folds)

    # Summary of average metrics across folds for all approaches
    print("\nSummary of Average Metrics Across Folds:")
    print("\nFull Fine-Tuning:")
    print(f"Avg Train Loss: {np.mean(full_fold_metrics['train_loss']) if full_fold_metrics['train_loss'] else 0:.4f}")
    print(f"Avg Validation Loss: {np.mean(full_fold_metrics['val_loss']) if full_fold_metrics['val_loss'] else 0:.4f}")
    print(f"Avg Test Loss: {np.mean(full_fold_metrics['test_loss']) if full_fold_metrics['test_loss'] else 0:.4f}")
    print(
        f"Avg Precision (Val): {np.mean(full_fold_metrics['precision']) if full_fold_metrics['precision'] else 0:.4f}")
    print(f"Avg Recall (Val): {np.mean(full_fold_metrics['recall']) if full_fold_metrics['recall'] else 0:.4f}")
    print(f"Avg F1-Score (Val): {np.mean(full_fold_metrics['f1']) if full_fold_metrics['f1'] else 0:.4f}")
    print("\nText Encoder Fine-Tuning:")
    print(f"Avg Train Loss: {np.mean(text_fold_metrics['train_loss']) if text_fold_metrics['train_loss'] else 0:.4f}")
    print(f"Avg Validation Loss: {np.mean(text_fold_metrics['val_loss']) if text_fold_metrics['val_loss'] else 0:.4f}")
    print(f"Avg Test Loss: {np.mean(text_fold_metrics['test_loss']) if text_fold_metrics['test_loss'] else 0:.4f}")
    print(
        f"Avg Precision (Val): {np.mean(text_fold_metrics['precision']) if text_fold_metrics['precision'] else 0:.4f}")
    print(f"Avg Recall (Val): {np.mean(text_fold_metrics['recall']) if text_fold_metrics['recall'] else 0:.4f}")
    print(f"Avg F1-Score (Val): {np.mean(text_fold_metrics['f1']) if text_fold_metrics['f1'] else 0:.4f}")
    print("\nVision-Only Fine-Tuning:")
    print(
        f"Avg Train Loss: {np.mean(vision_fold_metrics['train_loss']) if vision_fold_metrics['train_loss'] else 0:.4f}")
    print(
        f"Avg Validation Loss: {np.mean(vision_fold_metrics['val_loss']) if vision_fold_metrics['val_loss'] else 0:.4f}")
    print(f"Avg Test Loss: {np.mean(vision_fold_metrics['test_loss']) if vision_fold_metrics['test_loss'] else 0:.4f}")
    print(
        f"Avg Precision (Val): {np.mean(vision_fold_metrics['precision']) if vision_fold_metrics['precision'] else 0:.4f}")
    print(f"Avg Recall (Val): {np.mean(vision_fold_metrics['recall']) if vision_fold_metrics['recall'] else 0:.4f}")
    print(f"Avg F1-Score (Val): {np.mean(vision_fold_metrics['f1']) if vision_fold_metrics['f1'] else 0:.4f}")
    print("\nK-Fold training complete. Results saved for analysis.")