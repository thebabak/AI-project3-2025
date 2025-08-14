import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import seaborn as sns
from torch.multiprocessing import freeze_support
import open_clip
from datasets import load_dataset
import json
from zipfile import ZipFile
import shutil
from torch.cuda.amp import autocast
from torch.amp import GradScaler
import torch.optim as optim
import time
from tqdm import tqdm
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation")
warnings.filterwarnings("ignore", message="Repo card metadata block was not found")


# Custom Dataset Class
class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subcategories, augment=False):
        self.dataset = dataset
        self.subcategories = subcategories
        self.augment = augment
        self.transform = self._get_transforms()

    def _get_transforms(self):
        from torchvision import transforms
        if self.augment:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = self.subcategories.index(item['subCategory'])
        image = self.transform(image)
        return image, label


# Model Classes
class FullFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.is_encoder_frozen = freeze_encoder
        if freeze_encoder:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]
        layers = []
        input_size = output_size
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            input_size = 512
        layers.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        features = self.visual_encoder(images)
        logits = self.classifier(features)
        return logits


class TextEncoderFineTunedCLIP(nn.Module):
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder = base_model.transformer
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device
        self.subcategories = subcategories
        if freeze_visual:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
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

    def forward(self, images, text_inputs=None):
        image_features = self.visual_encoder(images)
        if text_inputs is None:
            text_inputs = self.tokenizer([f"a photo of {cat}" for cat in self.subcategories]).to(self.device)
        text_features = self.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_features, text_features.T) * 100.0
        return logits


class VisionOnlyFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        for param in self.visual_encoder.parameters():
            param.requires_grad = True
        dummy_input = torch.randn(1, 3, 224, 224).to(next(base_model.parameters()).device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]
        layers = []
        input_size = output_size
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            input_size = 512
        layers.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        features = self.visual_encoder(images)
        logits = self.classifier(features)
        return logits


class PartialFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder = base_model.transformer
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = next(base_model.parameters()).device
        self.num_classes = num_classes

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]

        self.classifier = nn.Linear(output_size, num_classes)

        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder, freeze_percentage, "text")

        self.text_projection = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final = base_model.ln_final
        self.token_embedding = base_model.token_embedding

    def _freeze_layers(self, module, freeze_percentage, encoder_type):
        named_params = list(module.named_parameters())
        total_params = len(named_params)
        freeze_until = int(total_params * freeze_percentage)

        print(
            f"Freezing first {freeze_percentage * 100:.1f}% of {encoder_type} encoder layers ({freeze_until}/{total_params} parameters)")
        for i, (name, param) in enumerate(named_params):
            if i < freeze_until:
                param.requires_grad = False
                print(f"Frozen {name}")
            else:
                param.requires_grad = True
                print(f"Trainable {name}")
        frozen_params = sum(p.numel() for n, p in named_params[:freeze_until])
        trainable_params = sum(p.numel() for n, p in named_params[freeze_until:])
        print(
            f"{encoder_type.capitalize()} Encoder: Frozen {frozen_params} params, Trainable {trainable_params} params")

    def encode_text(self, text_inputs):
        x = self.token_embedding(text_inputs)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text_inputs.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images, text_inputs=None):
        image_features = self.visual_encoder(images)
        if text_inputs is not None:
            text_features = self.encode_text(text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logits = torch.matmul(image_features, text_features.T) * 100.0
            return logits
        return self.classifier(image_features)


# Utility Functions
def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_weights)
    model = model.to(device)
    model.eval()
    return model, preprocess, preprocess


def load_fashion_dataset():
    dataset = load_dataset("ceyda/fashion-products-small")
    train_data = dataset['train']
    subcategories = sorted(set(item['subCategory'] for item in train_data))
    print(f"Loaded dataset with {len(subcategories)} subcategories.")
    return train_data, subcategories


def compute_class_weights(dataset, subcategories):
    labels = [subcategories.index(item['subCategory']) for item in dataset]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    weights = total_samples / (len(subcategories) * class_counts)
    return torch.tensor(weights, dtype=torch.float)


def save_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")


def save_logs_as_zip(log_dir, zip_name):
    if os.path.exists(log_dir):
        with ZipFile(f"{zip_name}.zip", 'w') as zipf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.join(root.replace(log_dir, log_dir.split('/')[-2]), file))
        print(f"Logs zipped to {zip_name}.zip")
        shutil.rmtree(log_dir, ignore_errors=True)


def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    print(f"Metrics for {title_suffix}:")
    for key, value in metrics.items():
        print(f"{key}: {value} (length: {len(value)})")
    metric_lengths = [len(metrics[key]) for key in metrics if len(metrics[key]) > 0]
    if not metric_lengths:
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
    save_path = os.path.join("logs/main3-4", f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def compare_metrics(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
    print("Metrics lengths for comparison:")
    print(f"Full Fine-Tuning val_loss: {len(full_metrics['val_loss'])}")
    print(f"Text Encoder Fine-Tuning val_loss: {len(text_metrics['val_loss'])}")
    print(f"Vision-Only Fine-Tuning val_loss: {len(vision_metrics['val_loss'])}")
    print(f"Partial Fine-Tuning val_loss: {len(partial_metrics['val_loss'])}")
    max_len = max(
        len(full_metrics["val_loss"]),
        len(text_metrics["val_loss"]),
        len(vision_metrics["val_loss"]),
        len(partial_metrics["val_loss"])
    ) if any([len(full_metrics["val_loss"]), len(text_metrics["val_loss"]), len(vision_metrics["val_loss"]),
              len(partial_metrics["val_loss"])]) else k_folds
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
    if len(partial_metrics["val_loss"]) > 0:
        plt.plot(folds[:len(partial_metrics["val_loss"])], partial_metrics["val_loss"],
                 label="Partial Fine-Tuning Val Loss", marker="d")
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
    if len(partial_metrics["test_loss"]) > 0:
        plt.plot(folds[:len(partial_metrics["test_loss"])], partial_metrics["test_loss"],
                 label="Partial Fine-Tuning Test Loss", marker="d")
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
    if len(partial_metrics["precision"]) > 0:
        plt.plot(folds[:len(partial_metrics["precision"])], partial_metrics["precision"],
                 label="Partial Fine-Tuning Precision", marker="d")
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
    if len(partial_metrics["recall"]) > 0:
        plt.plot(folds[:len(partial_metrics["recall"])], partial_metrics["recall"], label="Partial Fine-Tuning Recall",
                 marker="d")
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
    if len(partial_metrics["f1"]) > 0:
        plt.plot(folds[:len(partial_metrics["f1"])], partial_metrics["f1"], label="Partial Fine-Tuning F1-Score",
                 marker="d")
    plt.xlabel("Fold")
    plt.ylabel("F1-Score (Val)")
    plt.title("F1-Score Comparison Across Folds (Val)")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join("logs/main3-4", "comparison_metrics_all.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")


def create_comparison_table(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
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
            'Vision F1-Score (Val)': vision_metrics['f1'][fold] if fold < len(vision_metrics['f1']) else 0,
            'Partial Train Loss': partial_metrics['train_loss'][fold] if fold < len(
                partial_metrics['train_loss']) else 0,
            'Partial Val Loss': partial_metrics['val_loss'][fold] if fold < len(partial_metrics['val_loss']) else 0,
            'Partial Test Loss': partial_metrics['test_loss'][fold] if fold < len(partial_metrics['test_loss']) else 0,
            'Partial Precision (Val)': partial_metrics['precision'][fold] if fold < len(
                partial_metrics['precision']) else 0,
            'Partial Recall (Val)': partial_metrics['recall'][fold] if fold < len(partial_metrics['recall']) else 0,
            'Partial F1-Score (Val)': partial_metrics['f1'][fold] if fold < len(partial_metrics['f1']) else 0
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
        'Vision F1-Score (Val)': np.mean(vision_metrics['f1']) if vision_metrics['f1'] else 0,
        'Partial Train Loss': np.mean(partial_metrics['train_loss']) if partial_metrics['train_loss'] else 0,
        'Partial Val Loss': np.mean(partial_metrics['val_loss']) if partial_metrics['val_loss'] else 0,
        'Partial Test Loss': np.mean(partial_metrics['test_loss']) if partial_metrics['test_loss'] else 0,
        'Partial Precision (Val)': np.mean(partial_metrics['precision']) if partial_metrics['precision'] else 0,
        'Partial Recall (Val)': np.mean(partial_metrics['recall']) if partial_metrics['recall'] else 0,
        'Partial F1-Score (Val)': np.mean(partial_metrics['f1']) if partial_metrics['f1'] else 0
    })
    comparison_df = pd.DataFrame(comparison_data)
    save_path = os.path.join("logs/main3-4", "comparison_table_all.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    print(f"\nComparison Table saved to {save_path}")
    print(comparison_df)
    return comparison_df


def analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]  # Use the actual size of the confusion matrix
    expected_classes = len(subcategories)
    if num_classes != expected_classes:
        print(
            f"Warning: Confusion matrix size ({num_classes}x{num_classes}) does not match number of subcategories ({expected_classes}). Some classes may not be present in {split_type} data.")

    misclassifications = []
    for true_idx in range(num_classes):
        total_true = np.sum(conf_matrix[true_idx])
        if total_true == 0:
            continue
        for pred_idx in range(num_classes):
            if true_idx != pred_idx and conf_matrix[true_idx][pred_idx] > 0:
                count = conf_matrix[true_idx][pred_idx]
                percentage = (count / total_true) * 100
                # Ensure true_idx and pred_idx are within bounds of subcategories
                true_label = subcategories[true_idx] if true_idx < len(subcategories) else f"Unknown_{true_idx}"
                pred_label = subcategories[pred_idx] if pred_idx < len(subcategories) else f"Unknown_{pred_idx}"
                misclassifications.append({
                    'True Class': true_label,
                    'Predicted Class': pred_label,
                    'Count': count,
                    'Percentage of True Class': percentage
                })
    if misclassifications:
        misclass_df = pd.DataFrame(misclassifications)
        misclass_df = misclass_df.sort_values(by='Count', ascending=False).head(10)
        print(f"\nSignificant Misclassifications for {model_type} (Fold {fold}, Epoch {epoch}, {split_type}):")
        print(misclass_df)
        save_path = os.path.join("logs/main3-4", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                                 f"misclassifications_epoch_{epoch}_{split_type.lower()}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        misclass_df.to_csv(save_path, index=False)
        print(f"Misclassifications saved to {save_path}")
        return misclass_df
    return None


def plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    expected_classes = len(subcategories)
    if num_classes != expected_classes:
        print(
            f"Warning: Confusion matrix size ({num_classes}x{num_classes}) does not match number of subcategories ({expected_classes}) for plotting. Using subset of labels.")
        labels = subcategories[:num_classes] if num_classes <= len(subcategories) else [f"Class_{i}" for i in
                                                                                        range(num_classes)]
    else:
        labels = subcategories

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join("logs/main3-4", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                             f"confusion_matrix_epoch_{epoch}_{split_type.lower()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix plot saved to {save_path}")


def evaluate_model_with_metrics(model, dataloader, criterion, subcategories, device, fold, epoch, model_type,
                                split_type="Validation"):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Debug: Check unique classes in labels and predictions
    unique_labels = set(all_labels)
    unique_preds = set(all_preds)
    print(
        f"Debug {split_type} (Fold {fold}, Epoch {epoch}): Unique labels={len(unique_labels)}/{len(subcategories)}, Unique preds={len(unique_preds)}/{len(subcategories)}")
    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted',
                                                               zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type)
    misclass_df = analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type)
    print(
        f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): Loss={avg_loss:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, precision, recall, f1, conf_matrix, misclass_df


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcategories, device,
                log_dir, fold, accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
    scaler = GradScaler('cuda')  # Updated to new syntax for mixed precision training
    best_f1 = 0.0
    patience = 3
    patience_counter = 0
    metrics = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_acc': [], 'val_acc': [], 'test_acc': [],
        'precision_val': [], 'recall_val': [], 'f1_val': [],
        'precision_test': [], 'recall_test': [], 'f1_test': []
    }
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        optimizer.zero_grad()
        # Add tqdm progress bar for batch loop
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                total_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)
                # Update progress bar with current loss and accuracy
                running_train_acc = correct_train / total_train
                running_train_loss = total_train_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{running_train_loss:.4f}", "acc": f"{running_train_acc:.4f}"})
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            val_loss, val_precision, val_recall, val_f1, val_conf_matrix, _ = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Validation"
            )
            test_loss, test_precision, test_recall, test_f1, test_conf_matrix, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Test"
            )
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['test_loss'].append(test_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(val_precision)  # Placeholder, adjust if accuracy is separate
            metrics['test_acc'].append(test_precision)
            metrics['precision_val'].append(val_precision)
            metrics['recall_val'].append(val_recall)
            metrics['f1_val'].append(val_f1)
            metrics['precision_test'].append(test_precision)
            metrics['recall_test'].append(test_recall)
            metrics['f1_test'].append(test_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                save_path = os.path.join(log_dir, f'best_{model_type.lower().replace(" ", "_")}_clip.pth')
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at {save_path} with F1={best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        else:
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(0.0)
            metrics['test_loss'].append(0.0)
            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(0.0)
            metrics['test_acc'].append(0.0)
            metrics['precision_val'].append(0.0)
            metrics['recall_val'].append(0.0)
            metrics['f1_val'].append(0.0)
            metrics['precision_test'].append(0.0)
            metrics['recall_test'].append(0.0)
            metrics['f1_test'].append(0.0)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    return metrics


if __name__ == '__main__':
    freeze_support()
    # Record overall start time
    overall_start_time = time.time()
    print(f"\nOverall Training Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")

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
    partial_fold_metrics = {"train_loss": [], "val_loss": [], "test_loss": [], "precision": [], "recall": [], "f1": []}

    # Dictionary to store timing for each approach across folds for comparison
    timing_results = {
        "Full Fine-Tuning": [],
        "Text Encoder Fine-Tuning": [],
        "Vision-Only Fine-Tuning": [],
        "Partial Fine-Tuning": []
    }

    # Checkpoint file path
    checkpoint_file = os.path.join("logs/main3-4", "checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    # Load checkpoint if it exists
    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: Resuming from Fold {checkpoint['current_fold'] + 1}")
            if 'full_fold_metrics' in checkpoint:
                full_fold_metrics = checkpoint['full_fold_metrics']
            if 'text_fold_metrics' in checkpoint:
                text_fold_metrics = checkpoint['text_fold_metrics']
            if 'vision_fold_metrics' in checkpoint:
                vision_fold_metrics = checkpoint['vision_fold_metrics']
            if 'partial_fold_metrics' in checkpoint:
                partial_fold_metrics = checkpoint['partial_fold_metrics']
            if 'timing_results' in checkpoint:
                timing_results = checkpoint['timing_results']
            print("Metrics loaded from checkpoint:")
            print("Full Fine-Tuning Metrics:", {k: len(v) for k, v in full_fold_metrics.items()})
            print("Text Encoder Fine-Tuning Metrics:", {k: len(v) for k, v in text_fold_metrics.items()})
            print("Vision-Only Fine-Tuning Metrics:", {k: len(v) for k, v in vision_fold_metrics.items()})
            print("Partial Fine-Tuning Metrics:", {k: len(v) for k, v in partial_fold_metrics.items()})
            print("Timing Results Loaded:", {k: len(v) for k, v in timing_results.items()})
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    # Set number of workers based on CPU cores
    num_workers = min(8, os.cpu_count())
    print(f"Using {num_workers} workers for data loading.")

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

        if str(fold) not in checkpoint['completed_approaches']:
            checkpoint['completed_approaches'][str(fold)] = []

        # 1. Full Fine-Tuning
        if 'full' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Full Fine-Tuned Model...")
            start_time = time.time()
            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            params_to_optimize_full = full_model.classifier.parameters() if full_model.is_encoder_frozen else full_model.parameters()
            optimizer_full = optim.AdamW(params_to_optimize_full, lr=1e-4)
            full_log_dir = os.path.join("logs/main3-4", "full_finetune", f"fold_{fold + 1}")
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
            if len(full_fold_metrics["val_loss"]) <= fold:
                full_fold_metrics["train_loss"].append(full_metrics_dict['train_loss'][-1])
                full_fold_metrics["val_loss"].append(full_val_loss)
                full_fold_metrics["test_loss"].append(full_test_loss)
                full_fold_metrics["precision"].append(full_precision)
                full_fold_metrics["recall"].append(full_recall)
                full_fold_metrics["f1"].append(full_f1)
            save_metrics_to_csv({
                "epoch": full_metrics_dict['epoch'], "train_loss": full_metrics_dict['train_loss'],
                "val_loss": full_metrics_dict['val_loss'], "test_loss": full_metrics_dict['test_loss'],
                "train_acc": full_metrics_dict['train_acc'], "val_acc": full_metrics_dict['val_acc'],
                "test_acc": full_metrics_dict['test_acc'], "precision_val": full_metrics_dict['precision_val'],
                "recall_val": full_metrics_dict['recall_val'], "f1_val": full_metrics_dict['f1_val'],
                "precision_test": full_metrics_dict['precision_test'], "recall_test": full_metrics_dict['recall_test'],
                "f1_test": full_metrics_dict['f1_test']
            }, os.path.join("logs/main3-4", f"metrics_full_finetune_fold_{fold + 1}.csv"))
            save_logs_as_zip(full_log_dir, os.path.join("logs/main3-4", "full_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()
            end_time = time.time()
            duration = end_time - start_time
            timing_results["Full Fine-Tuning"].append(duration)
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Full Fine-Tuning (Fold {fold + 1}) took {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            checkpoint['completed_approaches'][str(fold)].append('full')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results'] = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Full Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Full Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")
            if fold < len(timing_results["Full Fine-Tuning"]) and timing_results["Full Fine-Tuning"][fold] > 0:
                duration = timing_results["Full Fine-Tuning"][fold]
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Previously recorded time for Full Fine-Tuning Fold {fold + 1}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        # 2. Text Encoder Fine-Tuning
        if 'text' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Text Encoder Fine-Tuned Model...")
            start_time = time.time()
            text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer_text = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)
            text_log_dir = os.path.join("logs/main3-4", "text_encoder_finetune", f"fold_{fold + 1}")
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
                "Text Encoder Fine-Tuning", split_type="Test"
            )
            if len(text_fold_metrics["val_loss"]) <= fold:
                text_fold_metrics["train_loss"].append(text_metrics_dict['train_loss'][-1])
                text_fold_metrics["val_loss"].append(text_val_loss)
                text_fold_metrics["test_loss"].append(text_test_loss)
                text_fold_metrics["precision"].append(text_precision)
                text_fold_metrics["recall"].append(text_recall)
                text_fold_metrics["f1"].append(text_f1)
            save_metrics_to_csv({
                "epoch": text_metrics_dict['epoch'], "train_loss": text_metrics_dict['train_loss'],
                "val_loss": text_metrics_dict['val_loss'], "test_loss": text_metrics_dict['test_loss'],
                "train_acc": text_metrics_dict['train_acc'], "val_acc": text_metrics_dict['val_acc'],
                "test_acc": text_metrics_dict['test_acc'], "precision_val": text_metrics_dict['precision_val'],
                "recall_val": text_metrics_dict['recall_val'], "f1_val": text_metrics_dict['f1_val'],
                "precision_test": text_metrics_dict['precision_test'], "recall_test": text_metrics_dict['recall_test'],
                "f1_test": text_metrics_dict['f1_test']
            }, os.path.join("logs/main3-4", f"metrics_text_encoder_fold_{fold + 1}.csv"))
            save_logs_as_zip(text_log_dir,
                             os.path.join("logs/main3-4", "text_encoder_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()
            end_time = time.time()
            duration = end_time - start_time
            timing_results["Text Encoder Fine-Tuning"].append(duration)
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"Text Encoder Fine-Tuning (Fold {fold + 1}) took {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            checkpoint['completed_approaches'][str(fold)].append('text')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results'] = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Text Encoder Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Text Encoder Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")
            if fold < len(timing_results["Text Encoder Fine-Tuning"]) and timing_results["Text Encoder Fine-Tuning"][
                fold] > 0:
                duration = timing_results["Text Encoder Fine-Tuning"][fold]
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Previously recorded time for Text Encoder Fine-Tuning Fold {fold + 1}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        # 3. Vision-Only Fine-Tuning
        if 'vision' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Vision-Only Fine-Tuned Model...")
            start_time = time.time()
            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer_vision = optim.AdamW(vision_model.parameters(), lr=1e-4)
            vision_log_dir = os.path.join("logs/main3-4", "vision_only_finetune", f"fold_{fold + 1}")
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
                "Vision-Only Fine-Tuning", split_type="Validation"
            )
            vision_test_loss, vision_test_precision, vision_test_recall, vision_test_f1, vision_test_conf_matrix, vision_test_misclass = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold + 1, "Final",
                "Vision-Only Fine-Tuning", split_type="Test"
            )
            if len(vision_fold_metrics["val_loss"]) <= fold:
                vision_fold_metrics["train_loss"].append(vision_metrics_dict['train_loss'][-1])
                vision_fold_metrics["val_loss"].append(vision_val_loss)
                vision_fold_metrics["test_loss"].append(vision_test_loss)
                vision_fold_metrics["precision"].append(vision_precision)
                vision_fold_metrics["recall"].append(vision_recall)
                vision_fold_metrics["f1"].append(vision_f1)
            save_metrics_to_csv({
                "epoch": vision_metrics_dict['epoch'], "train_loss": vision_metrics_dict['train_loss'],
                "val_loss": vision_metrics_dict['val_loss'], "test_loss": vision_metrics_dict['test_loss'],
                "train_acc": vision_metrics_dict['train_acc'], "val_acc": vision_metrics_dict['val_acc'],
                "test_acc": vision_metrics_dict['test_acc'], "precision_val": vision_metrics_dict['precision_val'],
                "recall_val": vision_metrics_dict['recall_val'], "f1_val": vision_metrics_dict['f1_val'],
                "precision_test": vision_metrics_dict['precision_test'],
                "recall_test": vision_metrics_dict['recall_test'],
                "f1_test": vision_metrics_dict['f1_test']
            }, os.path.join("logs/main3-4", f"metrics_vision_only_fold_{fold + 1}.csv"))
            save_logs_as_zip(vision_log_dir,
                             os.path.join("logs/main3-4", "vision_only_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()
            end_time = time.time()
            duration = end_time - start_time
            timing_results["Vision-Only Fine-Tuning"].append(duration)
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"Vision-Only Fine-Tuning (Fold {fold + 1}) took {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            checkpoint['completed_approaches'][str(fold)].append('vision')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results'] = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Vision-Only Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Vision-Only Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")
            if fold < len(timing_results["Vision-Only Fine-Tuning"]) and timing_results["Vision-Only Fine-Tuning"][
                fold] > 0:
                duration = timing_results["Vision-Only Fine-Tuning"][fold]
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Previously recorded time for Vision-Only Fine-Tuning Fold {fold + 1}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

        # 4. Partial Fine-Tuning (Last 30% of Layers)
        if 'partial' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Partial Fine-Tuned Model (Last 30% of Layers)...")
            start_time = time.time()
            partial_model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            params_to_optimize_partial = [p for p in partial_model.parameters() if p.requires_grad]
            optimizer_partial = optim.AdamW(params_to_optimize_partial, lr=1e-4)
            partial_log_dir = os.path.join("logs/main3-4", "partial_finetune", f"fold_{fold + 1}")
            partial_metrics_dict = train_model(
                partial_model, train_loader, val_loader, test_loader, criterion, optimizer_partial, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=partial_log_dir, fold=fold + 1,
                accumulation_steps=4, validate_every=2, model_type="Partial Fine-Tuning"
            )
            partial_best_model_path = os.path.join(partial_log_dir, f'best_partial_fine-tuning_clip.pth')
            if os.path.exists(partial_best_model_path):
                partial_model.load_state_dict(torch.load(partial_best_model_path))
                print(f"Loaded best model from {partial_best_model_path}")
            else:
                print(f"Best model file not found at {partial_best_model_path}. Using current model weights.")
            partial_val_loss, partial_precision, partial_recall, partial_f1, partial_conf_matrix, partial_val_misclass = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Partial Fine-Tuning",
                split_type="Validation"
            )
            partial_test_loss, partial_test_precision, partial_test_recall, partial_test_f1, partial_test_conf_matrix, partial_test_misclass = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Partial Fine-Tuning",
                split_type="Test"
            )
            if len(partial_fold_metrics["val_loss"]) <= fold:
                partial_fold_metrics["train_loss"].append(partial_metrics_dict['train_loss'][-1])
                partial_fold_metrics["val_loss"].append(partial_val_loss)
                partial_fold_metrics["test_loss"].append(partial_test_loss)
                partial_fold_metrics["precision"].append(partial_precision)
                partial_fold_metrics["recall"].append(partial_recall)
                partial_fold_metrics["f1"].append(partial_f1)
            save_metrics_to_csv({
                "epoch": partial_metrics_dict['epoch'], "train_loss": partial_metrics_dict['train_loss'],
                "val_loss": partial_metrics_dict['val_loss'], "test_loss": partial_metrics_dict['test_loss'],
                "train_acc": partial_metrics_dict['train_acc'], "val_acc": partial_metrics_dict['val_acc'],
                "test_acc": partial_metrics_dict['test_acc'], "precision_val": partial_metrics_dict['precision_val'],
                "recall_val": partial_metrics_dict['recall_val'], "f1_val": partial_metrics_dict['f1_val'],
                "precision_test": partial_metrics_dict['precision_test'],
                "recall_test": partial_metrics_dict['recall_test'],
                "f1_test": partial_metrics_dict['f1_test']
            }, os.path.join("logs/main3-4", f"metrics_partial_finetune_fold_{fold + 1}.csv"))
            save_logs_as_zip(partial_log_dir, os.path.join("logs/main3-4", "partial_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()
            end_time = time.time()
            duration = end_time - start_time
            timing_results["Partial Fine-Tuning"].append(duration)
            hours, remainder = divmod(duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Partial Fine-Tuning (Fold {fold + 1}) took {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            checkpoint['completed_approaches'][str(fold)].append('partial')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics'] = full_fold_metrics
            checkpoint['text_fold_metrics'] = text_fold_metrics
            checkpoint['vision_fold_metrics'] = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results'] = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Partial Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Partial Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")
            if fold < len(timing_results["Partial Fine-Tuning"]) and timing_results["Partial Fine-Tuning"][fold] > 0:
                duration = timing_results["Partial Fine-Tuning"][fold]
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(
                    f"Previously recorded time for Partial Fine-Tuning Fold {fold + 1}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    # Debug: Check contents of metrics dictionaries before plotting
    print("\nDebug: Contents of metrics dictionaries before plotting:")
    print("Full Fine-Tuning Metrics:", {k: len(v) for k, v in full_fold_metrics.items()})
    print("Text Encoder Fine-Tuning Metrics:", {k: len(v) for k, v in text_fold_metrics.items()})
    print("Vision-Only Fine-Tuning Metrics:", {k: len(v) for k, v in vision_fold_metrics.items()})
    print("Partial Fine-Tuning Metrics:", {k: len(v) for k, v in partial_fold_metrics.items()})

    # Plot and save metrics for all approaches
    plot_fold_metrics(full_fold_metrics, k_folds, title_suffix="Full Fine-Tuning")
    plot_fold_metrics(text_fold_metrics, k_folds, title_suffix="Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold_metrics, k_folds, title_suffix="Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold_metrics, k_folds, title_suffix="Partial Fine-Tuning")
    compare_metrics(full_fold_metrics, text_fold_metrics, vision_fold_metrics, partial_fold_metrics, k_folds)
    save_metrics_to_csv(full_fold_metrics, os.path.join("logs/main3-4", "kfold_full_finetune_metrics.csv"))
    save_metrics_to_csv(text_fold_metrics, os.path.join("logs/main3-4", "kfold_text_encoder_metrics.csv"))
    save_metrics_to_csv(vision_fold_metrics, os.path.join("logs/main3-4", "kfold_vision_only_metrics.csv"))
    save_metrics_to_csv(partial_fold_metrics, os.path.join("logs/main3-4", "kfold_partial_finetune_metrics.csv"))

    # Create comparison table for all approaches
    create_comparison_table(full_fold_metrics, text_fold_metrics, vision_fold_metrics, partial_fold_metrics, k_folds)

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
    print("\nPartial Fine-Tuning (Last 30% of Layers):")
    print(
        f"Avg Train Loss: {np.mean(partial_fold_metrics['train_loss']) if partial_fold_metrics['train_loss'] else 0:.4f}")
    print(
        f"Avg Validation Loss: {np.mean(partial_fold_metrics['val_loss']) if partial_fold_metrics['val_loss'] else 0:.4f}")
    print(
        f"Avg Test Loss: {np.mean(partial_fold_metrics['test_loss']) if partial_fold_metrics['test_loss'] else 0:.4f}")
    print(
        f"Avg Precision (Val): {np.mean(partial_fold_metrics['precision']) if partial_fold_metrics['precision'] else 0:.4f}")
    print(f"Avg Recall (Val): {np.mean(partial_fold_metrics['recall']) if partial_fold_metrics['recall'] else 0:.4f}")
    print(f"Avg F1-Score (Val): {np.mean(partial_fold_metrics['f1']) if partial_fold_metrics['f1'] else 0:.4f}")

    # Summary of timing results for comparison
    print("\nTiming Comparison Across Approaches (Total Duration per Fold):")
    for approach, durations in timing_results.items():
        if durations:
            total_duration = sum(durations)
            avg_duration = total_duration / len(durations)
            hours, remainder = divmod(total_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            avg_hours, avg_remainder = divmod(avg_duration, 3600)
            avg_minutes, avg_seconds = divmod(avg_remainder, 60)
            print(f"{approach}:")
            print(f"  Total Time Across Folds: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
            print(f"  Average Time per Fold: {int(avg_hours):02d}:{int(avg_minutes):02d}:{int(avg_seconds):02d}")
            for fold_idx, duration in enumerate(durations, 1):
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"    Fold {fold_idx}: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        else:
            print(f"{approach}: No timing data available (skipped due to checkpoint).")

    # Record overall end time and total duration
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    hours, remainder = divmod(overall_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nOverall Training Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time))}")
    print(f"Total Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print("\nK-Fold training complete. Results saved for analysis.")

    # Save timing data to CSV for reference
    timing_records = []
    for approach, durations in timing_results.items():
        for fold_idx, duration in enumerate(durations, 1) if durations else range(1, k_folds + 1):
            if fold_idx <= len(durations) and duration > 0:
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                formatted_time = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
            else:
                formatted_time = "N/A"
                duration = 0
            timing_records.append({
                "Approach": approach,
                "Fold": fold_idx,
                "Duration (seconds)": duration,
                "Formatted Duration": formatted_time
            })
    timing_df = pd.DataFrame(timing_records)
    timing_save_path = os.path.join("logs/main3-4", "execution_timing.csv")
    timing_df.to_csv(timing_save_path, index=False)
    print(f"Timing data saved to {timing_save_path}")