import os
import time
import json
import shutil
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.multiprocessing import freeze_support

from sklearn.model_selection import KFold
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

import open_clip
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------
# Housekeeping
# ------------------------------
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation")
warnings.filterwarnings("ignore", message="Repo card metadata block was not found")

LOG_ROOT = "logs/main3-4"
os.makedirs(LOG_ROOT, exist_ok=True)

# ------------------------------
# Dataset
# ------------------------------
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
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        label = self.subcategories.index(item["subCategory"])
        image = self.transform(image)
        return image, label

# ------------------------------
# Models
# ------------------------------
class FullFineTunedCLIP(nn.Module):
    """
    Full fine-tuning of the CLIP visual encoder + linear head (classification).
    """
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=False):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.is_encoder_frozen = freeze_encoder
        if freeze_encoder:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        device = next(base_model.parameters()).device
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]

        layers = []
        in_dim = out_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5)]
            in_dim = 512
        layers += [nn.Linear(in_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        feats = self.visual_encoder(images)
        logits = self.classifier(feats)
        return logits


class TextEncoderFineTunedCLIP(nn.Module):
    """
    Freeze visual, train text encoder; produce similarity logits (image x text).
    """
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder = base_model.transformer
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device
        self.subcategories = subcategories

        if freeze_visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

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
            text_inputs = self.tokenizer([f"a photo of {c}" for c in self.subcategories]).to(self.device)
        text_features = self.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_features, text_features.T) * 100.0
        return logits


class VisionOnlyFineTunedCLIP(nn.Module):
    """
    Train visual encoder + linear head (classification).
    """
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = True

        device = next(base_model.parameters()).device
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]

        layers, in_dim = [], out_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5)]
            in_dim = 512
        layers += [nn.Linear(in_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        feats = self.visual_encoder(images)
        logits = self.classifier(feats)
        return logits


class PartialFineTunedCLIP(nn.Module):
    """
    Unfreeze the LAST ~30% of parameters in BOTH encoders.
    Produces similarity logits when text_inputs is provided (both encoders in path).
    """
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device

        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]

        # Optional fallback path (unused when similarity path is used)
        self.classifier = nn.Linear(out_dim, num_classes)

        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder,   freeze_percentage, "text")

        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding

    def _freeze_layers(self, module, freeze_percentage, encoder_type):
        named_params = list(module.named_parameters())
        total_tensors = len(named_params)
        freeze_until = int(total_tensors * freeze_percentage)
        print(
            f"Freezing first {freeze_percentage*100:.1f}% of {encoder_type} encoder parameter tensors "
            f"({freeze_until}/{total_tensors})"
        )
        for i, (_, p) in enumerate(named_params):
            p.requires_grad = i >= freeze_until
        frozen_params    = sum(p.numel() for _, p in named_params[:freeze_until])
        trainable_params = sum(p.numel() for _, p in named_params[freeze_until:])
        print(f"{encoder_type.capitalize()} Encoder: Frozen {frozen_params} params, Trainable {trainable_params} params")

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
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
            logits = torch.matmul(image_features, text_features.T) * 100.0
            return logits
        # fallback classifier path
        return self.classifier(image_features)

# ------------------------------
# Utilities
# ------------------------------
def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_weights)
    model = model.to(device).eval()
    return model, preprocess, preprocess

def load_fashion_dataset():
    dataset = load_dataset("ceyda/fashion-products-small")
    train_data = dataset["train"]
    subcategories = sorted(set(item["subCategory"] for item in train_data))
    print(f"Loaded dataset with {len(subcategories)} subcategories.")
    return train_data, subcategories

def compute_class_weights(dataset, subcategories):
    labels = [subcategories.index(item["subCategory"]) for item in dataset]
    class_counts = np.bincount(labels, minlength=len(subcategories))
    total = len(labels)
    weights = total / (len(subcategories) * np.clip(class_counts, 1, None))
    return torch.tensor(weights, dtype=torch.float)

def save_metrics_to_csv(metrics, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(metrics).to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

def save_logs_as_zip(log_dir, zip_name):
    if os.path.exists(log_dir):
        with ZipFile(f"{zip_name}.zip", "w") as zipf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.join(root.replace(log_dir, log_dir.split("/")[-2]), file))
        print(f"Logs zipped to {zip_name}.zip")
        shutil.rmtree(log_dir, ignore_errors=True)

# ------------------------------
# Evaluation, Plots & Tables
# ------------------------------
def plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    labels = subcategories[:num_classes] if num_classes <= len(subcategories) else [f"Class_{i}" for i in range(num_classes)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join(LOG_ROOT, model_type.lower().replace(" ", "_"), f"fold_{fold}",
                             f"confusion_matrix_epoch_{epoch}_{split_type.lower()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    rows = []
    for t in range(num_classes):
        total_true = np.sum(conf_matrix[t])
        if total_true == 0:
            continue
        for p in range(num_classes):
            if t != p and conf_matrix[t][p] > 0:
                rows.append({
                    "True Class": subcategories[t] if t < len(subcategories) else f"Unknown_{t}",
                    "Predicted Class": subcategories[p] if p < len(subcategories) else f"Unknown_{p}",
                    "Count": int(conf_matrix[t][p]),
                    "Percentage of True Class": 100.0 * conf_matrix[t][p] / total_true,
                })
    if rows:
        df = pd.DataFrame(rows).sort_values(by="Count", ascending=False).head(10)
        save_path = os.path.join(LOG_ROOT, model_type.lower().replace(" ", "_"), f"fold_{fold}",
                                 f"misclassifications_epoch_{epoch}_{split_type.lower()}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Misclassifications saved to {save_path}")

def evaluate_model_with_metrics(model, dataloader, criterion, subcategories, device, fold, epoch, model_type, split_type="Validation"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    # prepare prompts once if needed
    texts = None
    if hasattr(model, "tokenizer"):
        texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, text_inputs=texts) if texts is not None else model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(cm, subcategories, model_type, fold, epoch, split_type)
    analyze_misclassifications(cm, subcategories, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): Loss={avg_loss:.4f}, Acc={acc:.4f}, "
          f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, cm

def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    # Metrics dict expected keys: train_loss, val_loss, test_loss, train_acc, val_acc, test_acc
    folds = range(1, max(len(metrics.get("val_loss", [])),
                         len(metrics.get("test_loss", [])),
                         len(metrics.get("val_acc", [])),
                         len(metrics.get("test_acc", [])), k_folds) + 1)

    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    if len(metrics.get("val_loss", [])) > 0:
        plt.plot(folds[:len(metrics["val_loss"])], metrics["val_loss"], marker="o", label="Val Loss")
    if len(metrics.get("test_loss", [])) > 0:
        plt.plot(folds[:len(metrics["test_loss"])], metrics["test_loss"], marker="s", label="Test Loss")
    plt.title(f"Loss Across Folds – {title_suffix}"); plt.xlabel("Fold"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(2, 2, 2)
    if len(metrics.get("train_loss", [])) > 0:
        plt.plot(folds[:len(metrics["train_loss"])], metrics["train_loss"], marker=".", label="Train Loss")
    plt.title(f"Train Loss Across Folds – {title_suffix}"); plt.xlabel("Fold"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(2, 2, 3)
    if len(metrics.get("val_acc", [])) > 0:
        plt.plot(folds[:len(metrics["val_acc"])], metrics["val_acc"], marker="o", label="Val Acc")
    if len(metrics.get("test_acc", [])) > 0:
        plt.plot(folds[:len(metrics["test_acc"])], metrics["test_acc"], marker="s", label="Test Acc")
    plt.title(f"Accuracy Across Folds – {title_suffix}"); plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.legend()

    plt.subplot(2, 2, 4)
    if len(metrics.get("train_acc", [])) > 0:
        plt.plot(folds[:len(metrics["train_acc"])], metrics["train_acc"], marker=".", label="Train Acc")
    plt.title(f"Train Accuracy Across Folds – {title_suffix}"); plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    save_path = os.path.join(LOG_ROOT, f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Plot saved to {save_path}")

def compare_metrics(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(15, 12))

    # Val Loss
    plt.subplot(2, 2, 1)
    for name, m, mk in [("Full", full_metrics, "o"), ("Text", text_metrics, "x"),
                        ("Vision", vision_metrics, "s"), ("Partial", partial_metrics, "d")]:
        if len(m.get("val_loss", [])) > 0:
            plt.plot(folds[:len(m["val_loss"])], m["val_loss"], marker=mk, label=f"{name} Val Loss")
    plt.title("Validation Loss (All)"); plt.xlabel("Fold"); plt.ylabel("Loss"); plt.legend()

    # Test Loss
    plt.subplot(2, 2, 2)
    for name, m, mk in [("Full", full_metrics, "o"), ("Text", text_metrics, "x"),
                        ("Vision", vision_metrics, "s"), ("Partial", partial_metrics, "d")]:
        if len(m.get("test_loss", [])) > 0:
            plt.plot(folds[:len(m["test_loss"])], m["test_loss"], marker=mk, label=f"{name} Test Loss")
    plt.title("Test Loss (All)"); plt.xlabel("Fold"); plt.ylabel("Loss"); plt.legend()

    # Val Acc
    plt.subplot(2, 2, 3)
    for name, m, mk in [("Full", full_metrics, "o"), ("Text", text_metrics, "x"),
                        ("Vision", vision_metrics, "s"), ("Partial", partial_metrics, "d")]:
        if len(m.get("val_acc", [])) > 0:
            plt.plot(folds[:len(m["val_acc"])], m["val_acc"], marker=mk, label=f"{name} Val Acc")
    plt.title("Validation Accuracy (All)"); plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.legend()

    # Test Acc
    plt.subplot(2, 2, 4)
    for name, m, mk in [("Full", full_metrics, "o"), ("Text", text_metrics, "x"),
                        ("Vision", vision_metrics, "s"), ("Partial", partial_metrics, "d")]:
        if len(m.get("test_acc", [])) > 0:
            plt.plot(folds[:len(m["test_acc"])], m["test_acc"], marker=mk, label=f"{name} Test Acc")
    plt.title("Test Accuracy (All)"); plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.legend()

    plt.tight_layout()
    save_path = os.path.join(LOG_ROOT, "comparison_metrics_all.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Comparison plot saved to {save_path}")

def create_comparison_table(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
    def get(m, key, i):
        return m.get(key, [])[i] if i < len(m.get(key, [])) else 0.0

    rows = []
    for i in range(k_folds):
        rows.append({
            "Fold": i + 1,
            "Full Train Loss": get(full_metrics, "train_loss", i),
            "Full Train Acc":  get(full_metrics, "train_acc",  i),
            "Full Val Loss":   get(full_metrics, "val_loss",   i),
            "Full Val Acc":    get(full_metrics, "val_acc",    i),
            "Full Test Loss":  get(full_metrics, "test_loss",  i),
            "Full Test Acc":   get(full_metrics, "test_acc",   i),

            "Text Train Loss": get(text_metrics, "train_loss", i),
            "Text Train Acc":  get(text_metrics, "train_acc",  i),
            "Text Val Loss":   get(text_metrics, "val_loss",   i),
            "Text Val Acc":    get(text_metrics, "val_acc",    i),
            "Text Test Loss":  get(text_metrics, "test_loss",  i),
            "Text Test Acc":   get(text_metrics, "test_acc",   i),

            "Vision Train Loss": get(vision_metrics, "train_loss", i),
            "Vision Train Acc":  get(vision_metrics, "train_acc",  i),
            "Vision Val Loss":   get(vision_metrics, "val_loss",   i),
            "Vision Val Acc":    get(vision_metrics, "val_acc",    i),
            "Vision Test Loss":  get(vision_metrics, "test_loss",  i),
            "Vision Test Acc":   get(vision_metrics, "test_acc",   i),

            "Partial Train Loss": get(partial_metrics, "train_loss", i),
            "Partial Train Acc":  get(partial_metrics, "train_acc",  i),
            "Partial Val Loss":   get(partial_metrics, "val_loss",   i),
            "Partial Val Acc":    get(partial_metrics, "val_acc",    i),
            "Partial Test Loss":  get(partial_metrics, "test_loss",  i),
            "Partial Test Acc":   get(partial_metrics, "test_acc",   i),
        })

    def avg(m, k): return float(np.mean(m.get(k, []))) if m.get(k, []) else 0.0

    rows.append({
        "Fold": "Average",
        "Full Train Loss":   avg(full_metrics,   "train_loss"),
        "Full Train Acc":    avg(full_metrics,   "train_acc"),
        "Full Val Loss":     avg(full_metrics,   "val_loss"),
        "Full Val Acc":      avg(full_metrics,   "val_acc"),
        "Full Test Loss":    avg(full_metrics,   "test_loss"),
        "Full Test Acc":     avg(full_metrics,   "test_acc"),

        "Text Train Loss":   avg(text_metrics,   "train_loss"),
        "Text Train Acc":    avg(text_metrics,   "train_acc"),
        "Text Val Loss":     avg(text_metrics,   "val_loss"),
        "Text Val Acc":      avg(text_metrics,   "val_acc"),
        "Text Test Loss":    avg(text_metrics,   "test_loss"),
        "Text Test Acc":     avg(text_metrics,   "test_acc"),

        "Vision Train Loss": avg(vision_metrics, "train_loss"),
        "Vision Train Acc":  avg(vision_metrics, "train_acc"),
        "Vision Val Loss":   avg(vision_metrics, "val_loss"),
        "Vision Val Acc":    avg(vision_metrics, "val_acc"),
        "Vision Test Loss":  avg(vision_metrics, "test_loss"),
        "Vision Test Acc":   avg(vision_metrics, "test_acc"),

        "Partial Train Loss": avg(partial_metrics, "train_loss"),
        "Partial Train Acc":  avg(partial_metrics, "train_acc"),
        "Partial Val Loss":   avg(partial_metrics, "val_loss"),
        "Partial Val Acc":    avg(partial_metrics, "val_acc"),
        "Partial Test Loss":  avg(partial_metrics, "test_loss"),
        "Partial Test Acc":   avg(partial_metrics, "test_acc"),
    })

    df = pd.DataFrame(rows)
    save_path = os.path.join(LOG_ROOT, "comparison_table_all.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Comparison table saved to {save_path}")
    return df

# ------------------------------
# Training
# ------------------------------
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs,
                subcategories, device, log_dir, fold, accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    best_f1 = 0.0
    patience, patience_counter = 3, 0

    metrics = {
        "epoch": [], "train_loss": [], "val_loss": [], "test_loss": [],
        "train_acc": [], "val_acc": [], "test_acc": [],
        "precision_val": [], "recall_val": [], "f1_val": [],
        "precision_test": [], "recall_test": [], "f1_test": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0
        optimizer.zero_grad()

        texts = None
        if hasattr(model, "tokenizer"):
            texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

        with tqdm(train_loader, desc=f"{model_type} | Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for bidx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images, text_inputs=texts) if texts is not None else model(images)
                    loss = criterion(outputs, labels) / accumulation_steps

                scaler.scale(loss).backward()
                if (bidx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

                pbar.set_postfix({
                    "loss": f"{(total_train_loss / (bidx + 1)):.4f}",
                    "acc":  f"{(correct_train / max(total_train, 1)):.4f}",
                })

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        # Validate/Test periodically
        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Test"
            )

            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["test_loss"].append(test_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_acc"].append(val_acc)
            metrics["test_acc"].append(test_acc)
            metrics["precision_val"].append(val_prec)
            metrics["recall_val"].append(val_rec)
            metrics["f1_val"].append(val_f1)
            metrics["precision_test"].append(test_prec)
            metrics["recall_test"].append(test_rec)
            metrics["f1_test"].append(test_f1)

            # Early stopping on val F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                save_path = os.path.join(log_dir, f"best_{model_type.lower().replace(' ', '_')}_clip.pth")
                torch.save(model.state_dict(), save_path)
                print(f"Best model saved at {save_path} with F1={best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            # still log train-only epoch
            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(0.0)
            metrics["test_loss"].append(0.0)
            metrics["train_acc"].append(train_acc)
            metrics["val_acc"].append(0.0)
            metrics["test_acc"].append(0.0)
            metrics["precision_val"].append(0.0)
            metrics["recall_val"].append(0.0)
            metrics["f1_val"].append(0.0)
            metrics["precision_test"].append(0.0)
            metrics["recall_test"].append(0.0)
            metrics["f1_test"].append(0.0)

        print(f"{model_type} | Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    return metrics

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    freeze_support()
    overall_start = time.time()
    print(f"\nOverall Training Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}")

    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(device, "ViT-B-32", "openai")
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    num_workers = min(8, os.cpu_count() or 2)
    print(f"Using {num_workers} workers for data loading.")

    # Fold-level aggregators (store loss & accuracy)
    def init_fold_metrics():
        return {
            "train_loss": [], "val_loss": [], "test_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": [],
        }

    full_fold, text_fold, vision_fold, partial_fold = (
        init_fold_metrics(), init_fold_metrics(), init_fold_metrics(), init_fold_metrics()
    )

    timing_results = {
        "Full Fine-Tuning": [], "Text Encoder Fine-Tuning": [],
        "Vision-Only Fine-Tuning": [], "Partial Fine-Tuning": []
    }

    checkpoint_file = os.path.join(LOG_ROOT, "checkpoint.json")
    checkpoint = {"current_fold": 0, "completed_approaches": {}}

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: resuming from Fold {checkpoint['current_fold'] + 1}")
            for key, tgt in [
                ("full_fold_metrics", full_fold),
                ("text_fold_metrics", text_fold),
                ("vision_fold_metrics", vision_fold),
                ("partial_fold_metrics", partial_fold),
                ("timing_results", timing_results)
            ]:
                if key in checkpoint:
                    locals_name = { "full_fold_metrics": "full_fold",
                                    "text_fold_metrics": "text_fold",
                                    "vision_fold_metrics": "vision_fold",
                                    "partial_fold_metrics": "partial_fold",
                                    "timing_results": "timing_results" }[key]
                    locals()[locals_name] = checkpoint[key]
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        if fold_idx < checkpoint["current_fold"]:
            print(f"\nSkipping Fold {fold_idx + 1}/{k_folds} — already completed.")
            continue

        print(f"\n--- Fold {fold_idx + 1}/{k_folds} ---")
        train_val_size = len(train_val_idx)
        train_size = int(0.8 * train_val_size)
        train_idx = train_val_idx[:train_size]
        val_idx = train_val_idx[train_size:]

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        test_subset  = Subset(dataset, test_idx.tolist())

        train_ds = FashionDataset(train_subset, subcategories, augment=True)
        val_ds   = FashionDataset(val_subset,   subcategories)
        test_ds  = FashionDataset(test_subset,  subcategories)

        pin_mem = device.type == "cuda"
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

        if str(fold_idx) not in checkpoint["completed_approaches"]:
            checkpoint["completed_approaches"][str(fold_idx)] = []

        # ----- 1) Full Fine-Tuning -----
        if "full" not in checkpoint["completed_approaches"][str(fold_idx)]:
            print("\nTraining: Full Fine-Tuning …")
            start = time.time()

            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(full_model.parameters(), lr=1e-5, weight_decay=1e-4)

            log_dir = os.path.join(LOG_ROOT, "full_finetune", f"fold_{fold_idx + 1}")
            full_metrics = train_model(
                full_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device, log_dir=log_dir,
                fold=fold_idx + 1, accumulation_steps=4, validate_every=2, model_type="Full Fine-Tuning"
            )

            best = os.path.join(log_dir, "best_full_fine-tuning_clip.pth")
            if os.path.exists(best):
                full_model.load_state_dict(torch.load(best, map_location=device))
                print(f"Loaded best model from {best}")

            # Final eval for fold-level aggregation
            v_loss, v_acc, _, _, _, _ = evaluate_model_with_metrics(
                full_model, val_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Validation"
            )
            t_loss, t_acc, _, _, _, _ = evaluate_model_with_metrics(
                full_model, test_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Test"
            )

            full_fold["train_loss"].append(full_metrics["train_loss"][-1] if full_metrics["train_loss"] else 0.0)
            full_fold["train_acc"].append(full_metrics["train_acc"][-1] if full_metrics["train_acc"] else 0.0)
            full_fold["val_loss"].append(v_loss);  full_fold["val_acc"].append(v_acc)
            full_fold["test_loss"].append(t_loss); full_fold["test_acc"].append(t_acc)

            save_metrics_to_csv(full_metrics, os.path.join(LOG_ROOT, f"metrics_full_finetune_fold_{fold_idx + 1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(LOG_ROOT, "full_finetune", f"fold_{fold_idx + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Full Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Full Fine-Tuning (Fold {fold_idx + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint["completed_approaches"][str(fold_idx)].append("full")
            checkpoint["current_fold"] = fold_idx
            checkpoint["full_fold_metrics"] = full_fold
            checkpoint["text_fold_metrics"] = text_fold
            checkpoint["vision_fold_metrics"] = vision_fold
            checkpoint["partial_fold_metrics"] = partial_fold
            checkpoint["timing_results"] = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            print("Checkpoint saved.")
        else:
            print("Skipping Full Fine-Tuning — already completed.")

        # ----- 2) Text Encoder Fine-Tuning -----
        if "text" not in checkpoint["completed_approaches"][str(fold_idx)]:
            print("\nTraining: Text Encoder Fine-Tuning …")
            start = time.time()

            text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)

            log_dir = os.path.join(LOG_ROOT, "text_encoder_finetune", f"fold_{fold_idx + 1}")
            text_metrics = train_model(
                text_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device, log_dir=log_dir,
                fold=fold_idx + 1, accumulation_steps=4, validate_every=2, model_type="Text Encoder Fine-Tuning"
            )

            best = os.path.join(log_dir, "best_text_encoder_fine-tuning_clip.pth")
            if os.path.exists(best):
                text_model.load_state_dict(torch.load(best, map_location=device))
                print(f"Loaded best model from {best}")

            v_loss, v_acc, _, _, _, _ = evaluate_model_with_metrics(
                text_model, val_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Validation"
            )
            t_loss, t_acc, _, _, _, _ = evaluate_model_with_metrics(
                text_model, test_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Test"
            )

            text_fold["train_loss"].append(text_metrics["train_loss"][-1] if text_metrics["train_loss"] else 0.0)
            text_fold["train_acc"].append(text_metrics["train_acc"][-1] if text_metrics["train_acc"] else 0.0)
            text_fold["val_loss"].append(v_loss);  text_fold["val_acc"].append(v_acc)
            text_fold["test_loss"].append(t_loss); text_fold["test_acc"].append(t_acc)

            save_metrics_to_csv(text_metrics, os.path.join(LOG_ROOT, f"metrics_text_encoder_fold_{fold_idx + 1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(LOG_ROOT, "text_encoder_finetune", f"fold_{fold_idx + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Text Encoder Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Text Encoder Fine-Tuning (Fold {fold_idx + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint["completed_approaches"][str(fold_idx)].append("text")
            checkpoint["current_fold"] = fold_idx
            checkpoint["full_fold_metrics"] = full_fold
            checkpoint["text_fold_metrics"] = text_fold
            checkpoint["vision_fold_metrics"] = vision_fold
            checkpoint["partial_fold_metrics"] = partial_fold
            checkpoint["timing_results"] = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            print("Checkpoint saved.")
        else:
            print("Skipping Text Encoder Fine-Tuning — already completed.")

        # ----- 3) Vision-Only Fine-Tuning -----
        if "vision" not in checkpoint["completed_approaches"][str(fold_idx)]:
            print("\nTraining: Vision-Only Fine-Tuning …")
            start = time.time()

            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(vision_model.parameters(), lr=1e-4)

            log_dir = os.path.join(LOG_ROOT, "vision_only_finetune", f"fold_{fold_idx + 1}")
            vision_metrics = train_model(
                vision_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device, log_dir=log_dir,
                fold=fold_idx + 1, accumulation_steps=4, validate_every=2, model_type="Vision-Only Fine-Tuning"
            )

            best = os.path.join(log_dir, "best_vision-only_fine-tuning_clip.pth")
            if os.path.exists(best):
                vision_model.load_state_dict(torch.load(best, map_location=device))
                print(f"Loaded best model from {best}")

            v_loss, v_acc, _, _, _, _ = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Validation"
            )
            t_loss, t_acc, _, _, _, _ = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Test"
            )

            vision_fold["train_loss"].append(vision_metrics["train_loss"][-1] if vision_metrics["train_loss"] else 0.0)
            vision_fold["train_acc"].append(vision_metrics["train_acc"][-1] if vision_metrics["train_acc"] else 0.0)
            vision_fold["val_loss"].append(v_loss);   vision_fold["val_acc"].append(v_acc)
            vision_fold["test_loss"].append(t_loss);  vision_fold["test_acc"].append(t_acc)

            save_metrics_to_csv(vision_metrics, os.path.join(LOG_ROOT, f"metrics_vision_only_fold_{fold_idx + 1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(LOG_ROOT, "vision_only_finetune", f"fold_{fold_idx + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Vision-Only Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Vision-Only Fine-Tuning (Fold {fold_idx + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint["completed_approaches"][str(fold_idx)].append("vision")
            checkpoint["current_fold"] = fold_idx
            checkpoint["full_fold_metrics"] = full_fold
            checkpoint["text_fold_metrics"] = text_fold
            checkpoint["vision_fold_metrics"] = vision_fold
            checkpoint["partial_fold_metrics"] = partial_fold
            checkpoint["timing_results"] = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            print("Checkpoint saved.")
        else:
            print("Skipping Vision-Only Fine-Tuning — already completed.")

        # ----- 4) Partial Fine-Tuning (last 30% both encoders) -----
        if "partial" not in checkpoint["completed_approaches"][str(fold_idx)]:
            print("\nTraining: Partial Fine-Tuning (last 30% of BOTH encoders) …")
            start = time.time()

            partial_model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            params = [p for p in partial_model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=1e-4)

            log_dir = os.path.join(LOG_ROOT, "partial_finetune", f"fold_{fold_idx + 1}")
            partial_metrics = train_model(
                partial_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device, log_dir=log_dir,
                fold=fold_idx + 1, accumulation_steps=4, validate_every=2, model_type="Partial Fine-Tuning"
            )

            best = os.path.join(log_dir, "best_partial_fine-tuning_clip.pth")
            if os.path.exists(best):
                partial_model.load_state_dict(torch.load(best, map_location=device))
                print(f"Loaded best model from {best}")

            v_loss, v_acc, _, _, _, _ = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Validation"
            )
            t_loss, t_acc, _, _, _, _ = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold_idx + 1, "Final", "Test"
            )

            partial_fold["train_loss"].append(partial_metrics["train_loss"][-1] if partial_metrics["train_loss"] else 0.0)
            partial_fold["train_acc"].append(partial_metrics["train_acc"][-1] if partial_metrics["train_acc"] else 0.0)
            partial_fold["val_loss"].append(v_loss);   partial_fold["val_acc"].append(v_acc)
            partial_fold["test_loss"].append(t_loss);  partial_fold["test_acc"].append(t_acc)  # <- ensures Partial Test Acc is recorded

            save_metrics_to_csv(partial_metrics, os.path.join(LOG_ROOT, f"metrics_partial_finetune_fold_{fold_idx + 1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(LOG_ROOT, "partial_finetune", f"fold_{fold_idx + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Partial Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Partial Fine-Tuning (Fold {fold_idx + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint["completed_approaches"][str(fold_idx)].append("partial")
            checkpoint["current_fold"] = fold_idx
            checkpoint["full_fold_metrics"] = full_fold
            checkpoint["text_fold_metrics"] = text_fold
            checkpoint["vision_fold_metrics"] = vision_fold
            checkpoint["partial_fold_metrics"] = partial_fold
            checkpoint["timing_results"] = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
            print("Checkpoint saved.")
        else:
            print("Skipping Partial Fine-Tuning — already completed.")

    # Debug lengths
    print("\nDebug — fold metrics lengths:")
    for name, d in [("Full", full_fold), ("Text", text_fold), ("Vision", vision_fold), ("Partial", partial_fold)]:
        print(name, {k: len(v) for k, v in d.items()})

    # Plots & CSVs
    plot_fold_metrics(full_fold,    k_folds, "Full Fine-Tuning")
    plot_fold_metrics(text_fold,    k_folds, "Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold,  k_folds, "Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold, k_folds, "Partial Fine-Tuning")
    compare_metrics(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    save_metrics_to_csv(full_fold,    os.path.join(LOG_ROOT, "kfold_full_finetune_metrics.csv"))
    save_metrics_to_csv(text_fold,    os.path.join(LOG_ROOT, "kfold_text_encoder_metrics.csv"))
    save_metrics_to_csv(vision_fold,  os.path.join(LOG_ROOT, "kfold_vision_only_metrics.csv"))
    save_metrics_to_csv(partial_fold, os.path.join(LOG_ROOT, "kfold_partial_finetune_metrics.csv"))

    # Comparison table (includes Train/Val/Test Acc)
    create_comparison_table(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    # Timing CSV
    timing_rows = []
    for approach, durations in timing_results.items():
        if durations:
            for i, d in enumerate(durations, 1):
                h, rem = divmod(d, 3600); m, s = divmod(rem, 60)
                timing_rows.append({
                    "Approach": approach, "Fold": i,
                    "Duration (seconds)": d,
                    "Formatted Duration": f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                })
        else:
            for i in range(1, k_folds + 1):
                timing_rows.append({"Approach": approach, "Fold": i, "Duration (seconds)": 0, "Formatted Duration": "N/A"})
    timing_df = pd.DataFrame(timing_rows)
    tpath = os.path.join(LOG_ROOT, "execution_timing.csv")
    timing_df.to_csv(tpath, index=False)
    print(f"Timing data saved to {tpath}")

    overall_end = time.time()
    H, R = divmod(overall_end - overall_start, 3600); M, S = divmod(R, 60)
    print(f"\nOverall Training Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}")
    print(f"Total Duration: {int(H):02d}:{int(M):02d}:{int(S):02d}")
