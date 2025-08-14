# -*- coding: utf-8 -*-
import os
import json
import time
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

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from datasets import load_dataset
from tqdm import tqdm

import open_clip
from torch.multiprocessing import freeze_support

# ------------------------------
# Warnings
# ------------------------------
warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation")
warnings.filterwarnings("ignore", message="Repo card metadata block was not found")

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
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = self.subcategories.index(item['subCategory'])
        image = self.transform(image)
        return image, label

# ------------------------------
# Models
# ------------------------------
class FullFineTunedCLIP(nn.Module):
    """
    Vision encoder + classifier head.
    freeze_encoder=False => truly full fine-tuning of visual encoder + head.
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


class TextEncoderFineTunedCLIP(nn.Module):
    """Freeze visual, train text; produce similarity logits vs class prompts."""
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = device
        self.subcategories  = subcategories

        if freeze_visual:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding

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
        logits = (image_features @ text_features.T) * 100.0
        return logits


class VisionOnlyFineTunedCLIP(nn.Module):
    """Train visual encoder + classifier head (image-only classifier)."""
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
    Unfreeze the LAST ~30% of parameter tensors in BOTH encoders.
    Uses similarity logits when text_inputs is provided (prompt-based classification).
    """
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device

        # For optional image-only path (not used when text inputs are provided)
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]
        self.classifier = nn.Linear(out_dim, num_classes)

        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder,   freeze_percentage, "text")

        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding

    def _freeze_layers(self, module, freeze_percentage, enc_name):
        named_params = list(module.named_parameters())
        total = len(named_params)
        freeze_until = int(total * freeze_percentage)
        print(f"Freezing first {freeze_percentage*100:.1f}% of {enc_name} encoder "
              f"({freeze_until}/{total} parameter tensors)")
        for i, (_, p) in enumerate(named_params):
            p.requires_grad = i >= freeze_until
        frozen_params    = sum(p.numel() for _, p in named_params[:freeze_until])
        trainable_params = sum(p.numel() for _, p in named_params[freeze_until:])
        print(f"{enc_name.capitalize()} Encoder: Frozen {frozen_params}, Trainable {trainable_params}")

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
        if text_inputs is not None:  # similarity path
            text_features = self.encode_text(text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
            return (image_features @ text_features.T) * 100.0
        # fallback classifier
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
    counts = np.bincount(labels, minlength=len(subcategories))
    total = len(labels)
    weights = total / (len(subcategories) * np.clip(counts, 1, None))
    return torch.tensor(weights, dtype=torch.float)

def save_metrics_to_csv(metrics, filename):
    df = pd.DataFrame(metrics)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Metrics saved to {filename}")

def save_logs_as_zip(log_dir, zip_name):
    if os.path.exists(log_dir):
        with ZipFile(f"{zip_name}.zip", 'w') as zipf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    src = os.path.join(root, file)
                    arc = os.path.join(root.replace(log_dir, os.path.basename(os.path.dirname(log_dir))), file)
                    zipf.write(src, arc)
        print(f"Logs zipped to {zip_name}.zip")
        shutil.rmtree(log_dir, ignore_errors=True)

def plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    labels = subcategories[:num_classes] if num_classes <= len(subcategories) else [f"Class_{i}" for i in range(num_classes)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    out = os.path.join("logs/main3-4aa", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                       f"confusion_matrix_epoch_{epoch}_{split_type.lower()}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out); plt.close()
    print(f"Confusion matrix saved to {out}")

def analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    n = conf_matrix.shape[0]
    mis = []
    for t in range(n):
        total_true = np.sum(conf_matrix[t])
        if total_true == 0:
            continue
        for p in range(n):
            if t != p and conf_matrix[t][p] > 0:
                cnt = conf_matrix[t][p]
                mis.append({
                    "True Class": subcategories[t] if t < len(subcategories) else f"Class_{t}",
                    "Predicted Class": subcategories[p] if p < len(subcategories) else f"Class_{p}",
                    "Count": cnt,
                    "Percentage of True Class": 100.0 * cnt / total_true
                })
    if mis:
        df = pd.DataFrame(mis).sort_values("Count", ascending=False).head(10)
        out = os.path.join("logs/main3-4aa", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                           f"misclassifications_epoch_{epoch}_{split_type.lower()}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Misclassifications saved to {out}")
        return df
    return None

def evaluate_model_with_metrics(model, dataloader, criterion, subcategories, device,
                                fold, epoch, model_type, split_type="Validation"):
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []

    # If model supports tokenizer, build prompts once
    texts = None
    if hasattr(model, "tokenizer"):
        texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, text_inputs=texts) if texts is not None else model(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = tot_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                               average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, subcategories, model_type, fold, epoch, split_type)
    analyze_misclassifications(cm, subcategories, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): "
          f"Loss={avg_loss:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, cm

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs, subcategories, device, log_dir, fold,
                accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    best_f1 = 0.0
    patience, patience_counter = 3, 0

    metrics = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "test_loss": [], "test_acc": [],
        "precision_val": [], "recall_val": [], "f1_val": [],
        "precision_test": [], "recall_test": [], "f1_test": []
    }

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        correct_train, total_train = 0, 0
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
                total_train   += labels.size(0)

                pbar.set_postfix({
                    "loss": f"{(total_train_loss / (bidx+1)):.4f}",
                    "acc":  f"{(correct_train / max(total_train,1)):.4f}"
                })

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        # Validate (and test) at interval or final epoch
        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Test"
            )

            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_loss"].append(v_loss)
            metrics["val_acc"].append(v_acc)
            metrics["test_loss"].append(t_loss)
            metrics["test_acc"].append(t_acc)
            metrics["precision_val"].append(v_p)
            metrics["recall_val"].append(v_r)
            metrics["f1_val"].append(v_f1)
            metrics["precision_test"].append(t_p)
            metrics["recall_test"].append(t_r)
            metrics["f1_test"].append(t_f1)

            if v_f1 > best_f1:
                best_f1 = v_f1
                patience_counter = 0
                best_path = os.path.join(log_dir, f"best_{model_type.lower().replace(' ', '_')}_clip.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved: {best_path} (Val F1={best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            # still track train epoch rows (val/test 0.0 placeholders)
            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss)
            metrics["train_acc"].append(train_acc)
            metrics["val_loss"].append(0.0)
            metrics["val_acc"].append(0.0)
            metrics["test_loss"].append(0.0)
            metrics["test_acc"].append(0.0)
            metrics["precision_val"].append(0.0)
            metrics["recall_val"].append(0.0)
            metrics["f1_val"].append(0.0)
            metrics["precision_test"].append(0.0)
            metrics["recall_test"].append(0.0)
            metrics["f1_test"].append(0.0)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    return metrics

# ------------------------------
# Plotting helpers across folds
# ------------------------------
def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    def maybe(key): return metrics.get(key, [])
    metric_lengths = [len(maybe(k)) for k in metrics if len(maybe(k)) > 0]
    if not metric_lengths:
        print(f"No data to plot for {title_suffix}")
        return
    num_points = min(metric_lengths)
    folds = range(1, num_points + 1)

    plt.figure(figsize=(14, 10))
    # Val Loss
    plt.subplot(2, 2, 1)
    if len(maybe("val_loss")) > 0:
        plt.plot(folds[:len(maybe("val_loss"))], maybe("val_loss"), marker="o", label="Val Loss")
    plt.xlabel("Fold"); plt.ylabel("Loss"); plt.title(f"Validation Loss — {title_suffix}"); plt.legend()

    # Test Loss
    plt.subplot(2, 2, 2)
    if len(maybe("test_loss")) > 0:
        plt.plot(folds[:len(maybe("test_loss"))], maybe("test_loss"), marker="o", label="Test Loss")
    plt.xlabel("Fold"); plt.ylabel("Loss"); plt.title(f"Test Loss — {title_suffix}"); plt.legend()

    # Val Acc
    plt.subplot(2, 2, 3)
    if len(maybe("val_acc")) > 0:
        plt.plot(folds[:len(maybe("val_acc"))], maybe("val_acc"), marker="o", label="Val Acc")
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.title(f"Validation Accuracy — {title_suffix}"); plt.legend()

    # Test Acc
    plt.subplot(2, 2, 4)
    if len(maybe("test_acc")) > 0:
        plt.plot(folds[:len(maybe("test_acc"))], maybe("test_acc"), marker="o", label="Test Acc")
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.title(f"Test Accuracy — {title_suffix}"); plt.legend()

    plt.tight_layout()
    out = os.path.join("logs/main3-4aa", f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")

def compare_metrics(full_m, text_m, vision_m, partial_m, k_folds):
    def series(d, key): return d.get(key, [])
    max_len = max(len(series(full_m,"val_loss")), len(series(text_m,"val_loss")),
                  len(series(vision_m,"val_loss")), len(series(partial_m,"val_loss")), k_folds)
    folds = range(1, max_len + 1)
    plt.figure(figsize=(15, 15))

    panels = [
        ("Validation Loss", "val_loss"),
        ("Test Loss",       "test_loss"),
        ("Validation Acc",  "val_acc"),
        ("Test Acc",        "test_acc"),
        ("Validation F1",   "f1"),
    ]
    names = [("Full", full_m, "o"), ("Text", text_m, "x"),
             ("Vision", vision_m, "s"), ("Partial", partial_m, "d")]

    for i, (title, key) in enumerate(panels, 1):
        plt.subplot(3, 2, i)
        for name, m, marker in names:
            y = series(m, key)
            if len(y) > 0:
                plt.plot(folds[:len(y)], y, marker=marker, label=name)
        plt.xlabel("Fold"); plt.ylabel(title); plt.title(title); plt.legend()

    plt.tight_layout()
    out = os.path.join("logs/main3-4aa", "comparison_metrics_all.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")

def plot_full_vs_partial(full_m, partial_m, k_folds):
    def s(d, k): return d.get(k, [])[:k_folds]
    folds = range(1, k_folds + 1)
    pairs = [("Val Accuracy", "val_acc"), ("Test Accuracy", "test_acc"),
             ("Val Loss", "val_loss"), ("Test Loss", "test_loss")]
    plt.figure(figsize=(12, 10))
    for i, (title, key) in enumerate(pairs, 1):
        plt.subplot(2, 2, i)
        plt.plot(folds[:len(s(full_m, key))],    s(full_m, key),    marker="o", label="Full FT")
        plt.plot(folds[:len(s(partial_m, key))], s(partial_m, key), marker="d", label="Partial FT (last 30%)")
        plt.xlabel("Fold"); plt.ylabel(title); plt.title(f"Full vs Partial — {title}"); plt.legend()
    plt.tight_layout()
    out = os.path.join("logs/main3-4aa", "full_vs_partial_summary.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out); plt.close()
    print(f"Saved: {out}")

# ------------------------------
# Fold-level comparison table
# ------------------------------
def create_fold_comparison_table(full_m, text_m, vision_m, partial_m, k_folds):
    def get(m, key, i):
        lst = m.get(key, [])
        return lst[i] if i < len(lst) else 0

    rows = []
    for i in range(k_folds):
        rows.append({
            "Fold": i + 1,
            "Full Train Loss":   get(full_m,   "train_loss", i),
            "Full Train Acc":    get(full_m,   "train_acc",  i),
            "Full Val Loss":     get(full_m,   "val_loss",   i),
            "Full Val Acc":      get(full_m,   "val_acc",    i),
            "Full Test Loss":    get(full_m,   "test_loss",  i),
            "Full Test Acc":     get(full_m,   "test_acc",   i),

            "Text Train Loss":   get(text_m,   "train_loss", i),
            "Text Train Acc":    get(text_m,   "train_acc",  i),
            "Text Val Loss":     get(text_m,   "val_loss",   i),
            "Text Val Acc":      get(text_m,   "val_acc",    i),
            "Text Test Loss":    get(text_m,   "test_loss",  i),
            "Text Test Acc":     get(text_m,   "test_acc",   i),

            "Vision Train Loss": get(vision_m, "train_loss", i),
            "Vision Train Acc":  get(vision_m, "train_acc",  i),
            "Vision Val Loss":   get(vision_m, "val_loss",   i),
            "Vision Val Acc":    get(vision_m, "val_acc",    i),
            "Vision Test Loss":  get(vision_m, "test_loss",  i),
            "Vision Test Acc":   get(vision_m, "test_acc",   i),

            "Partial Train Loss": get(partial_m, "train_loss", i),
            "Partial Train Acc":  get(partial_m, "train_acc",  i),
            "Partial Val Loss":   get(partial_m, "val_loss",   i),
            "Partial Val Acc":    get(partial_m, "val_acc",    i),
            "Partial Test Loss":  get(partial_m, "test_loss",  i),
            "Partial Test Acc":   get(partial_m, "test_acc",   i),
        })

    def avg(m, k):
        lst = m.get(k, [])
        return float(np.mean(lst)) if lst else 0.0

    rows.append({
        "Fold": "Average",
        "Full Train Loss":   avg(full_m,   "train_loss"),
        "Full Train Acc":    avg(full_m,   "train_acc"),
        "Full Val Loss":     avg(full_m,   "val_loss"),
        "Full Val Acc":      avg(full_m,   "val_acc"),
        "Full Test Loss":    avg(full_m,   "test_loss"),
        "Full Test Acc":     avg(full_m,   "test_acc"),

        "Text Train Loss":   avg(text_m,   "train_loss"),
        "Text Train Acc":    avg(text_m,   "train_acc"),
        "Text Val Loss":     avg(text_m,   "val_loss"),
        "Text Val Acc":      avg(text_m,   "val_acc"),
        "Text Test Loss":    avg(text_m,   "test_loss"),
        "Text Test Acc":     avg(text_m,   "test_acc"),

        "Vision Train Loss": avg(vision_m, "train_loss"),
        "Vision Train Acc":  avg(vision_m, "train_acc"),
        "Vision Val Loss":   avg(vision_m, "val_loss"),
        "Vision Val Acc":    avg(vision_m, "val_acc"),
        "Vision Test Loss":  avg(vision_m, "test_loss"),
        "Vision Test Acc":   avg(vision_m, "test_acc"),

        "Partial Train Loss": avg(partial_m, "train_loss"),
        "Partial Train Acc":  avg(partial_m, "train_acc"),
        "Partial Val Loss":   avg(partial_m, "val_loss"),
        "Partial Val Acc":    avg(partial_m, "val_acc"),
        "Partial Test Loss":  avg(partial_m, "test_loss"),
        "Partial Test Acc":   avg(partial_m, "test_acc"),
    })

    df = pd.DataFrame(rows)
    out = os.path.join("logs/main3-4aa", "fold_comparison_table.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved fold comparison table: {out}")
    return df

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    freeze_support()
    overall_start = time.time()
    print(f"\nOverall Training Started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}")

    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(device, "ViT-B-32", "openai")
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    # K-Fold
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    def init_fold_metrics():
        return {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "test_loss":  [], "test_acc":  [],
            "precision":  [], "recall":    [], "f1": []
        }

    full_fold = init_fold_metrics()
    text_fold = init_fold_metrics()
    vision_fold = init_fold_metrics()
    partial_fold = init_fold_metrics()

    timing = {
        "Full Fine-Tuning": [], "Text Encoder Fine-Tuning": [],
        "Vision-Only Fine-Tuning": [], "Partial Fine-Tuning": []
    }

    base_dir = os.path.join("logs", "main3-4")
    ckpt_path = os.path.join(base_dir, "checkpoint.json")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "r") as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint. Resuming from fold {checkpoint['current_fold'] + 1}")
            if "full_fold"   in checkpoint: full_fold   = checkpoint["full_fold"]
            if "text_fold"   in checkpoint: text_fold   = checkpoint["text_fold"]
            if "vision_fold" in checkpoint: vision_fold = checkpoint["vision_fold"]
            if "partial_fold"in checkpoint: partial_fold= checkpoint["partial_fold"]
            if "timing"      in checkpoint: timing      = checkpoint["timing"]
        except Exception as e:
            print(f"Checkpoint load error: {e}. Starting fresh.")

    num_workers = min(8, os.cpu_count())
    print(f"DataLoader workers: {num_workers}")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        if fold < checkpoint["current_fold"]:
            print(f"\nSkipping Fold {fold+1}/{k_folds} (already completed)")
            continue

        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        tv_size = len(train_val_idx)
        train_size = int(0.8 * tv_size)
        train_idx = train_val_idx[:train_size]
        val_idx   = train_val_idx[train_size:]

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        test_subset  = Subset(dataset, test_idx.tolist())

        train_ds = FashionDataset(train_subset, subcategories, augment=True)
        val_ds   = FashionDataset(val_subset,   subcategories)
        test_ds  = FashionDataset(test_subset,  subcategories)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        if str(fold) not in checkpoint["completed_approaches"]:
            checkpoint["completed_approaches"][str(fold)] = []

        # 1) Full Fine-Tuning (ALL visual params train)
        if "full" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining: Full Fine-Tuning")
            start = time.time()

            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(full_model.parameters(), lr=1e-5, weight_decay=1e-4)

            log_dir = os.path.join(base_dir, "full_finetune", f"fold_{fold+1}")
            full_metrics = train_model(
                full_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device,
                log_dir=log_dir, fold=fold+1, accumulation_steps=4, validate_every=2,
                model_type="Full Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_full_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                full_model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"Loaded best weights: {best_path}")

            # Final fold-level eval
            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                full_model, val_loader, criterion, subcategories, device, fold+1, 999, "Full Fine-Tuning", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                full_model, test_loader, criterion, subcategories, device, fold+1, 999, "Full Fine-Tuning", "Test"
            )

            full_fold["train_loss"].append(full_metrics["train_loss"][-1] if full_metrics["train_loss"] else 0)
            full_fold["train_acc"].append(full_metrics["train_acc"][-1] if full_metrics["train_acc"] else 0)
            full_fold["val_loss"].append(v_loss);  full_fold["val_acc"].append(v_acc)
            full_fold["test_loss"].append(t_loss); full_fold["test_acc"].append(t_acc)
            full_fold["precision"].append(v_p);    full_fold["recall"].append(v_r); full_fold["f1"].append(v_f1)

            save_metrics_to_csv(full_metrics, os.path.join(base_dir, f"metrics_full_finetune_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(base_dir, "full_finetune", f"fold_{fold+1}_logs"))

            dur = time.time() - start
            timing["Full Fine-Tuning"].append(dur)
            h, rem = divmod(dur, 3600); m, s = divmod(rem, 60)
            print(f"Full FT (Fold {fold+1}) time: {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint["completed_approaches"][str(fold)].append("full")
            checkpoint["current_fold"] = fold
            checkpoint["full_fold"]    = full_fold
            checkpoint["text_fold"]    = text_fold
            checkpoint["vision_fold"]  = vision_fold
            checkpoint["partial_fold"] = partial_fold
            checkpoint["timing"]       = timing
            with open(ckpt_path, "w") as f: json.dump(checkpoint, f)

        # 2) Text Encoder Fine-Tuning
        if "text" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining: Text Encoder Fine-Tuning")
            start = time.time()

            text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)

            log_dir = os.path.join(base_dir, "text_encoder_finetune", f"fold_{fold+1}")
            text_metrics = train_model(
                text_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device,
                log_dir=log_dir, fold=fold+1, accumulation_steps=4, validate_every=2,
                model_type="Text Encoder Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_text_encoder_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                text_model.load_state_dict(torch.load(best_path, map_location=device))

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                text_model, val_loader, criterion, subcategories, device, fold+1, 999, "Text Encoder Fine-Tuning", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                text_model, test_loader, criterion, subcategories, device, fold+1, 999, "Text Encoder Fine-Tuning", "Test"
            )

            text_fold["train_loss"].append(text_metrics["train_loss"][-1] if text_metrics["train_loss"] else 0)
            text_fold["train_acc"].append(text_metrics["train_acc"][-1] if text_metrics["train_acc"] else 0)
            text_fold["val_loss"].append(v_loss);  text_fold["val_acc"].append(v_acc)
            text_fold["test_loss"].append(t_loss); text_fold["test_acc"].append(t_acc)
            text_fold["precision"].append(v_p);    text_fold["recall"].append(v_r); text_fold["f1"].append(v_f1)

            save_metrics_to_csv(text_metrics, os.path.join(base_dir, f"metrics_text_encoder_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(base_dir, "text_encoder_finetune", f"fold_{fold+1}_logs"))

            dur = time.time() - start
            timing["Text Encoder Fine-Tuning"].append(dur)
            checkpoint["completed_approaches"][str(fold)].append("text")
            checkpoint["current_fold"] = fold
            checkpoint["timing"]       = timing
            checkpoint["text_fold"]    = text_fold
            with open(ckpt_path, "w") as f: json.dump(checkpoint, f)

        # 3) Vision-Only Fine-Tuning
        if "vision" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining: Vision-Only Fine-Tuning")
            start = time.time()

            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(vision_model.parameters(), lr=1e-4)

            log_dir = os.path.join(base_dir, "vision_only_finetune", f"fold_{fold+1}")
            vision_metrics = train_model(
                vision_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device,
                log_dir=log_dir, fold=fold+1, accumulation_steps=4, validate_every=2,
                model_type="Vision-Only Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_vision-only_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                vision_model.load_state_dict(torch.load(best_path, map_location=device))

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold+1, 999, "Vision-Only Fine-Tuning", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold+1, 999, "Vision-Only Fine-Tuning", "Test"
            )

            vision_fold["train_loss"].append(vision_metrics["train_loss"][-1] if vision_metrics["train_loss"] else 0)
            vision_fold["train_acc"].append(vision_metrics["train_acc"][-1] if vision_metrics["train_acc"] else 0)
            vision_fold["val_loss"].append(v_loss);  vision_fold["val_acc"].append(v_acc)
            vision_fold["test_loss"].append(t_loss); vision_fold["test_acc"].append(t_acc)
            vision_fold["precision"].append(v_p);    vision_fold["recall"].append(v_r); vision_fold["f1"].append(v_f1)

            save_metrics_to_csv(vision_metrics, os.path.join(base_dir, f"metrics_vision_only_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(base_dir, "vision_only_finetune", f"fold_{fold+1}_logs"))

            dur = time.time() - start
            timing["Vision-Only Fine-Tuning"].append(dur)
            checkpoint["completed_approaches"][str(fold)].append("vision")
            checkpoint["current_fold"] = fold
            checkpoint["timing"]       = timing
            checkpoint["vision_fold"]  = vision_fold
            with open(ckpt_path, "w") as f: json.dump(checkpoint, f)

        # 4) Partial Fine-Tuning (last 30% in both encoders; prompt similarity)
        if "partial" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining: Partial Fine-Tuning (Last 30% layers in BOTH encoders)")
            start = time.time()

            partial_model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            params = [p for p in partial_model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(params, lr=1e-4)

            log_dir = os.path.join(base_dir, "partial_finetune", f"fold_{fold+1}")
            partial_metrics = train_model(
                partial_model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs=4, subcategories=subcategories, device=device,
                log_dir=log_dir, fold=fold+1, accumulation_steps=4, validate_every=2,
                model_type="Partial Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_partial_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                partial_model.load_state_dict(torch.load(best_path, map_location=device))

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold+1, 999, "Partial Fine-Tuning", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold+1, 999, "Partial Fine-Tuning", "Test"
            )

            partial_fold["train_loss"].append(partial_metrics["train_loss"][-1] if partial_metrics["train_loss"] else 0)
            partial_fold["train_acc"].append(partial_metrics["train_acc"][-1] if partial_metrics["train_acc"] else 0)
            partial_fold["val_loss"].append(v_loss);  partial_fold["val_acc"].append(v_acc)
            partial_fold["test_loss"].append(t_loss); partial_fold["test_acc"].append(t_acc)
            partial_fold["precision"].append(v_p);    partial_fold["recall"].append(v_r); partial_fold["f1"].append(v_f1)

            save_metrics_to_csv(partial_metrics, os.path.join(base_dir, f"metrics_partial_finetune_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join(base_dir, "partial_finetune", f"fold_{fold+1}_logs"))

            dur = time.time() - start
            timing["Partial Fine-Tuning"].append(dur)
            checkpoint["completed_approaches"][str(fold)].append("partial")
            checkpoint["current_fold"] = fold
            checkpoint["timing"]       = timing
            checkpoint["partial_fold"] = partial_fold
            with open(ckpt_path, "w") as f: json.dump(checkpoint, f)

    # ------------------------------
    # After all folds: plots & tables
    # ------------------------------
    print("\nMetrics dict lengths before plotting:")
    for name, d in [("Full", full_fold), ("Text", text_fold), ("Vision", vision_fold), ("Partial", partial_fold)]:
        print(name, {k: len(v) for k, v in d.items()})

    # Per-approach fold plots
    plot_fold_metrics(full_fold,    k_folds, "Full Fine-Tuning")
    plot_fold_metrics(text_fold,    k_folds, "Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold,  k_folds, "Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold, k_folds, "Partial Fine-Tuning")

    # All approaches on common charts
    compare_metrics(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    # Focused Full vs Partial plot
    plot_full_vs_partial(full_fold, partial_fold, k_folds)

    # Fold-level comparison CSV (exact headers incl. Train/Val/Test Loss & Acc for each approach)
    fold_table = create_fold_comparison_table(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    # Save per-approach fold summary CSVs too
    def save_fold_summary(name, d):
        out = os.path.join(base_dir, f"kfold_{name}.csv")
        save_metrics_to_csv(d, out)
    save_fold_summary("full_finetune_metrics",    full_fold)
    save_fold_summary("text_encoder_metrics",     text_fold)
    save_fold_summary("vision_only_metrics",      vision_fold)
    save_fold_summary("partial_finetune_metrics", partial_fold)

    # Timing CSV
    timing_rows = []
    for approach, durations in timing.items():
        if durations:
            for i, dur in enumerate(durations, 1):
                h, rem = divmod(dur, 3600); m, s = divmod(rem, 60)
                timing_rows.append({
                    "Approach": approach, "Fold": i,
                    "Duration (seconds)": dur,
                    "Formatted Duration": f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                })
        else:
            for i in range(1, k_folds + 1):
                timing_rows.append({"Approach": approach, "Fold": i,
                                    "Duration (seconds)": 0, "Formatted Duration": "N/A"})
    timing_df = pd.DataFrame(timing_rows)
    timing_out = os.path.join(base_dir, "execution_timing.csv")
    timing_df.to_csv(timing_out, index=False)
    print(f"Timing saved: {timing_out}")

    overall_end = time.time()
    h, rem = divmod(overall_end - overall_start, 3600); m, s = divmod(rem, 60)
    print(f"\nOverall Training Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}")
    print(f"Total Duration: {int(h):02d}:{int(m):02d}:{int(s):02d}")