import os
import json
import time
import shutil
import warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import KFold
from torch.multiprocessing import freeze_support
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

# AMP (use the cuda variant for widest compatibility)
from torch.cuda.amp import autocast, GradScaler

import open_clip

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
    Full fine-tuning of the vision encoder + classifier head (freeze_encoder=False).
    Classifier head over image features.
    """
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=False):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.is_encoder_frozen = freeze_encoder
        for p in self.visual_encoder.parameters():
            p.requires_grad = not freeze_encoder

        device = next(base_model.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy_input).shape[1]

        layers, in_dim = [], out_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5)]
            in_dim = 512
        layers += [nn.Linear(in_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        feats = self.visual_encoder(images)
        return self.classifier(feats)


class TextEncoderFineTunedCLIP(nn.Module):
    """
    Freeze visual; fine-tune text (transformer). Uses CLIP similarity logits.
    """
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = not freeze_visual

        self.text_encoder = base_model.transformer
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device
        self.subcategories = subcategories

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
        logits = torch.matmul(image_features, text_features.T) * 100.0
        return logits


class VisionOnlyFineTunedCLIP(nn.Module):
    """
    Fine-tune the vision encoder (all of it) + classifier head.
    """
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = True

        device = next(base_model.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy_input).shape[1]

        layers, in_dim = [], out_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.5)]
            in_dim = 512
        layers += [nn.Linear(in_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, images, text_inputs=None):
        feats = self.visual_encoder(images)
        return self.classifier(feats)


class PartialFineTunedCLIP(nn.Module):
    """
    Unfreeze the LAST ~30% of parameters in BOTH encoders.
    Uses similarity logits (image·text) so both encoders participate.
    """
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device
        self.num_classes    = num_classes

        # Freeze first X% parameter tensors for both encoders
        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder,   freeze_percentage, "text")

        # Projection/text blocks from CLIP
        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding

    @staticmethod
    def _freeze_layers(module, freeze_percentage, name):
        params = list(module.named_parameters())
        cut = int(len(params) * freeze_percentage)
        for i, (_, p) in enumerate(params):
            p.requires_grad = i >= cut
        frozen = sum(p.numel() for _, p in params[:cut])
        trainable = sum(p.numel() for _, p in params[cut:])
        print(f"{name.capitalize()} encoder: Frozen {frozen} params, Trainable {trainable} params "
              f"({freeze_percentage*100:.1f}% frozen)")

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
            text_inputs = self.tokenizer([f"a photo of {i}" for i in range(self.num_classes)]).to(self.device)
        text_features = self.encode_text(text_inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        logits = torch.matmul(image_features, text_features.T) * 100.0
        return logits

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
                for f in files:
                    absp = os.path.join(root, f)
                    relp = os.path.join(root.replace(log_dir, log_dir.split('/')[-2]), f)
                    zipf.write(absp, relp)
        print(f"Logs zipped to {zip_name}.zip")
        shutil.rmtree(log_dir, ignore_errors=True)

def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    # metrics dict expected keys: train_loss, val_loss, test_loss, val_acc, test_acc
    print(f"Metrics for {title_suffix}:")
    for k, v in metrics.items():
        print(f"{k}: {len(v)} values")

    def _plot(ax, xs, ys, label, ylabel, title):
        if len(ys) > 0:
            ax.plot(xs[:len(ys)], ys, marker="o", label=label)
            ax.set_xlabel("Fold")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()

    max_len = max([len(metrics.get(k, [])) for k in ["val_loss", "test_loss", "val_acc", "test_acc"]] + [k_folds])
    folds = range(1, max_len + 1)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    _plot(axs[0,0], folds, metrics.get("val_loss", []),  "Val Loss",  "Loss", f"{title_suffix}: Val Loss")
    _plot(axs[0,1], folds, metrics.get("test_loss", []), "Test Loss", "Loss", f"{title_suffix}: Test Loss")
    _plot(axs[1,0], folds, metrics.get("val_acc", []),   "Val Acc",   "Accuracy", f"{title_suffix}: Val Acc")
    _plot(axs[1,1], folds, metrics.get("test_acc", []),  "Test Acc",  "Accuracy", f"{title_suffix}: Test Acc")

    plt.tight_layout()
    save_path = os.path.join("logs/main3-4a", f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Plot saved to {save_path}")

def compare_metrics_graph(all_metrics, k_folds):
    """
    all_metrics: dict of name -> metrics dict (train_loss, val_loss, test_loss, val_acc, test_acc)
    Saves a single 'graph of all' with val/test loss & accuracy for each approach across folds.
    """
    max_len = max([max(len(m.get("val_loss", [])), len(m.get("test_loss", [])),
                       len(m.get("val_acc", [])),  len(m.get("test_acc", []))) for m in all_metrics.values()] + [k_folds])
    folds = range(1, max_len + 1)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    for name, m in all_metrics.items():
        if len(m.get("val_loss", [])) > 0:
            axs[0,0].plot(folds[:len(m["val_loss"])],  m["val_loss"],  marker="o", label=name)
        if len(m.get("test_loss", [])) > 0:
            axs[0,1].plot(folds[:len(m["test_loss"])], m["test_loss"], marker="o", label=name)
        if len(m.get("val_acc", [])) > 0:
            axs[1,0].plot(folds[:len(m["val_acc"])],   m["val_acc"],   marker="o", label=name)
        if len(m.get("test_acc", [])) > 0:
            axs[1,1].plot(folds[:len(m["test_acc"])],  m["test_acc"],  marker="o", label=name)

    axs[0,0].set_title("Validation Loss (All)"); axs[0,0].set_xlabel("Fold"); axs[0,0].set_ylabel("Loss"); axs[0,0].legend()
    axs[0,1].set_title("Test Loss (All)");       axs[0,1].set_xlabel("Fold"); axs[0,1].set_ylabel("Loss"); axs[0,1].legend()
    axs[1,0].set_title("Validation Accuracy (All)"); axs[1,0].set_xlabel("Fold"); axs[1,0].set_ylabel("Accuracy"); axs[1,0].legend()
    axs[1,1].set_title("Test Accuracy (All)");   axs[1,1].set_xlabel("Fold"); axs[1,1].set_ylabel("Accuracy"); axs[1,1].legend()

    plt.tight_layout()
    save_path = os.path.join("logs/main3-4a", "comparison_metrics_all.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Comparison plot saved to {save_path}")

def analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    misclass = []
    for t in range(num_classes):
        total_true = np.sum(conf_matrix[t])
        if total_true == 0: continue
        for p in range(num_classes):
            if t != p and conf_matrix[t][p] > 0:
                count = conf_matrix[t][p]
                pct = (count / total_true) * 100
                t_label = subcategories[t] if t < len(subcategories) else f"Unknown_{t}"
                p_label = subcategories[p] if p < len(subcategories) else f"Unknown_{p}"
                misclass.append({'True Class': t_label, 'Predicted Class': p_label,
                                 'Count': count, 'Percentage of True Class': pct})
    if not misclass:
        return None
    df = pd.DataFrame(misclass).sort_values(by='Count', ascending=False).head(10)
    save_path = os.path.join("logs/main3-4a", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                             f"misclassifications_epoch_{epoch}_{split_type.lower()}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Misclassifications saved to {save_path}")
    return df

def plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    labels = subcategories[:num_classes] if num_classes <= len(subcategories) else [f"Class_{i}" for i in range(num_classes)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join("logs/main3-4a", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                             f"confusion_matrix_epoch_{epoch}_{split_type.lower()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def evaluate_model_with_metrics(model, dataloader, criterion, subcategories, device, fold, epoch, model_type,
                                split_type="Validation"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    # Ensure same class order for similarity models
    texts = None
    if hasattr(model, "tokenizer"):
        texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, text_inputs=texts) if texts is not None else model(images)
            # Sanity: class dim must match #classes
            assert outputs.shape[1] == len(subcategories), \
                f"logit classes {outputs.shape[1]} != len(subcategories) {len(subcategories)}"
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Diagnostics
    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    print(f"[{model_type} {split_type}] correct/total = {correct}/{len(all_labels)}  (acc={acc:.4f})")

    plot_confusion_matrix(cm, subcategories, model_type, fold, epoch, split_type)
    analyze_misclassifications(cm, subcategories, model_type, fold, epoch, split_type)

    return avg_loss, acc, precision, recall, f1, cm

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcategories, device,
                log_dir, fold, accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
    scaler = GradScaler()
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

        texts = None
        if hasattr(model, "tokenizer"):
            texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

        with tqdm(train_loader, desc=f"{model_type} | Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = model(images, text_inputs=texts) if texts is not None else model(images)
                    loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train += labels.size(0)

                pbar.set_postfix({
                    "loss": f"{(total_train_loss / (batch_idx + 1)):.4f}",
                    "acc":  f"{(correct_train / max(total_train,1)):.4f}"
                })

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                model, val_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcategories, device, fold, epoch + 1, model_type, "Test"
            )

            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(avg_train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['test_loss'].append(test_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_acc'].append(val_acc)
            metrics['test_acc'].append(test_acc)
            metrics['precision_val'].append(val_prec)
            metrics['recall_val'].append(val_rec)
            metrics['f1_val'].append(val_f1)
            metrics['precision_test'].append(test_prec)
            metrics['recall_test'].append(test_rec)
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
            # still record train-only epoch for completeness
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

        print(f"{model_type} — Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

    return metrics

def create_wide_comparison_table(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds, save_path):
    """
    Produces a wide per-fold CSV like:
    Fold, <Approach> Train Loss/Acc, <Approach> Val Loss/Acc, <Approach> Test Loss/Acc, ...
    """
    def get(m, key, i): return m.get(key, [])[i] if i < len(m.get(key, [])) else 0.0

    rows = []
    for i in range(k_folds):
        row = {"Fold": i + 1}
        for label, m in [("Full", full_metrics), ("Text", text_metrics),
                         ("Vision", vision_metrics), ("Partial", partial_metrics)]:
            row[f"{label} Train Loss"] = get(m, "train_loss", i)
            row[f"{label} Train Acc"]  = get(m, "train_acc",  i)
            row[f"{label} Val Loss"]   = get(m, "val_loss",   i)
            row[f"{label} Val Acc"]    = get(m, "val_acc",    i)
            row[f"{label} Test Loss"]  = get(m, "test_loss",  i)
            row[f"{label} Test Acc"]   = get(m, "test_acc",   i)
        rows.append(row)

    # Average row
    def avg(lst): return float(np.mean(lst)) if lst else 0.0
    avg_row = {"Fold": "Average"}
    for label, m in [("Full", full_metrics), ("Text", text_metrics),
                     ("Vision", vision_metrics), ("Partial", partial_metrics)]:
        for key, nice in [("train_loss","Train Loss"), ("train_acc","Train Acc"),
                          ("val_loss","Val Loss"), ("val_acc","Val Acc"),
                          ("test_loss","Test Loss"), ("test_acc","Test Acc")]:
            avg_row[f"{label} {nice}"] = avg(m.get(key, []))
    rows.append(avg_row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Wide comparison table saved to {save_path}")
    return df

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    freeze_support()
    overall_start_time = time.time()
    print(f"\nOverall Training Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start_time))}")

    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device, model_name="ViT-B-32", pretrained_weights="openai"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    # K-Fold
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    def init_fold_metrics():
        return {"train_loss": [], "val_loss": [], "test_loss": [],
                "train_acc": [], "val_acc": [], "test_acc": []}

    full_fold_metrics    = init_fold_metrics()
    text_fold_metrics    = init_fold_metrics()
    vision_fold_metrics  = init_fold_metrics()
    partial_fold_metrics = init_fold_metrics()

    timing_results = {
        "Full Fine-Tuning": [],
        "Text Encoder Fine-Tuning": [],
        "Vision-Only Fine-Tuning": [],
        "Partial Fine-Tuning": []
    }

    checkpoint_file = os.path.join("logs/main3-4a", "checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: resume from fold {checkpoint['current_fold'] + 1}")
            for key in ["full_fold_metrics","text_fold_metrics","vision_fold_metrics","partial_fold_metrics","timing_results"]:
                if key in checkpoint:
                    locals()[key] = checkpoint[key]
        except Exception as e:
            print(f"Failed to load checkpoint, starting fresh: {e}")

    num_workers = min(8, os.cpu_count() or 2)
    print(f"Using {num_workers} workers for data loading.")

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        if fold < checkpoint["current_fold"]:
            print(f"\nSkipping Fold {fold+1}/{k_folds} — already completed.")
            continue

        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        tv_size = len(train_val_idx)
        train_size = int(0.8 * tv_size)
        train_idx = train_val_idx[:train_size]
        val_idx   = train_val_idx[train_size:]

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        test_subset  = Subset(dataset, test_idx.tolist())

        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset   = FashionDataset(val_subset,   subcategories)
        test_dataset  = FashionDataset(test_subset,  subcategories)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        if str(fold) not in checkpoint["completed_approaches"]:
            checkpoint["completed_approaches"][str(fold)] = []

        # 1) Full Fine-Tuning
        if "full" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining Full Fine-Tuned Model...")
            start = time.time()
            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(full_model.parameters(), lr=1e-5, weight_decay=1e-4)
            log_dir = os.path.join("logs/main3-4a", "full_finetune", f"fold_{fold+1}")

            full_metrics = train_model(
                full_model, train_loader, val_loader, test_loader,
                criterion, optimizer, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=log_dir, fold=fold+1,
                accumulation_steps=4, validate_every=2, model_type="Full Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_full_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                full_model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"Loaded best model from {best_path}")

            v_loss, v_acc, *_ = evaluate_model_with_metrics(
                full_model, val_loader, criterion, subcategories, device, fold+1, "Full Fine-Tuning", "Validation"
            )
            t_loss, t_acc, *_ = evaluate_model_with_metrics(
                full_model, test_loader, criterion, subcategories, device, fold+1, "Full Fine-Tuning", "Test"
            )

            full_fold_metrics["train_loss"].append(full_metrics["train_loss"][-1] if full_metrics["train_loss"] else 0.0)
            full_fold_metrics["train_acc"].append(full_metrics["train_acc"][-1] if full_metrics["train_acc"] else 0.0)
            full_fold_metrics["val_loss"].append(v_loss)
            full_fold_metrics["val_acc"].append(v_acc)
            full_fold_metrics["test_loss"].append(t_loss)
            full_fold_metrics["test_acc"].append(t_acc)

            save_metrics_to_csv(full_metrics, os.path.join("logs/main3-4a", f"metrics_full_finetune_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join("logs/main3-4a", "full_finetune", f"fold_{fold+1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Full Fine-Tuning"].append(duration)
            print(f"Full FT fold {fold+1} took {int(duration//3600):02d}:{int(duration%3600//60):02d}:{int(duration%60):02d}")

            checkpoint["completed_approaches"][str(fold)].append("full")
            checkpoint["current_fold"] = fold
            checkpoint["full_fold_metrics"]    = full_fold_metrics
            checkpoint["text_fold_metrics"]    = text_fold_metrics
            checkpoint["vision_fold_metrics"]  = vision_fold_metrics
            checkpoint["partial_fold_metrics"] = partial_fold_metrics
            checkpoint["timing_results"]       = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
        else:
            print("Skipping Full FT — already done.")

        # 2) Text Encoder Fine-Tuning
        if "text" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining Text Encoder Fine-Tuned Model...")
            start = time.time()
            text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)
            log_dir = os.path.join("logs/main3-4a", "text_encoder_finetune", f"fold_{fold+1}")

            text_metrics = train_model(
                text_model, train_loader, val_loader, test_loader,
                criterion, optimizer, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=log_dir, fold=fold+1,
                accumulation_steps=4, validate_every=2, model_type="Text Encoder Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_text_encoder_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                text_model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"Loaded best model from {best_path}")

            v_loss, v_acc, *_ = evaluate_model_with_metrics(
                text_model, val_loader, criterion, subcategories, device, fold+1, "Text Encoder Fine-Tuning", "Validation"
            )
            t_loss, t_acc, *_ = evaluate_model_with_metrics(
                text_model, test_loader, criterion, subcategories, device, fold+1, "Text Encoder Fine-Tuning", "Test"
            )

            text_fold_metrics["train_loss"].append(text_metrics["train_loss"][-1] if text_metrics["train_loss"] else 0.0)
            text_fold_metrics["train_acc"].append(text_metrics["train_acc"][-1] if text_metrics["train_acc"] else 0.0)
            text_fold_metrics["val_loss"].append(v_loss)
            text_fold_metrics["val_acc"].append(v_acc)
            text_fold_metrics["test_loss"].append(t_loss)
            text_fold_metrics["test_acc"].append(t_acc)

            save_metrics_to_csv(text_metrics, os.path.join("logs/main3-4a", f"metrics_text_encoder_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join("logs/main3-4a", "text_encoder_finetune", f"fold_{fold+1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Text Encoder Fine-Tuning"].append(duration)
            print(f"Text FT fold {fold+1} took {int(duration//3600):02d}:{int(duration%3600//60):02d}:{int(duration%60):02d}")

            checkpoint["completed_approaches"][str(fold)].append("text")
            checkpoint["current_fold"] = fold
            checkpoint["text_fold_metrics"]    = text_fold_metrics
            checkpoint["vision_fold_metrics"]  = vision_fold_metrics
            checkpoint["partial_fold_metrics"] = partial_fold_metrics
            checkpoint["timing_results"]       = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
        else:
            print("Skipping Text FT — already done.")

        # 3) Vision-Only Fine-Tuning
        if "vision" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining Vision-Only Fine-Tuned Model...")
            start = time.time()
            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.AdamW(vision_model.parameters(), lr=1e-4)
            log_dir = os.path.join("logs/main3-4a", "vision_only_finetune", f"fold_{fold+1}")

            vision_metrics = train_model(
                vision_model, train_loader, val_loader, test_loader,
                criterion, optimizer, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=log_dir, fold=fold+1,
                accumulation_steps=4, validate_every=2, model_type="Vision-Only Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_vision-only_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                vision_model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"Loaded best model from {best_path}")

            v_loss, v_acc, *_ = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold+1, "Vision-Only Fine-Tuning", "Validation"
            )
            t_loss, t_acc, *_ = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold+1, "Vision-Only Fine-Tuning", "Test"
            )

            vision_fold_metrics["train_loss"].append(vision_metrics["train_loss"][-1] if vision_metrics["train_loss"] else 0.0)
            vision_fold_metrics["train_acc"].append(vision_metrics["train_acc"][-1] if vision_metrics["train_acc"] else 0.0)
            vision_fold_metrics["val_loss"].append(v_loss)
            vision_fold_metrics["val_acc"].append(v_acc)
            vision_fold_metrics["test_loss"].append(t_loss)
            vision_fold_metrics["test_acc"].append(t_acc)

            save_metrics_to_csv(vision_metrics, os.path.join("logs/main3-4a", f"metrics_vision_only_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join("logs/main3-4a", "vision_only_finetune", f"fold_{fold+1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Vision-Only Fine-Tuning"].append(duration)
            print(f"Vision FT fold {fold+1} took {int(duration//3600):02d}:{int(duration%3600//60):02d}:{int(duration%60):02d}")

            checkpoint["completed_approaches"][str(fold)].append("vision")
            checkpoint["current_fold"] = fold
            checkpoint["vision_fold_metrics"]  = vision_fold_metrics
            checkpoint["partial_fold_metrics"] = partial_fold_metrics
            checkpoint["timing_results"]       = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
        else:
            print("Skipping Vision FT — already done.")

        # 4) Partial Fine-Tuning (last 30% of both encoders)
        if "partial" not in checkpoint["completed_approaches"][str(fold)]:
            print("\nTraining Partial Fine-Tuned Model (Last 30% of BOTH encoders)...")
            start = time.time()
            partial_model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            opt_params = [p for p in partial_model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(opt_params, lr=1e-4)
            log_dir = os.path.join("logs/main3-4a", "partial_finetune", f"fold_{fold+1}")

            partial_metrics = train_model(
                partial_model, train_loader, val_loader, test_loader,
                criterion, optimizer, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=log_dir, fold=fold+1,
                accumulation_steps=4, validate_every=2, model_type="Partial Fine-Tuning"
            )

            best_path = os.path.join(log_dir, "best_partial_fine-tuning_clip.pth")
            if os.path.exists(best_path):
                partial_model.load_state_dict(torch.load(best_path, map_location=device))
                print(f"Loaded best model from {best_path}")

            v_loss, v_acc, *_ = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold+1, "Partial Fine-Tuning", "Validation"
            )
            t_loss, t_acc, *_ = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold+1, "Partial Fine-Tuning", "Test"
            )

            partial_fold_metrics["train_loss"].append(partial_metrics["train_loss"][-1] if partial_metrics["train_loss"] else 0.0)
            partial_fold_metrics["train_acc"].append(partial_metrics["train_acc"][-1] if partial_metrics["train_acc"] else 0.0)
            partial_fold_metrics["val_loss"].append(v_loss)
            partial_fold_metrics["val_acc"].append(v_acc)
            partial_fold_metrics["test_loss"].append(t_loss)
            partial_fold_metrics["test_acc"].append(t_acc)

            save_metrics_to_csv(partial_metrics, os.path.join("logs/main3-4a", f"metrics_partial_finetune_fold_{fold+1}.csv"))
            save_logs_as_zip(log_dir, os.path.join("logs/main3-4a", "partial_finetune", f"fold_{fold+1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start
            timing_results["Partial Fine-Tuning"].append(duration)
            print(f"Partial FT fold {fold+1} took {int(duration//3600):02d}:{int(duration%3600//60):02d}:{int(duration%60):02d}")

            checkpoint["completed_approaches"][str(fold)].append("partial")
            checkpoint["current_fold"] = fold
            checkpoint["partial_fold_metrics"] = partial_fold_metrics
            checkpoint["timing_results"]       = timing_results
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)
        else:
            print("Skipping Partial FT — already done.")

    # --- Summary & Plots ---
    print("\nFinal fold metric lengths before plotting:")
    for name, d in [("Full",full_fold_metrics), ("Text",text_fold_metrics),
                    ("Vision",vision_fold_metrics), ("Partial",partial_fold_metrics)]:
        print(name, {k: len(v) for k, v in d.items()})

    plot_fold_metrics(full_fold_metrics,    k_folds, "Full Fine-Tuning")
    plot_fold_metrics(text_fold_metrics,    k_folds, "Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold_metrics,  k_folds, "Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold_metrics, k_folds, "Partial Fine-Tuning")

    compare_metrics_graph({
        "Full": full_fold_metrics,
        "Text": text_fold_metrics,
        "Vision": vision_fold_metrics,
        "Partial": partial_fold_metrics
    }, k_folds)

    # Persist kfold summaries
    save_metrics_to_csv(full_fold_metrics,    os.path.join("logs/main3-4a", "kfold_full_finetune_metrics.csv"))
    save_metrics_to_csv(text_fold_metrics,    os.path.join("logs/main3-4a", "kfold_text_encoder_metrics.csv"))
    save_metrics_to_csv(vision_fold_metrics,  os.path.join("logs/main3-4a", "kfold_vision_only_metrics.csv"))
    save_metrics_to_csv(partial_fold_metrics, os.path.join("logs/main3-4a", "kfold_partial_finetune_metrics.csv"))

    # Wide comparison table (all approaches, train/val/test loss+acc)
    create_wide_comparison_table(
        full_fold_metrics, text_fold_metrics, vision_fold_metrics, partial_fold_metrics,
        k_folds, save_path=os.path.join("logs/main3-4a", "comparison_table_wide.csv")
    )

    # Timing CSV
    timing_records = []
    for approach, durations in timing_results.items():
        if durations:
            for i, d in enumerate(durations, 1):
                h, rem = divmod(d, 3600); m, s = divmod(rem, 60)
                timing_records.append({
                    "Approach": approach,
                    "Fold": i,
                    "Duration (seconds)": d,
                    "Formatted Duration": f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                })
        else:
            for i in range(1, k_folds+1):
                timing_records.append({
                    "Approach": approach, "Fold": i,
                    "Duration (seconds)": 0, "Formatted Duration": "N/A"
                })
    timing_df = pd.DataFrame(timing_records)
    timing_save = os.path.join("logs/main3-4a", "execution_timing.csv")
    os.makedirs(os.path.dirname(timing_save), exist_ok=True)
    timing_df.to_csv(timing_save, index=False)
    print(f"Timing saved to {timing_save}")

    end = time.time()
    h, rem = divmod(end - overall_start_time, 3600); m, s = divmod(rem, 60)
    print(f"\nOverall Finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Total Duration: {int(h):02d}:{int(m):02d}:{int(s):02d}")
