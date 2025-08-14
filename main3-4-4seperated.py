import os
import json
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from zipfile import ZipFile
from torch.utils.data import DataLoader, Subset
from torch.multiprocessing import freeze_support
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from datasets import load_dataset
import open_clip

warnings.filterwarnings("ignore", message="These pretrained weights were trained with QuickGELU activation")
warnings.filterwarnings("ignore", message="Repo card metadata block was not found")

# =========================================================
# OUTPUT ROOTS (ALL PNGs & CSVs go here)
# =========================================================
PNG_DIR = os.path.join("logs", "main3-4aa", "png")
CSV_DIR = os.path.join("logs", "main3-4aa", "csv")
os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# =========================================================
# Dataset
# =========================================================
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

# =========================================================
# Models
# =========================================================
class FullFineTunedCLIP(nn.Module):
    """
    Vision encoder + classifier head.
    freeze_encoder=False => all visual params + head train.
    """
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=False):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.is_encoder_frozen = freeze_encoder
        if freeze_encoder:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

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
        logits = self.classifier(feats)
        return logits


class TextEncoderFineTunedCLIP(nn.Module):
    """Freeze visual, train text; uses similarity logits."""
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
        img_feats = self.visual_encoder(images)
        if text_inputs is None:
            text_inputs = self.tokenizer([f"a photo of {c}" for c in self.subcategories]).to(self.device)
        txt_feats = self.encode_text(text_inputs)

        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        logits = torch.matmul(img_feats, txt_feats.T) * 100.0
        return logits


class VisionOnlyFineTunedCLIP(nn.Module):
    """Train visual encoder + classifier head."""
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
        logits = self.classifier(feats)
        return logits


class PartialFineTunedCLIP(nn.Module):
    """
    Unfreeze the LAST ~30% of parameters in BOTH encoders.
    Uses similarity logits when text_inputs is provided.
    """
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy_input).shape[1]

        self.classifier = nn.Linear(out_dim, num_classes)  # fallback path

        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder,   freeze_percentage, "text")

        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding

    def _freeze_layers(self, module, freeze_percentage, encoder_type):
        named_params = list(module.named_parameters())
        total = len(named_params)
        freeze_until = int(total * freeze_percentage)
        print(f"Freezing first {freeze_percentage*100:.1f}% of {encoder_type} encoder tensors "
              f"({freeze_until}/{total})")
        for i, (_, p) in enumerate(named_params):
            p.requires_grad = i >= freeze_until

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
        img_feats = self.visual_encoder(images)
        if text_inputs is not None:
            txt_feats = self.encode_text(text_inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            logits = torch.matmul(img_feats, txt_feats.T) * 100.0
            return logits
        return self.classifier(img_feats)

# =========================================================
# Utilities
# =========================================================
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

# ---------- File saving helpers: ALWAYS route to CSV_DIR / PNG_DIR ----------
def save_metrics_to_csv(metrics, filename):
    """Save CSV to CSV_DIR (basename of filename is kept)."""
    df = pd.DataFrame(metrics)
    basename = os.path.basename(filename) if filename else "metrics.csv"
    final_path = os.path.join(CSV_DIR, basename)
    df.to_csv(final_path, index=False)
    print(f"Metrics saved to {final_path}")

def save_dataframe_to_csv(df, filename):
    basename = os.path.basename(filename) if filename else "table.csv"
    final_path = os.path.join(CSV_DIR, basename)
    df.to_csv(final_path, index=False)
    print(f"Table saved to {final_path}")

def save_png_current_figure(filename_base):
    """Save current Matplotlib figure to PNG_DIR."""
    basename = os.path.basename(filename_base)
    if not basename.lower().endswith(".png"):
        basename += ".png"
    final_path = os.path.join(PNG_DIR, basename)
    plt.savefig(final_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved to {final_path}")

def save_logs_as_zip(src_dir, zip_base_name):
    """Optional: zip raw logs (not PNG/CSV); outputs to the same folder as zip_base_name."""
    if os.path.exists(src_dir):
        zip_path = f"{zip_base_name}.zip"
        with ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(src_dir):
                for file in files:
                    zipf.write(os.path.join(root, file),
                               os.path.join(root.replace(src_dir, os.path.basename(src_dir)), file))
        print(f"Logs zipped to {zip_path}")
        shutil.rmtree(src_dir, ignore_errors=True)

# ---------- Plotting / comparison ----------
def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    print(f"Metrics for {title_suffix}:")
    for k, v in metrics.items():
        print(f"{k}: len={len(v)}")

    metric_lengths = [len(metrics[k]) for k in metrics if len(metrics[k]) > 0]
    if not metric_lengths:
        print(f"Warning: No metrics data for {title_suffix}. Skipping plot.")
        return
    num_points = min(metric_lengths)
    folds = list(range(1, num_points + 1))

    plt.figure(figsize=(14, 10))

    # Val Loss
    plt.subplot(2, 2, 1)
    if len(metrics.get("val_loss", [])) > 0:
        plt.plot(folds[:len(metrics["val_loss"])], metrics["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Fold"); plt.ylabel("Val Loss"); plt.title(f"Val Loss - {title_suffix}"); plt.legend()

    # Val Acc
    plt.subplot(2, 2, 2)
    if len(metrics.get("val_acc", [])) > 0:
        plt.plot(folds[:len(metrics["val_acc"])], metrics["val_acc"], marker="o", label="Val Acc")
    plt.xlabel("Fold"); plt.ylabel("Val Acc"); plt.title(f"Val Acc - {title_suffix}"); plt.legend()

    # Precision
    plt.subplot(2, 2, 3)
    if len(metrics.get("precision", [])) > 0:
        plt.plot(folds[:len(metrics["precision"])], metrics["precision"], marker="o", label="Precision")
    plt.xlabel("Fold"); plt.ylabel("Precision"); plt.title(f"Precision - {title_suffix}"); plt.legend()

    # F1
    plt.subplot(2, 2, 4)
    if len(metrics.get("f1", [])) > 0:
        plt.plot(folds[:len(metrics["f1"])], metrics["f1"], marker="o", label="F1")
    plt.xlabel("Fold"); plt.ylabel("F1"); plt.title(f"F1 - {title_suffix}"); plt.legend()

    save_png_current_figure(f"fold_metrics_{title_suffix.replace(' ', '_')}.png")

def compare_metrics(full_m, text_m, vision_m, partial_m, k_folds):
    print("Preparing comparison plots...")

    max_len = max(
        len(full_m.get("val_loss", [])),
        len(text_m.get("val_loss", [])),
        len(vision_m.get("val_loss", [])),
        len(partial_m.get("val_loss", [])),
        1
    )
    folds = list(range(1, max_len + 1))
    plt.figure(figsize=(15, 15))

    def plot_line(ax_idx, key, title, ylabel):
        plt.subplot(3, 2, ax_idx)
        for name, m, marker in [
            ("Full", full_m, "o"),
            ("Text", text_m, "x"),
            ("Vision", vision_m, "s"),
            ("Partial", partial_m, "d")
        ]:
            vals = m.get(key, [])
            if len(vals) > 0:
                plt.plot(folds[:len(vals)], vals, marker=marker, label=name)
        plt.xlabel("Fold"); plt.ylabel(ylabel); plt.title(title); plt.legend()

    plot_line(1, "val_loss",  "Validation Loss Comparison", "Val Loss")
    plot_line(2, "test_loss", "Test Loss Comparison",       "Test Loss")
    plot_line(3, "val_acc",   "Validation Accuracy Comparison", "Val Acc")
    plot_line(4, "test_acc",  "Test Accuracy Comparison",       "Test Acc")
    plot_line(5, "f1",        "F1 (Validation) Comparison",     "F1 (Val)")

    save_png_current_figure("comparison_metrics_all.png")

def create_comparison_table(full_m, text_m, vision_m, partial_m, k_folds):
    def get(m, key, i):
        return m.get(key, [])[i] if i < len(m.get(key, [])) else 0

    rows = []
    for i in range(k_folds):
        rows.append({
            'Fold': i + 1,
            'Full Train Loss': get(full_m, 'train_loss', i),
            'Full Train Acc':  get(full_m, 'train_acc', i),
            'Full Val Loss':   get(full_m, 'val_loss', i),
            'Full Val Acc':    get(full_m, 'val_acc', i),
            'Full Test Loss':  get(full_m, 'test_loss', i),
            'Full Test Acc':   get(full_m, 'test_acc', i),

            'Text Train Loss': get(text_m, 'train_loss', i),
            'Text Train Acc':  get(text_m, 'train_acc', i),
            'Text Val Loss':   get(text_m, 'val_loss', i),
            'Text Val Acc':    get(text_m, 'val_acc', i),
            'Text Test Loss':  get(text_m, 'test_loss', i),
            'Text Test Acc':   get(text_m, 'test_acc', i),

            'Vision Train Loss': get(vision_m, 'train_loss', i),
            'Vision Train Acc':  get(vision_m, 'train_acc', i),
            'Vision Val Loss':   get(vision_m, 'val_loss', i),
            'Vision Val Acc':    get(vision_m, 'val_acc', i),
            'Vision Test Loss':  get(vision_m, 'test_loss', i),
            'Vision Test Acc':   get(vision_m, 'test_acc', i),

            'Partial Train Loss': get(partial_m, 'train_loss', i),
            'Partial Train Acc':  get(partial_m, 'train_acc', i),
            'Partial Val Loss':   get(partial_m, 'val_loss', i),
            'Partial Val Acc':    get(partial_m, 'val_acc', i),
            'Partial Test Loss':  get(partial_m, 'test_loss', i),
            'Partial Test Acc':   get(partial_m, 'test_acc', i),
        })

    def avg(lst): return float(np.mean(lst)) if lst else 0.0

    rows.append({
        'Fold': 'Average',
        'Full Train Loss':   avg(full_m.get('train_loss', [])),
        'Full Train Acc':    avg(full_m.get('train_acc', [])),
        'Full Val Loss':     avg(full_m.get('val_loss', [])),
        'Full Val Acc':      avg(full_m.get('val_acc', [])),
        'Full Test Loss':    avg(full_m.get('test_loss', [])),
        'Full Test Acc':     avg(full_m.get('test_acc', [])),

        'Text Train Loss':   avg(text_m.get('train_loss', [])),
        'Text Train Acc':    avg(text_m.get('train_acc', [])),
        'Text Val Loss':     avg(text_m.get('val_loss', [])),
        'Text Val Acc':      avg(text_m.get('val_acc', [])),
        'Text Test Loss':    avg(text_m.get('test_loss', [])),
        'Text Test Acc':     avg(text_m.get('test_acc', [])),

        'Vision Train Loss': avg(vision_m.get('train_loss', [])),
        'Vision Train Acc':  avg(vision_m.get('train_acc', [])),
        'Vision Val Loss':   avg(vision_m.get('val_loss', [])),
        'Vision Val Acc':    avg(vision_m.get('val_acc', [])),
        'Vision Test Loss':  avg(vision_m.get('test_loss', [])),
        'Vision Test Acc':   avg(vision_m.get('test_acc', [])),

        'Partial Train Loss': avg(partial_m.get('train_loss', [])),
        'Partial Train Acc':  avg(partial_m.get('train_acc', [])),
        'Partial Val Loss':   avg(partial_m.get('val_loss', [])),
        'Partial Val Acc':    avg(partial_m.get('val_acc', [])),
        'Partial Test Loss':  avg(partial_m.get('test_loss', [])),
        'Partial Test Acc':   avg(partial_m.get('test_acc', [])),
    })

    df = pd.DataFrame(rows)
    save_dataframe_to_csv(df, "comparison_table_all.csv")
    return df

def analyze_misclassifications(conf_matrix, subcats, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    misclass = []
    for t in range(num_classes):
        total_true = np.sum(conf_matrix[t])
        if total_true == 0:
            continue
        for p in range(num_classes):
            if t != p and conf_matrix[t][p] > 0:
                count = conf_matrix[t][p]
                pct = (count / total_true) * 100
                true_label = subcats[t] if t < len(subcats) else f"Unknown_{t}"
                pred_label = subcats[p] if p < len(subcats) else f"Unknown_{p}"
                misclass.append({
                    'True Class': true_label,
                    'Predicted Class': pred_label,
                    'Count': count,
                    'Percentage of True Class': pct
                })
    if misclass:
        df = pd.DataFrame(misclass).sort_values(by='Count', ascending=False).head(10)
        fname = f"misclass_{model_type.replace(' ', '_').lower()}_fold{fold}_epoch{epoch}_{split_type.lower()}.csv"
        save_dataframe_to_csv(df, fname)
        return df
    return None

def plot_confusion_matrix(conf_matrix, subcats, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    labels = subcats[:num_classes] if num_classes <= len(subcats) else [f"Class_{i}" for i in range(num_classes)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"confmat_{model_type.replace(' ', '_').lower()}_fold{fold}_epoch{epoch}_{split_type.lower()}.png"
    save_png_current_figure(fname)

def evaluate_model_with_metrics(model, dataloader, criterion, subcats, device, fold, epoch, model_type,
                                split_type="Validation"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    texts = None
    if hasattr(model, "tokenizer"):
        texts = model.tokenizer([f"a photo of {c}" for c in subcats]).to(device)

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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(conf_matrix, subcats, model_type, fold, epoch, split_type)
    _ = analyze_misclassifications(conf_matrix, subcats, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): "
          f"Loss={avg_loss:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, conf_matrix

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcats, device,
                fold, accumulation_steps=4, validate_every=2, model_type="Model"):
    scaler = GradScaler('cuda')
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
            texts = model.tokenizer([f"a photo of {c}" for c in subcats]).to(device)

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

        # Validate + Test periodically
        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                model, val_loader, criterion, subcats, device, fold, epoch + 1, model_type, "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcats, device, fold, epoch + 1, model_type, "Test"
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
                # save best weights (pth not restricted by PNG/CSV policy)
                save_path = os.path.join("logs", "main3-4aa", f'best_{model_type.replace(" ", "_").lower()}_fold{fold}.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Best {model_type} model saved at {save_path} with Val F1={best_f1:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping for {model_type} at epoch {epoch + 1}")
                    break
        else:
            # still log train metrics for continuity
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

        print(f"{model_type} - Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    return metrics

# =========================================================
# Main
# =========================================================
if __name__ == '__main__':
    freeze_support()
    overall_start = time.time()
    print(f"\nTraining started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}")

    device = setup_environment()
    clip_model, preprocess_train, preprocess_val = load_clip_model(device, "ViT-B-32", "openai")
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    def init_fold_metrics():
        return {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "test_loss": [], "test_acc": [],
            "precision": [], "recall": [], "f1": []
        }

    full_fold = init_fold_metrics()
    text_fold = init_fold_metrics()
    vision_fold = init_fold_metrics()
    partial_fold = init_fold_metrics()

    timing = {
        "Full Fine-Tuning": [],
        "Text Encoder Fine-Tuning": [],
        "Vision-Only Fine-Tuning": [],
        "Partial Fine-Tuning": []
    }

    checkpoint_file = os.path.join("logs", "main3-4aa", "checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    checkpoint = {"current_fold": 0, "completed": {}}

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: resume from fold {checkpoint['current_fold'] + 1}")
            for key_name, var_ref in [
                ("full_fold", full_fold),
                ("text_fold", text_fold),
                ("vision_fold", vision_fold),
                ("partial_fold", partial_fold),
                ("timing", timing)
            ]:
                if key_name in checkpoint:
                    locals()[key_name] = checkpoint[key_name]
        except Exception as e:
            print(f"Checkpoint load error: {e}. Starting fresh.")

    num_workers = min(8, os.cpu_count())
    print(f"Using {num_workers} workers.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        if fold < checkpoint.get('current_fold', 0):
            print(f"\nSkip Fold {fold + 1}/{k_folds} (already done)")
            continue

        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        train_val_size = len(train_val_idx)
        train_size = int(0.8 * train_val_size)
        train_indices = train_val_idx[:train_size]
        val_indices   = train_val_idx[train_size:]

        train_subset = Subset(dataset, train_indices.tolist())
        val_subset   = Subset(dataset, val_indices.tolist())
        test_subset  = Subset(dataset, test_idx.tolist())

        train_ds = FashionDataset(train_subset, subcategories, augment=True)
        val_ds   = FashionDataset(val_subset,   subcategories)
        test_ds  = FashionDataset(test_subset,  subcategories)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        if str(fold) not in checkpoint['completed']:
            checkpoint['completed'][str(fold)] = []

        # -------- 1) FULL FINE-TUNING --------
        if 'full' not in checkpoint['completed'][str(fold)]:
            print("\nTraining: Full Fine-Tuning")
            start = time.time()
            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            optimizer_full = optim.AdamW(full_model.parameters(), lr=1e-5, weight_decay=1e-4)

            full_metrics = train_model(
                full_model, train_loader, val_loader, test_loader, criterion, optimizer_full, num_epochs=4,
                subcats=subcategories, device=device, fold=fold + 1, accumulation_steps=4, validate_every=2,
                model_type="Full Fine-Tuning"
            )

            # Final eval with best weights if saved
            best_pth = os.path.join("logs", "main3-4aa", f'best_full_fine-tuning_fold{fold+1}.pth')
            if os.path.exists(best_pth):
                full_model.load_state_dict(torch.load(best_pth, map_location=device))
                print(f"Loaded best Full model from {best_pth}")

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                full_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                full_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            full_fold["train_loss"].append(full_metrics['train_loss'][-1] if full_metrics['train_loss'] else 0)
            full_fold["train_acc"].append(full_metrics['train_acc'][-1] if full_metrics['train_acc'] else 0)
            full_fold["val_loss"].append(v_loss);   full_fold["val_acc"].append(v_acc)
            full_fold["test_loss"].append(t_loss);  full_fold["test_acc"].append(t_acc)
            full_fold["precision"].append(v_p);     full_fold["recall"].append(v_r); full_fold["f1"].append(v_f1)

            save_metrics_to_csv(full_metrics, f"metrics_full_finetune_fold_{fold + 1}.csv")

            duration = time.time() - start
            timing["Full Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Full Fine-Tuning fold {fold + 1} took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed'][str(fold)].append('full')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold'] = full_fold
            checkpoint['text_fold'] = text_fold
            checkpoint['vision_fold'] = vision_fold
            checkpoint['partial_fold'] = partial_fold
            checkpoint['timing'] = timing
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        # -------- 2) TEXT ENCODER FINE-TUNING --------
        if 'text' not in checkpoint['completed'][str(fold)]:
            print("\nTraining: Text Encoder Fine-Tuning")
            start = time.time()
            text_model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            optimizer_text = optim.AdamW(text_model.text_encoder.parameters(), lr=1e-4)

            text_metrics = train_model(
                text_model, train_loader, val_loader, test_loader, criterion, optimizer_text, num_epochs=4,
                subcats=subcategories, device=device, fold=fold + 1, accumulation_steps=4, validate_every=2,
                model_type="Text Encoder Fine-Tuning"
            )

            best_pth = os.path.join("logs", "main3-4aa", f'best_text_encoder_fine-tuning_fold{fold+1}.pth')
            if os.path.exists(best_pth):
                text_model.load_state_dict(torch.load(best_pth, map_location=device))
                print(f"Loaded best Text model from {best_pth}")

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                text_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                text_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            text_fold["train_loss"].append(text_metrics['train_loss'][-1] if text_metrics['train_loss'] else 0)
            text_fold["train_acc"].append(text_metrics['train_acc'][-1] if text_metrics['train_acc'] else 0)
            text_fold["val_loss"].append(v_loss);   text_fold["val_acc"].append(v_acc)
            text_fold["test_loss"].append(t_loss);  text_fold["test_acc"].append(t_acc)
            text_fold["precision"].append(v_p);     text_fold["recall"].append(v_r); text_fold["f1"].append(v_f1)

            save_metrics_to_csv(text_metrics, f"metrics_text_encoder_fold_{fold + 1}.csv")

            duration = time.time() - start
            timing["Text Encoder Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Text FT fold {fold + 1} took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed'][str(fold)].append('text')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold'] = full_fold
            checkpoint['text_fold'] = text_fold
            checkpoint['vision_fold'] = vision_fold
            checkpoint['partial_fold'] = partial_fold
            checkpoint['timing'] = timing
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        # -------- 3) VISION-ONLY FINE-TUNING --------
        if 'vision' not in checkpoint['completed'][str(fold)]:
            print("\nTraining: Vision-Only Fine-Tuning")
            start = time.time()
            vision_model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            optimizer_vision = optim.AdamW(vision_model.parameters(), lr=1e-4)

            vision_metrics = train_model(
                vision_model, train_loader, val_loader, test_loader, criterion, optimizer_vision, num_epochs=4,
                subcats=subcategories, device=device, fold=fold + 1, accumulation_steps=4, validate_every=2,
                model_type="Vision-Only Fine-Tuning"
            )

            best_pth = os.path.join("logs", "main3-4aa", f'best_vision-only_fine-tuning_fold{fold+1}.pth')
            if os.path.exists(best_pth):
                vision_model.load_state_dict(torch.load(best_pth, map_location=device))
                print(f"Loaded best Vision model from {best_pth}")

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            vision_fold["train_loss"].append(vision_metrics['train_loss'][-1] if vision_metrics['train_loss'] else 0)
            vision_fold["train_acc"].append(vision_metrics['train_acc'][-1] if vision_metrics['train_acc'] else 0)
            vision_fold["val_loss"].append(v_loss);   vision_fold["val_acc"].append(v_acc)
            vision_fold["test_loss"].append(t_loss);  vision_fold["test_acc"].append(t_acc)
            vision_fold["precision"].append(v_p);     vision_fold["recall"].append(v_r); vision_fold["f1"].append(v_f1)

            save_metrics_to_csv(vision_metrics, f"metrics_vision_only_fold_{fold + 1}.csv")

            duration = time.time() - start
            timing["Vision-Only Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Vision FT fold {fold + 1} took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed'][str(fold)].append('vision')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold'] = full_fold
            checkpoint['text_fold'] = text_fold
            checkpoint['vision_fold'] = vision_fold
            checkpoint['partial_fold'] = partial_fold
            checkpoint['timing'] = timing
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        # -------- 4) PARTIAL (LAST 30% of BOTH encoders) --------
        if 'partial' not in checkpoint['completed'][str(fold)]:
            print("\nTraining: Partial Fine-Tuning (last 30% of both encoders)")
            start = time.time()
            partial_model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            params_partial = [p for p in partial_model.parameters() if p.requires_grad]
            optimizer_partial = optim.AdamW(params_partial, lr=1e-4)

            partial_metrics = train_model(
                partial_model, train_loader, val_loader, test_loader, criterion, optimizer_partial, num_epochs=4,
                subcats=subcategories, device=device, fold=fold + 1, accumulation_steps=4, validate_every=2,
                model_type="Partial Fine-Tuning"
            )

            best_pth = os.path.join("logs", "main3-4aa", f'best_partial_fine-tuning_fold{fold+1}.pth')
            if os.path.exists(best_pth):
                partial_model.load_state_dict(torch.load(best_pth, map_location=device))
                print(f"Loaded best Partial model from {best_pth}")

            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            partial_fold["train_loss"].append(partial_metrics['train_loss'][-1] if partial_metrics['train_loss'] else 0)
            partial_fold["train_acc"].append(partial_metrics['train_acc'][-1] if partial_metrics['train_acc'] else 0)
            partial_fold["val_loss"].append(v_loss);   partial_fold["val_acc"].append(v_acc)
            partial_fold["test_loss"].append(t_loss);  partial_fold["test_acc"].append(t_acc)
            partial_fold["precision"].append(v_p);     partial_fold["recall"].append(v_r); partial_fold["f1"].append(v_f1)

            save_metrics_to_csv(partial_metrics, f"metrics_partial_finetune_fold_{fold + 1}.csv")

            duration = time.time() - start
            timing["Partial Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Partial FT fold {fold + 1} took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed'][str(fold)].append('partial')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold'] = full_fold
            checkpoint['text_fold'] = text_fold
            checkpoint['vision_fold'] = vision_fold
            checkpoint['partial_fold'] = partial_fold
            checkpoint['timing'] = timing
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

    # -------- After all folds: plots & CSVs (PNG/CSV dirs enforced) --------
    print("\nFinal metric dict sizes:")
    print("Full:",    {k: len(v) for k, v in full_fold.items()})
    print("Text:",    {k: len(v) for k, v in text_fold.items()})
    print("Vision:",  {k: len(v) for k, v in vision_fold.items()})
    print("Partial:", {k: len(v) for k, v in partial_fold.items()})

    # Per-approach fold plots
    plot_fold_metrics(full_fold,    k_folds, title_suffix="Full Fine-Tuning")
    plot_fold_metrics(text_fold,    k_folds, title_suffix="Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold,  k_folds, title_suffix="Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold, k_folds, title_suffix="Partial Fine-Tuning")

    # Comparison plot
    compare_metrics(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    # Persist per-approach kfold CSVs
    save_metrics_to_csv(full_fold,    "kfold_full_finetune_metrics.csv")
    save_metrics_to_csv(text_fold,    "kfold_text_encoder_metrics.csv")
    save_metrics_to_csv(vision_fold,  "kfold_vision_only_metrics.csv")
    save_metrics_to_csv(partial_fold, "kfold_partial_finetune_metrics.csv")

    # Comparison table (includes train/val/test loss & acc)
    _ = create_comparison_table(full_fold, text_fold, vision_fold, partial_fold, k_folds)

    # Timing CSV
    timing_records = []
    for approach, durations in timing.items():
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
            for i in range(1, k_folds + 1):
                timing_records.append({
                    "Approach": approach, "Fold": i,
                    "Duration (seconds)": 0, "Formatted Duration": "N/A"
                })
    timing_df = pd.DataFrame(timing_records)
    save_dataframe_to_csv(timing_df, "execution_timing.csv")

    # Print summaries
    def safe_avg(d, key): return float(np.mean(d.get(key, []))) if d.get(key, []) else 0.0
    print("\nSummary of Average Metrics Across Folds:")
    for name, d in [("Full Fine-Tuning", full_fold),
                    ("Text Encoder Fine-Tuning", text_fold),
                    ("Vision-Only Fine-Tuning", vision_fold),
                    ("Partial Fine-Tuning (Last 30%)", partial_fold)]:
        print(f"\n{name}:")
        print(f"Avg Train Loss: {safe_avg(d, 'train_loss'):.4f} | Avg Train Acc: {safe_avg(d, 'train_acc'):.4f}")
        print(f"Avg Val   Loss: {safe_avg(d, 'val_loss'):.4f}   | Avg Val   Acc: {safe_avg(d, 'val_acc'):.4f}")
        print(f"Avg Test  Loss: {safe_avg(d, 'test_loss'):.4f}  | Avg Test  Acc: {safe_avg(d, 'test_acc'):.4f}")
        print(f"Avg Precision:  {safe_avg(d, 'precision'):.4f} | Avg Recall: {safe_avg(d, 'recall'):.4f} | Avg F1: {safe_avg(d, 'f1'):.4f}")

    overall_end = time.time()
    h, rem = divmod(overall_end - overall_start, 3600); m, s = divmod(rem, 60)
    print(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}")
    print(f"Total Duration: {int(h):02d}:{int(m):02d}:{int(s):02d}")
