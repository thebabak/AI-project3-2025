import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
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

# ------------------------------
# Models
# ------------------------------
class FullFineTunedCLIP(nn.Module):
    """
    Vision encoder + classifier head. With freeze_encoder=False we truly fine-tune ALL vision params + head.
    (This uses classification head over image features, not contrastive; adequate for the comparison.)
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
            output_size = self.visual_encoder(dummy_input).shape[1]

        layers = []
        in_dim = output_size
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
    """Train visual encoder + classifier head."""
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = True

        device = next(base_model.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]

        layers, in_dim = [], output_size
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
    Uses similarity logits when text_inputs is provided (so both encoders are on the path).
    """
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output_size = self.visual_encoder(dummy_input).shape[1]

        # If user ever wants to use classifier path; not used when text_inputs is passed
        self.classifier = nn.Linear(output_size, num_classes)

        self._freeze_layers(self.visual_encoder, freeze_percentage, "visual")
        self._freeze_layers(self.text_encoder,   freeze_percentage, "text")

        self.text_projection     = base_model.text_projection
        self.positional_embedding= base_model.positional_embedding
        self.ln_final            = base_model.ln_final
        self.token_embedding     = base_model.token_embedding

    def _freeze_layers(self, module, freeze_percentage, encoder_type):
        named_params = list(module.named_parameters())
        total_params = len(named_params)
        freeze_until = int(total_params * freeze_percentage)
        print(f"Freezing first {freeze_percentage*100:.1f}% of {encoder_type} encoder layers "
              f"({freeze_until}/{total_params} parameter tensors)")
        for i, (name, p) in enumerate(named_params):
            p.requires_grad = i >= freeze_until
        frozen_params    = sum(p.numel() for _, p in named_params[:freeze_until])
        trainable_params = sum(p.numel() for _, p in named_params[freeze_until:])
        print(f"{encoder_type.capitalize()} Encoder: Frozen {frozen_params} params, "
              f"Trainable {trainable_params} params")

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
        # fallback: classifier head on image features
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
    class_counts = np.bincount(labels, minlength=len(subcategories))
    total_samples = len(labels)
    weights = total_samples / (len(subcategories) * np.clip(class_counts, 1, None))
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
                    zipf.write(os.path.join(root, file),
                               os.path.join(root.replace(log_dir, log_dir.split('/')[-2]), file))
        print(f"Logs zipped to {zip_name}.zip")
        shutil.rmtree(log_dir, ignore_errors=True)

def plot_fold_metrics(metrics, k_folds, title_suffix=""):
    print(f"Metrics for {title_suffix}:")
    for k, v in metrics.items():
        print(f"{k}: {v} (length: {len(v)})")
    metric_lengths = [len(metrics[k]) for k in metrics if len(metrics[k]) > 0]
    if not metric_lengths:
        print(f"Warning: No metrics data available for {title_suffix}. Skipping plot.")
        return
    num_points = min(metric_lengths) if metric_lengths else 0
    folds = range(1, num_points + 1) if num_points > 0 else range(1, k_folds + 1)

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    if len(metrics["val_loss"]) > 0:
        plt.plot(folds[:len(metrics["val_loss"])], metrics["val_loss"], label="Validation Loss", marker="o")
    plt.xlabel("Fold"); plt.ylabel("Validation Loss")
    plt.title(f"Validation Loss Across Folds {title_suffix}")
    plt.legend()

    plt.subplot(2, 2, 2)
    if "val_acc" in metrics and len(metrics["val_acc"]) > 0:
        plt.plot(folds[:len(metrics["val_acc"])], metrics["val_acc"], label="Validation Accuracy", marker="o")
    plt.xlabel("Fold"); plt.ylabel("Validation Accuracy")
    plt.title(f"Validation Accuracy Across Folds {title_suffix}")
    plt.legend()

    plt.subplot(2, 2, 3)
    if len(metrics.get("precision", [])) > 0:
        plt.plot(folds[:len(metrics["precision"])], metrics["precision"], label="Precision", marker="o")
    plt.xlabel("Fold"); plt.ylabel("Precision")
    plt.title(f"Precision Across Folds {title_suffix}")
    plt.legend()

    plt.subplot(2, 2, 4)
    if len(metrics.get("f1", [])) > 0:
        plt.plot(folds[:len(metrics["f1"])], metrics["f1"], label="F1", marker="o")
    plt.xlabel("Fold"); plt.ylabel("F1")
    plt.title(f"F1 Across Folds {title_suffix}")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join("logs/main3-4", f"fold_metrics_{title_suffix.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Plot saved to {save_path}")

def compare_metrics(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
    print("Metrics lengths for comparison:")
    print(f"Full Fine-Tuning val_loss: {len(full_metrics.get('val_loss', []))}")
    print(f"Text Encoder Fine-Tuning val_loss: {len(text_metrics.get('val_loss', []))}")
    print(f"Vision-Only Fine-Tuning val_loss: {len(vision_metrics.get('val_loss', []))}")
    print(f"Partial Fine-Tuning val_loss: {len(partial_metrics.get('val_loss', []))}")

    max_len = max(
        len(full_metrics.get("val_loss", [])),
        len(text_metrics.get("val_loss", [])),
        len(vision_metrics.get("val_loss", [])),
        len(partial_metrics.get("val_loss", []))
    ) if any([
        len(full_metrics.get("val_loss", [])),
        len(text_metrics.get("val_loss", [])),
        len(vision_metrics.get("val_loss", [])),
        len(partial_metrics.get("val_loss", []))
    ]) else k_folds

    folds = range(1, max_len + 1)
    plt.figure(figsize=(15, 15))

    # Val Loss
    plt.subplot(3, 2, 1)
    for name, m, style in [
        ("Full", full_metrics, "o"),
        ("Text", text_metrics, "x"),
        ("Vision", vision_metrics, "s"),
        ("Partial", partial_metrics, "d")
    ]:
        if len(m.get("val_loss", [])) > 0:
            plt.plot(folds[:len(m["val_loss"])], m["val_loss"], label=f"{name} Val Loss", marker=style)
    plt.xlabel("Fold"); plt.ylabel("Val Loss"); plt.title("Validation Loss Comparison"); plt.legend()

    # Test Loss
    plt.subplot(3, 2, 2)
    for name, m, style in [
        ("Full", full_metrics, "o"),
        ("Text", text_metrics, "x"),
        ("Vision", vision_metrics, "s"),
        ("Partial", partial_metrics, "d")
    ]:
        if len(m.get("test_loss", [])) > 0:
            plt.plot(folds[:len(m["test_loss"])], m["test_loss"], label=f"{name} Test Loss", marker=style)
    plt.xlabel("Fold"); plt.ylabel("Test Loss"); plt.title("Test Loss Comparison"); plt.legend()

    # Val Acc
    plt.subplot(3, 2, 3)
    for name, m, style in [
        ("Full", full_metrics, "o"),
        ("Text", text_metrics, "x"),
        ("Vision", vision_metrics, "s"),
        ("Partial", partial_metrics, "d")
    ]:
        if len(m.get("val_acc", [])) > 0:
            plt.plot(folds[:len(m["val_acc"])], m["val_acc"], label=f"{name} Val Acc", marker=style)
    plt.xlabel("Fold"); plt.ylabel("Val Acc"); plt.title("Validation Accuracy Comparison"); plt.legend()

    # Test Acc
    plt.subplot(3, 2, 4)
    for name, m, style in [
        ("Full", full_metrics, "o"),
        ("Text", text_metrics, "x"),
        ("Vision", vision_metrics, "s"),
        ("Partial", partial_metrics, "d")
    ]:
        if len(m.get("test_acc", [])) > 0:
            plt.plot(folds[:len(m["test_acc"])], m["test_acc"], label=f"{name} Test Acc", marker=style)
    plt.xlabel("Fold"); plt.ylabel("Test Acc"); plt.title("Test Accuracy Comparison"); plt.legend()

    # Val F1
    plt.subplot(3, 2, 5)
    for name, m, style in [
        ("Full", full_metrics, "o"),
        ("Text", text_metrics, "x"),
        ("Vision", vision_metrics, "s"),
        ("Partial", partial_metrics, "d")
    ]:
        if len(m.get("f1", [])) > 0:
            plt.plot(folds[:len(m["f1"])], m["f1"], label=f"{name} F1 (Val)", marker=style)
    plt.xlabel("Fold"); plt.ylabel("F1 (Val)"); plt.title("F1 (Val) Comparison"); plt.legend()

    plt.tight_layout()
    save_path = os.path.join("logs/main3-4", "comparison_metrics_all.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Comparison plot saved to {save_path}")

def create_comparison_table(full_metrics, text_metrics, vision_metrics, partial_metrics, k_folds):
    def get(m, key, i):
        return m.get(key, [])[i] if i < len(m.get(key, [])) else 0

    rows = []
    for i in range(k_folds):
        rows.append({
            'Fold': i + 1,
            'Full Train Loss': get(full_metrics, 'train_loss', i),
            'Full Val Loss':   get(full_metrics, 'val_loss', i),
            'Full Test Loss':  get(full_metrics, 'test_loss', i),
            'Full Val Acc':    get(full_metrics, 'val_acc', i),
            'Full Test Acc':   get(full_metrics, 'test_acc', i),

            'Text Train Loss': get(text_metrics, 'train_loss', i),
            'Text Val Loss':   get(text_metrics, 'val_loss', i),
            'Text Test Loss':  get(text_metrics, 'test_loss', i),
            'Text Val Acc':    get(text_metrics, 'val_acc', i),
            'Text Test Acc':   get(text_metrics, 'test_acc', i),

            'Vision Train Loss': get(vision_metrics, 'train_loss', i),
            'Vision Val Loss':   get(vision_metrics, 'val_loss', i),
            'Vision Test Loss':  get(vision_metrics, 'test_loss', i),
            'Vision Val Acc':    get(vision_metrics, 'val_acc', i),
            'Vision Test Acc':   get(vision_metrics, 'test_acc', i),

            'Partial Train Loss': get(partial_metrics, 'train_loss', i),
            'Partial Val Loss':   get(partial_metrics, 'val_loss', i),
            'Partial Test Loss':  get(partial_metrics, 'test_loss', i),
            'Partial Val Acc':    get(partial_metrics, 'val_acc', i),
            'Partial Test Acc':   get(partial_metrics, 'test_acc', i),
        })

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    rows.append({
        'Fold': 'Average',
        'Full Train Loss': avg(full_metrics.get('train_loss', [])),
        'Full Val Loss':   avg(full_metrics.get('val_loss', [])),
        'Full Test Loss':  avg(full_metrics.get('test_loss', [])),
        'Full Val Acc':    avg(full_metrics.get('val_acc', [])),
        'Full Test Acc':   avg(full_metrics.get('test_acc', [])),

        'Text Train Loss': avg(text_metrics.get('train_loss', [])),
        'Text Val Loss':   avg(text_metrics.get('val_loss', [])),
        'Text Test Loss':  avg(text_metrics.get('test_loss', [])),
        'Text Val Acc':    avg(text_metrics.get('val_acc', [])),
        'Text Test Acc':   avg(text_metrics.get('test_acc', [])),

        'Vision Train Loss': avg(vision_metrics.get('train_loss', [])),
        'Vision Val Loss':   avg(vision_metrics.get('val_loss', [])),
        'Vision Test Loss':  avg(vision_metrics.get('test_loss', [])),
        'Vision Val Acc':    avg(vision_metrics.get('val_acc', [])),
        'Vision Test Acc':   avg(vision_metrics.get('test_acc', [])),

        'Partial Train Loss': avg(partial_metrics.get('train_loss', [])),
        'Partial Val Loss':   avg(partial_metrics.get('val_loss', [])),
        'Partial Test Loss':  avg(partial_metrics.get('test_loss', [])),
        'Partial Val Acc':    avg(partial_metrics.get('val_acc', [])),
        'Partial Test Acc':   avg(partial_metrics.get('test_acc', [])),
    })

    comparison_df = pd.DataFrame(rows)
    save_path = os.path.join("logs/main3-4", "comparison_table_all.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    comparison_df.to_csv(save_path, index=False)
    print(f"\nComparison Table saved to {save_path}")
    print(comparison_df)
    return comparison_df

def analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    expected = len(subcategories)
    if num_classes != expected:
        print(f"Warning: Confusion matrix size ({num_classes}x{num_classes}) != #classes ({expected}).")

    misclass = []
    for t in range(num_classes):
        total_true = np.sum(conf_matrix[t])
        if total_true == 0:
            continue
        for p in range(num_classes):
            if t != p and conf_matrix[t][p] > 0:
                count = conf_matrix[t][p]
                pct = (count / total_true) * 100
                true_label = subcategories[t] if t < len(subcategories) else f"Unknown_{t}"
                pred_label = subcategories[p] if p < len(subcategories) else f"Unknown_{p}"
                misclass.append({'True Class': true_label, 'Predicted Class': pred_label,
                                 'Count': count, 'Percentage of True Class': pct})
    if misclass:
        df = pd.DataFrame(misclass).sort_values(by='Count', ascending=False).head(10)
        print(f"\nSignificant Misclassifications for {model_type} (Fold {fold}, Epoch {epoch}, {split_type}):")
        print(df)
        save_path = os.path.join("logs/main3-4", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                                 f"misclassifications_epoch_{epoch}_{split_type.lower()}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Misclassifications saved to {save_path}")
        return df
    return None

def plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type="Validation"):
    num_classes = conf_matrix.shape[0]
    labels = subcategories[:num_classes] if num_classes <= len(subcategories) else [f"Class_{i}" for i in range(num_classes)]

    plt.figure(figsize=(15, 12))
    sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    save_path = os.path.join("logs/main3-4", model_type.lower().replace(" ", "_"), f"fold_{fold}",
                             f"confusion_matrix_epoch_{epoch}_{split_type.lower()}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path); plt.close()
    print(f"Confusion matrix plot saved to {save_path}")

def evaluate_model_with_metrics(model, dataloader, criterion, subcategories, device, fold, epoch, model_type,
                                split_type="Validation"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    # Prepare text prompts once per split if model supports tokenizer (Partial/Text encoders)
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    plot_confusion_matrix(conf_matrix, subcategories, model_type, fold, epoch, split_type)
    _ = analyze_misclassifications(conf_matrix, subcategories, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): "
          f"Loss={avg_loss:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, conf_matrix

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, subcategories, device,
                log_dir, fold, accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
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

        # If model supports tokenizer, prepare prompts once per epoch
        texts = None
        if hasattr(model, "tokenizer"):
            texts = model.tokenizer([f"a photo of {c}" for c in subcategories]).to(device)

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
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

        # Validate (and test) at interval or final epoch
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

# ------------------------------
# Main
# ------------------------------
if __name__ == '__main__':
    freeze_support()
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

    # Fold-level aggregation (store val/test acc explicitly for comparisons)
    def init_fold_metrics():
        return {
            "train_loss": [], "val_loss": [], "test_loss": [],
            "val_acc": [], "test_acc": [],
            "precision": [], "recall": [], "f1": []
        }

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

    checkpoint_file = os.path.join("logs/main3-4", "checkpoint.json")
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Loaded checkpoint: Resuming from Fold {checkpoint['current_fold'] + 1}")
            for key in ['full_fold_metrics','text_fold_metrics','vision_fold_metrics','partial_fold_metrics','timing_results']:
                if key in checkpoint:
                    locals()[key] = checkpoint[key]
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")

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
        val_indices   = train_val_indices[train_size:]

        train_subset = Subset(dataset, train_indices.tolist())
        val_subset   = Subset(dataset, val_indices.tolist())
        test_subset  = Subset(dataset, test_indices.tolist())

        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset   = FashionDataset(val_subset,   subcategories)
        test_dataset  = FashionDataset(test_subset,  subcategories)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=num_workers, pin_memory=True)
        val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        if str(fold) not in checkpoint['completed_approaches']:
            checkpoint['completed_approaches'][str(fold)] = []

        # 1) Full Fine-Tuning (ALL params train)
        if 'full' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Full Fine-Tuned Model...")
            start_time = time.time()

            full_model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer_full = optim.AdamW(full_model.parameters(), lr=1e-5, weight_decay=1e-4)  # lower LR for full FT

            full_log_dir = os.path.join("logs/main3-4", "full_finetune", f"fold_{fold + 1}")
            full_metrics_dict = train_model(
                full_model, train_loader, val_loader, test_loader, criterion, optimizer_full, num_epochs=4,
                subcategories=subcategories, device=device, log_dir=full_log_dir, fold=fold + 1,
                accumulation_steps=4, validate_every=2, model_type="Full Fine-Tuning"
            )

            # Evaluate once more (best is already saved during training)
            full_best_model_path = os.path.join(full_log_dir, f'best_full_fine-tuning_clip.pth')
            if os.path.exists(full_best_model_path):
                full_model.load_state_dict(torch.load(full_best_model_path, map_location=device))
                print(f"Loaded best model from {full_best_model_path}")

            # Final eval for fold-level numbers
            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                full_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                full_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            # Aggregate fold metrics (use validation accuracy for comparison; also store test_acc)
            full_fold_metrics["train_loss"].append(full_metrics_dict['train_loss'][-1] if full_metrics_dict['train_loss'] else 0)
            full_fold_metrics["val_loss"].append(val_loss)
            full_fold_metrics["test_loss"].append(test_loss)
            full_fold_metrics["val_acc"].append(val_acc)
            full_fold_metrics["test_acc"].append(test_acc)
            full_fold_metrics["precision"].append(val_prec)
            full_fold_metrics["recall"].append(val_rec)
            full_fold_metrics["f1"].append(val_f1)

            save_metrics_to_csv(full_metrics_dict, os.path.join("logs/main3-4", f"metrics_full_finetune_fold_{fold + 1}.csv"))
            save_logs_as_zip(full_log_dir, os.path.join("logs/main3-4", "full_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start_time
            timing_results["Full Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Full Fine-Tuning (Fold {fold + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed_approaches'][str(fold)].append('full')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics']    = full_fold_metrics
            checkpoint['text_fold_metrics']    = text_fold_metrics
            checkpoint['vision_fold_metrics']  = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results']       = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Full Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Full Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

        # 2) Text Encoder Fine-Tuning (reference / optional)
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
                text_model.load_state_dict(torch.load(text_best_model_path, map_location=device))
                print(f"Loaded best model from {text_best_model_path}")

            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                text_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                text_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            text_fold_metrics["train_loss"].append(text_metrics_dict['train_loss'][-1] if text_metrics_dict['train_loss'] else 0)
            text_fold_metrics["val_loss"].append(val_loss)
            text_fold_metrics["test_loss"].append(test_loss)
            text_fold_metrics["val_acc"].append(val_acc)
            text_fold_metrics["test_acc"].append(test_acc)
            text_fold_metrics["precision"].append(val_prec)
            text_fold_metrics["recall"].append(val_rec)
            text_fold_metrics["f1"].append(val_f1)

            save_metrics_to_csv(text_metrics_dict, os.path.join("logs/main3-4", f"metrics_text_encoder_fold_{fold + 1}.csv"))
            save_logs_as_zip(text_log_dir, os.path.join("logs/main3-4", "text_encoder_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start_time
            timing_results["Text Encoder Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Text Encoder Fine-Tuning (Fold {fold + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed_approaches'][str(fold)].append('text')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics']    = full_fold_metrics
            checkpoint['text_fold_metrics']    = text_fold_metrics
            checkpoint['vision_fold_metrics']  = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results']       = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Text Encoder Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Text Encoder Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

        # 3) Vision-Only Fine-Tuning (reference / optional)
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
                vision_model.load_state_dict(torch.load(vision_best_model_path, map_location=device))
                print(f"Loaded best model from {vision_best_model_path}")

            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                vision_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                vision_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            vision_fold_metrics["train_loss"].append(vision_metrics_dict['train_loss'][-1] if vision_metrics_dict['train_loss'] else 0)
            vision_fold_metrics["val_loss"].append(val_loss)
            vision_fold_metrics["test_loss"].append(test_loss)
            vision_fold_metrics["val_acc"].append(val_acc)
            vision_fold_metrics["test_acc"].append(test_acc)
            vision_fold_metrics["precision"].append(val_prec)
            vision_fold_metrics["recall"].append(val_rec)
            vision_fold_metrics["f1"].append(val_f1)

            save_metrics_to_csv(vision_metrics_dict, os.path.join("logs/main3-4", f"metrics_vision_only_fold_{fold + 1}.csv"))
            save_logs_as_zip(vision_log_dir, os.path.join("logs/main3-4", "vision_only_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start_time
            timing_results["Vision-Only Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Vision-Only Fine-Tuning (Fold {fold + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed_approaches'][str(fold)].append('vision')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics']    = full_fold_metrics
            checkpoint['text_fold_metrics']    = text_fold_metrics
            checkpoint['vision_fold_metrics']  = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results']       = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Vision-Only Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Vision-Only Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

        # 4) Partial Fine-Tuning (LAST 30% of BOTH encoders) — uses similarity path
        if 'partial' not in checkpoint['completed_approaches'][str(fold)]:
            print("\nTraining Partial Fine-Tuned Model (Last 30% of Layers in BOTH encoders)...")
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
                partial_model.load_state_dict(torch.load(partial_best_model_path, map_location=device))
                print(f"Loaded best model from {partial_best_model_path}")

            val_loss, val_acc, val_prec, val_rec, val_f1, _ = evaluate_model_with_metrics(
                partial_model, val_loader, criterion, subcategories, device, fold + 1, "Final", "Validation"
            )
            test_loss, test_acc, test_prec, test_rec, test_f1, _ = evaluate_model_with_metrics(
                partial_model, test_loader, criterion, subcategories, device, fold + 1, "Final", "Test"
            )

            partial_fold_metrics["train_loss"].append(partial_metrics_dict['train_loss'][-1] if partial_metrics_dict['train_loss'] else 0)
            partial_fold_metrics["val_loss"].append(val_loss)
            partial_fold_metrics["test_loss"].append(test_loss)
            partial_fold_metrics["val_acc"].append(val_acc)
            partial_fold_metrics["test_acc"].append(test_acc)
            partial_fold_metrics["precision"].append(val_prec)
            partial_fold_metrics["recall"].append(val_rec)
            partial_fold_metrics["f1"].append(val_f1)

            save_metrics_to_csv(partial_metrics_dict, os.path.join("logs/main3-4", f"metrics_partial_finetune_fold_{fold + 1}.csv"))
            save_logs_as_zip(partial_log_dir, os.path.join("logs/main3-4", "partial_finetune", f"fold_{fold + 1}_logs"))
            torch.cuda.empty_cache()

            duration = time.time() - start_time
            timing_results["Partial Fine-Tuning"].append(duration)
            h, rem = divmod(duration, 3600); m, s = divmod(rem, 60)
            print(f"Partial Fine-Tuning (Fold {fold + 1}) took {int(h):02d}:{int(m):02d}:{int(s):02d}")

            checkpoint['completed_approaches'][str(fold)].append('partial')
            checkpoint['current_fold'] = fold
            checkpoint['full_fold_metrics']    = full_fold_metrics
            checkpoint['text_fold_metrics']    = text_fold_metrics
            checkpoint['vision_fold_metrics']  = vision_fold_metrics
            checkpoint['partial_fold_metrics'] = partial_fold_metrics
            checkpoint['timing_results']       = timing_results
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
            print(f"Checkpoint saved after Partial Fine-Tuning for Fold {fold + 1}")
        else:
            print(f"\nSkipping Partial Fine-Tuning for Fold {fold + 1} - Already completed (from checkpoint)")

    # Debug: lengths before plotting
    print("\nDebug: Contents of metrics dictionaries before plotting:")
    print("Full:",    {k: len(v) for k, v in full_fold_metrics.items()})
    print("Text:",    {k: len(v) for k, v in text_fold_metrics.items()})
    print("Vision:",  {k: len(v) for k, v in vision_fold_metrics.items()})
    print("Partial:", {k: len(v) for k, v in partial_fold_metrics.items()})

    # Plot & save
    plot_fold_metrics(full_fold_metrics,    k_folds, title_suffix="Full Fine-Tuning")
    plot_fold_metrics(text_fold_metrics,    k_folds, title_suffix="Text Encoder Fine-Tuning")
    plot_fold_metrics(vision_fold_metrics,  k_folds, title_suffix="Vision-Only Fine-Tuning")
    plot_fold_metrics(partial_fold_metrics, k_folds, title_suffix="Partial Fine-Tuning")
    compare_metrics(full_fold_metrics, text_fold_metrics, vision_fold_metrics, partial_fold_metrics, k_folds)

    # Persist fold-level CSVs
    save_metrics_to_csv(full_fold_metrics,    os.path.join("logs/main3-4", "kfold_full_finetune_metrics.csv"))
    save_metrics_to_csv(text_fold_metrics,    os.path.join("logs/main3-4", "kfold_text_encoder_metrics.csv"))
    save_metrics_to_csv(vision_fold_metrics,  os.path.join("logs/main3-4", "kfold_vision_only_metrics.csv"))
    save_metrics_to_csv(partial_fold_metrics, os.path.join("logs/main3-4", "kfold_partial_finetune_metrics.csv"))

    # Comparison table (includes Acc)
    create_comparison_table(full_fold_metrics, text_fold_metrics, vision_fold_metrics, partial_fold_metrics, k_folds)

    # Print summaries
    def safe_avg(d, key):
        return float(np.mean(d.get(key, []))) if d.get(key, []) else 0.0

    print("\nSummary of Average Metrics Across Folds:")
    for name, d in [("Full Fine-Tuning", full_fold_metrics),
                    ("Text Encoder Fine-Tuning", text_fold_metrics),
                    ("Vision-Only Fine-Tuning", vision_fold_metrics),
                    ("Partial Fine-Tuning (Last 30%)", partial_fold_metrics)]:
        print(f"\n{name}:")
        print(f"Avg Train Loss:   {safe_avg(d, 'train_loss'):.4f}")
        print(f"Avg Val Loss:     {safe_avg(d, 'val_loss'):.4f}")
        print(f"Avg Test Loss:    {safe_avg(d, 'test_loss'):.4f}")
        print(f"Avg Val Acc:      {safe_avg(d, 'val_acc'):.4f}")
        print(f"Avg Test Acc:     {safe_avg(d, 'test_acc'):.4f}")
        print(f"Avg Precision:    {safe_avg(d, 'precision'):.4f}")
        print(f"Avg Recall:       {safe_avg(d, 'recall'):.4f}")
        print(f"Avg F1 (Val):     {safe_avg(d, 'f1'):.4f}")

    # Timing
    print("\nTiming Comparison Across Approaches (Total Duration per Fold):")
    for approach, durations in timing_results.items():
        if durations:
            total = sum(durations)
            avg = total / len(durations)
            h, rem = divmod(total, 3600); m, s = divmod(rem, 60)
            ah, arem = divmod(avg, 3600); am, asec = divmod(arem, 60)
            print(f"{approach}:")
            print(f"  Total Time:  {int(h):02d}:{int(m):02d}:{int(s):02d}")
            print(f"  Avg / Fold:  {int(ah):02d}:{int(am):02d}:{int(asec):02d}")
            for i, d in enumerate(durations, 1):
                h, rem = divmod(d, 3600); m, s = divmod(rem, 60)
                print(f"    Fold {i}: {int(h):02d}:{int(m):02d}:{int(s):02d}")
        else:
            print(f"{approach}: No timing data available.")

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    h, rem = divmod(overall_duration, 3600); m, s = divmod(rem, 60)
    print(f"\nOverall Training Finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end_time))}")
    print(f"Total Duration: {int(h):02d}:{int(m):02d}:{int(s):02d}")

    # Save timing data
    timing_records = []
    for approach, durations in timing_results.items():
        if durations:
            for fold_idx, d in enumerate(durations, 1):
                h, rem = divmod(d, 3600); m, s = divmod(rem, 60)
                formatted = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
                timing_records.append({"Approach": approach, "Fold": fold_idx, "Duration (seconds)": d, "Formatted Duration": formatted})
        else:
            for fold_idx in range(1, k_folds + 1):
                timing_records.append({"Approach": approach, "Fold": fold_idx, "Duration (seconds)": 0, "Formatted Duration": "N/A"})
    timing_df = pd.DataFrame(timing_records)
    timing_save_path = os.path.join("logs/main3-4", "execution_timingpart4.csv")
    timing_df.to_csv(timing_save_path, index=False)
    print(f"Timing data saved to {timing_save_path}")
