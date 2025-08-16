# -*- coding: utf-8 -*-
import os, json, time, shutil, warnings
from zipfile import ZipFile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# ---- AMP compatibility (PyTorch 1.x & 2.x) ----
try:
    from torch.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PyTorch 2.x
    AMP_IS_V2 = True
except Exception:
    from torch.cuda.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PyTorch 1.x
    AMP_IS_V2 = False

def autocast_ctx(device: torch.device):
    """Return a correct autocast context that works for both PT1.x and PT2.x."""
    if AMP_IS_V2:
        # torch.amp.autocast requires device_type in PT2.x
        return autocast_amp(device_type=device.type, enabled=(device.type != "cpu"))
    else:
        # torch.cuda.amp.autocast uses 'enabled' only
        return autocast_amp(enabled=torch.cuda.is_available())

def make_scaler(device: torch.device):
    return GradScalerAmp(enabled=(device.type == "cuda"))

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from datasets import load_dataset
from tqdm import tqdm
import open_clip
from torch.multiprocessing import freeze_support

# ------------------------------
# Output dirs
# ------------------------------
BASE_DIR = os.path.join("logs", "main3-4aa")
PNG_DIR  = os.path.join(BASE_DIR, "png")
CSV_DIR  = os.path.join(BASE_DIR, "csv")
BEST_DIR = os.path.join(BASE_DIR, "best")
for d in [PNG_DIR, CSV_DIR, BEST_DIR]:
    os.makedirs(d, exist_ok=True)

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

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        label = self.subcategories.index(item['subCategory'])
        image = self.transform(image)
        return image, label

# ------------------------------
# Models (baselines)
# ------------------------------
class FullFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1, freeze_encoder=False):
        super().__init__()
        self.visual_encoder = base_model.visual
        if freeze_encoder:
            for p in self.visual_encoder.parameters(): p.requires_grad = False
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
        return self.classifier(feats)

class TextEncoderFineTunedCLIP(nn.Module):
    def __init__(self, base_model, subcategories, device, freeze_visual=True):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = device
        self.subcategories  = subcategories
        if freeze_visual:
            for p in self.visual_encoder.parameters(): p.requires_grad = False
        self.text_projection      = base_model.text_projection
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.token_embedding      = base_model.token_embedding
    def encode_text(self, token_ids):
        x = self.token_embedding(token_ids); x = x + self.positional_embedding
        x = x.permute(1, 0, 2); x = self.text_encoder(x); x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.text_projection
        return x
    def forward(self, images, text_inputs=None):
        image_features = self.visual_encoder(images)
        if text_inputs is None:
            text_inputs = self.tokenizer([f"a photo of {c}" for c in self.subcategories]).to(self.device)
        text_features = self.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T) * 100.0

class VisionOnlyFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters(): p.requires_grad = True
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
        return self.classifier(feats)

class PartialFineTunedCLIP(nn.Module):
    """Unfreeze last ~30% of param tensors in BOTH encoders; use CLIP similarity when prompts are provided."""
    def __init__(self, base_model, num_classes, freeze_percentage=0.7):
        super().__init__()
        self.visual_encoder = base_model.visual
        self.text_encoder   = base_model.transformer
        self.tokenizer      = open_clip.get_tokenizer("ViT-B-32")
        self.device         = next(base_model.parameters()).device

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
        total = len(named_params); cutoff = int(total * freeze_percentage)
        print(f"Freezing first {freeze_percentage*100:.1f}% of {enc_name} ({cutoff}/{total} tensors)")
        for i, (_, p) in enumerate(named_params):
            p.requires_grad = i >= cutoff

    def encode_text(self, token_ids):
        x = self.token_embedding(token_ids); x = x + self.positional_embedding
        x = x.permute(1, 0, 2); x = self.text_encoder(x); x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images, text_inputs=None):
        image_features = self.visual_encoder(images)
        if text_inputs is not None:
            text_features = self.encode_text(text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
            return (image_features @ text_features.T) * 100.0
        return self.classifier(image_features)

# ------------------------------
# Custom LoRA (no PEFT needed)
# ------------------------------
class LoRALinear(nn.Module):
    """
    Drop-in LoRA wrapper for nn.Linear:
      y = x W^T + scale * (x (B^T A^T))   where A: in->r, B: r->out
    Base Linear weight is frozen; only LoRA A/B are trainable.
    """
    def __init__(self, linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.scale = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # Frozen original layer
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = None
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

        # LoRA A and B
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        # Init as in LoRA: B zeros, A small random
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = torch.nn.functional.linear(x, self.weight, self.bias)
        x_d = self.dropout(x)
        lora = self.lora_B(self.lora_A(x_d)) * self.scale
        return base + lora

def inject_lora_modules(root: nn.Module, target_substrings, r=8, alpha=16, dropout=0.05):
    """
    Recursively replace nn.Linear modules whose names contain any of the substrings
    with LoRALinear wrappers. Operates in-place.
    """
    for name, module in list(root.named_children()):
        full = module
        if isinstance(full, nn.Linear) and any(s in name.lower() for s in target_substrings):
            setattr(root, name, LoRALinear(full, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora_modules(full, target_substrings, r=r, alpha=alpha, dropout=dropout)

def discover_lora_targets(model, prefer=("qkv","q_proj","k_proj","v_proj","out_proj","proj","fc1","fc2","in_proj","mlp","attn")):
    """
    Heuristic: find Linear submodule names that include common transformer/MLP names.
    Falls back to "" (match-all) if none are found.
    """
    linear_names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            linear_names.append(name)

    lowers = [n.lower() for n in linear_names]
    chosen = set()
    for cand in prefer:
        c = cand.lower()
        if any(c in n for n in lowers):
            chosen.add(c)
    return sorted(chosen)

class LoRAClipBothEncoders(nn.Module):
    """
    Apply custom LoRA to BOTH encoders (vision + text) for discovered Linear layers.
    """
    def __init__(self, base_model, subcategories, device, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.subcategories = subcategories
        self.device = device
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        # Expose CLIP components
        self.visual = base_model.visual
        self.text_encoder = base_model.transformer
        self.token_embedding      = base_model.token_embedding
        self.positional_embedding = base_model.positional_embedding
        self.ln_final             = base_model.ln_final
        self.text_projection      = base_model.text_projection

        # Pick targets (fallback to match-all if none discovered)
        vis_targets = discover_lora_targets(self.visual)
        txt_targets = discover_lora_targets(self.text_encoder)
        if not vis_targets: vis_targets = [""]
        if not txt_targets: txt_targets = [""]

        print(f"[LoRA] Visual targets  substrings: {vis_targets}")
        print(f"[LoRA] Text targets    substrings: {txt_targets}")

        # Inject LoRA in-place
        inject_lora_modules(self.visual, vis_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        inject_lora_modules(self.text_encoder, txt_targets, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        # Freeze original (non-LoRA) weights
        for n, p in self.visual.named_parameters():
            if "lora_" not in n: p.requires_grad = False
        for n, p in self.text_encoder.named_parameters():
            if "lora_" not in n: p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, token_ids):
        x = self.token_embedding(token_ids); x = x + self.positional_embedding
        x = x.permute(1, 0, 2); x = self.text_encoder(x); x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, images, text_inputs=None):
        image_features = self.visual(images)
        if text_inputs is None:
            text_inputs = self.tokenizer([f"a photo of {c}" for c in self.subcategories]).to(self.device)
        text_features = self.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        return (image_features @ text_features.T) * 100.0

# ------------------------------
# Utilities
# ------------------------------
def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_weights)
    model = model.to(device); model.eval()
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

def save_csv(df_or_dict, filename):
    out = os.path.join(CSV_DIR, filename); os.makedirs(os.path.dirname(out), exist_ok=True)
    if isinstance(df_or_dict, pd.DataFrame):
        df_or_dict.to_csv(out, index=False)
    else:
        pd.DataFrame(df_or_dict).to_csv(out, index=False)
    print(f"CSV saved: {out}"); return out

def save_png_current_figure(filename):
    out = os.path.join(PNG_DIR, filename); os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"PNG saved: {out}"); return out

def save_logs_as_zip(log_dir, zip_stem):
    if os.path.exists(log_dir):
        zip_path = os.path.join(BASE_DIR, f"{zip_stem}.zip")
        with ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(log_dir):
                for file in files:
                    src = os.path.join(root, file)
                    arc = os.path.relpath(src, log_dir)
                    zipf.write(src, arc)
        print(f"Logs zipped: {zip_path}")
        shutil.rmtree(log_dir, ignore_errors=True)

# ------------------------------
# Confusion / Misclass
# ------------------------------
def plot_confusion_matrix(cm, subcats, model_type, fold, epoch, split_type="Validation"):
    n = cm.shape[0]
    labels = subcats[:n] if n <= len(subcats) else [f"Class_{i}" for i in range(n)]
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type} (Fold {fold}, Epoch {epoch}, {split_type})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    fname = f"{model_type.lower().replace(' ', '_')}_fold{fold}_epoch{epoch}_{split_type.lower()}_confmat.png"
    save_png_current_figure(fname)

def analyze_misclassifications(cm, subcats, model_type, fold, epoch, split_type="Validation"):
    n = cm.shape[0]; rows = []
    for t in range(n):
        tot = np.sum(cm[t])
        if tot == 0: continue
        for p in range(n):
            if t != p and cm[t][p] > 0:
                cnt = cm[t][p]
                rows.append({
                    "True Class": subcats[t] if t < len(subcats) else f"Class_{t}",
                    "Predicted Class": subcats[p] if p < len(subcats) else f"Class_{p}",
                    "Count": cnt, "Percentage of True Class": 100.0 * cnt / tot
                })
    if rows:
        df = pd.DataFrame(rows).sort_values("Count", ascending=False).head(10)
        fname = f"{model_type.lower().replace(' ', '_')}_fold{fold}_epoch{epoch}_{split_type.lower()}_misclass.csv"
        save_csv(df, fname)
        return df
    return None

# ------------------------------
# Eval & Train
# ------------------------------
def evaluate_model_with_metrics(model, loader, criterion, subcats, device, fold, epoch, model_type, split_type="Validation"):
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []
    texts = None
    if hasattr(model, "tokenizer"):
        texts = model.tokenizer([f"a photo of {c}" for c in subcats]).to(device)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, text_inputs=texts) if texts is not None else model(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = tot_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                               average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, subcats, model_type, fold, epoch, split_type)
    analyze_misclassifications(cm, subcats, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): "
          f"Loss={avg_loss:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, cm

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs, subcats, device, log_dir, fold,
                accumulation_steps=4, validate_every=2, model_type="Model"):
    os.makedirs(log_dir, exist_ok=True)
    scaler = make_scaler(device)
    best_f1, patience, patience_counter = 0.0, 3, 0

    metrics = {"epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "test_loss": [], "test_acc": [],
               "precision_val": [], "recall_val": [], "f1_val": [],
               "precision_test": [], "recall_test": [], "f1_test": []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss, correct_train, total_train = 0.0, 0, 0
        optimizer.zero_grad()

        texts = None
        if hasattr(model, "tokenizer"):
            texts = model.tokenizer([f"a photo of {c}" for c in subcats]).to(device)

        with tqdm(train_loader, desc=f"{model_type} | Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for bidx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                with autocast_ctx(device):
                    outputs = model(images, text_inputs=texts) if texts is not None else model(images)
                    loss = criterion(outputs, labels) / accumulation_steps
                scaler.scale(loss).backward()
                if (bidx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(outputs, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train   += labels.size(0)

                pbar.set_postfix({"loss": f"{(total_train_loss / (bidx+1)):.4f}",
                                  "acc":  f"{(correct_train / max(total_train,1)):.4f}"})

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        if (epoch + 1) % validate_every == 0 or epoch == num_epochs - 1:
            v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate_model_with_metrics(
                model, val_loader,  criterion, subcats, device, fold, epoch + 1, model_type, "Validation")
            t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate_model_with_metrics(
                model, test_loader, criterion, subcats, device, fold, epoch + 1, model_type, "Test")

            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss); metrics["train_acc"].append(train_acc)
            metrics["val_loss"].append(v_loss); metrics["val_acc"].append(v_acc)
            metrics["test_loss"].append(t_loss); metrics["test_acc"].append(t_acc)
            metrics["precision_val"].append(v_p); metrics["recall_val"].append(v_r); metrics["f1_val"].append(v_f1)
            metrics["precision_test"].append(t_p); metrics["recall_test"].append(t_r); metrics["f1_test"].append(t_f1)

            if v_f1 > best_f1:
                best_f1 = v_f1; patience_counter = 0
                best_path = os.path.join(BEST_DIR, f"best_{model_type.lower().replace(' ', '_')}_fold{fold}.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Best model saved: {best_path} (Val F1={best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            metrics["epoch"].append(epoch + 1)
            metrics["train_loss"].append(avg_train_loss); metrics["train_acc"].append(train_acc)
            metrics["val_loss"].append(0.0); metrics["val_acc"].append(0.0)
            metrics["test_loss"].append(0.0); metrics["test_acc"].append(0.0)
            metrics["precision_val"].append(0.0); metrics["recall_val"].append(0.0); metrics["f1_val"].append(0.0)
            metrics["precision_test"].append(0.0); metrics["recall_test"].append(0.0); metrics["f1_test"].append(0.0)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    save_csv(metrics, f"{model_type.lower().replace(' ', '_')}_fold{fold}_per_epoch.csv")
    return metrics

# ------------------------------
# Fold plots
# ------------------------------
def save_png(fig_name): save_png_current_figure(fig_name)

def plot_fold_metrics(metrics, title_suffix=""):
    def g(k): return metrics.get(k, [])
    L = [len(g(k)) for k in metrics if len(g(k)) > 0]
    if not L: return
    n = min(L); folds = range(1, n+1)
    plt.figure(figsize=(14,10))
    plt.subplot(2,2,1);
    if len(g("val_loss"))>0: plt.plot(folds[:len(g("val_loss"))], g("val_loss"), marker="o", label="Val Loss")
    plt.xlabel("Fold"); plt.ylabel("Loss"); plt.title(f"Validation Loss — {title_suffix}"); plt.legend()
    plt.subplot(2,2,2);
    if len(g("test_loss"))>0: plt.plot(folds[:len(g("test_loss"))], g("test_loss"), marker="o", label="Test Loss")
    plt.xlabel("Fold"); plt.ylabel("Loss"); plt.title(f"Test Loss — {title_suffix}"); plt.legend()
    plt.subplot(2,2,3);
    if len(g("val_acc"))>0: plt.plot(folds[:len(g("val_acc"))], g("val_acc"), marker="o", label="Val Acc")
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.title(f"Validation Accuracy — {title_suffix}"); plt.legend()
    plt.subplot(2,2,4);
    if len(g("test_acc"))>0: plt.plot(folds[:len(g("test_acc"))], g("test_acc"), marker="o", label="Test Acc")
    plt.xlabel("Fold"); plt.ylabel("Accuracy"); plt.title(f"Test Accuracy — {title_suffix}"); plt.legend()
    save_png(f"fold_metrics_{title_suffix.replace(' ', '_')}.png")

def compare_metrics(full_m, text_m, vision_m, partial_m, lora_m, k_folds):
    def s(d,k): return d.get(k, [])
    max_len = max(len(s(full_m,"val_loss")), len(s(text_m,"val_loss")),
                  len(s(vision_m,"val_loss")), len(s(partial_m,"val_loss")),
                  len(s(lora_m,"val_loss")), k_folds)
    folds = range(1, max_len+1)
    plt.figure(figsize=(16,16))
    panels = [("Validation Loss","val_loss"), ("Test Loss","test_loss"),
              ("Validation Acc","val_acc"), ("Test Acc","test_acc"), ("Validation F1","f1")]
    series = [("Full", full_m,"o"), ("Text", text_m,"x"),
              ("Vision", vision_m,"s"), ("Partial", partial_m,"d"), ("LoRA", lora_m,"^")]
    for i,(title,key) in enumerate(panels,1):
        plt.subplot(3,2,i)
        for name,m,marker in series:
            y = s(m,key)
            if len(y)>0: plt.plot(folds[:len(y)], y, marker=marker, label=name)
        plt.xlabel("Fold"); plt.ylabel(title); plt.title(title); plt.legend()
    save_png("comparison_metrics_all.png")

def plot_full_vs_partial_vs_lora(full_m, partial_m, lora_m, k_folds):
    def s(d,k): return d.get(k, [])[:k_folds]
    folds = range(1, k_folds+1)
    pairs = [("Val Accuracy","val_acc"), ("Test Accuracy","test_acc"),
             ("Val Loss","val_loss"), ("Test Loss","test_loss")]
    plt.figure(figsize=(12,10))
    for i,(title,key) in enumerate(pairs,1):
        plt.subplot(2,2,i)
        plt.plot(folds[:len(s(full_m,key))],    s(full_m,key),    marker="o", label="Full FT")
        plt.plot(folds[:len(s(partial_m,key))], s(partial_m,key), marker="d", label="Partial FT (last 30%)")
        plt.plot(folds[:len(s(lora_m,key))],    s(lora_m,key),    marker="^", label="LoRA (both encoders)")
        plt.xlabel("Fold"); plt.ylabel(title); plt.title(f"Full vs Partial vs LoRA — {title}"); plt.legend()
    save_png("full_vs_partial_vs_lora.png")

# ------------------------------
# Fold table
# ------------------------------
def create_fold_comparison_table(full_m, text_m, vision_m, partial_m, lora_m, k_folds):
    def get(m,k,i):
        v=m.get(k, []); return v[i] if i < len(v) else 0
    rows=[]
    for i in range(k_folds):
        rows.append({
            "Fold": i+1,
            "Full Train Loss": get(full_m,"train_loss",i), "Full Train Acc": get(full_m,"train_acc",i),
            "Full Val Loss": get(full_m,"val_loss",i), "Full Val Acc": get(full_m,"val_acc",i),
            "Full Test Loss": get(full_m,"test_loss",i), "Full Test Acc": get(full_m,"test_acc",i),
            "Text Train Loss": get(text_m,"train_loss",i), "Text Train Acc": get(text_m,"train_acc",i),
            "Text Val Loss": get(text_m,"val_loss",i), "Text Val Acc": get(text_m,"val_acc",i),
            "Text Test Loss": get(text_m,"test_loss",i), "Text Test Acc": get(text_m,"test_acc",i),
            "Vision Train Loss": get(vision_m,"train_loss",i), "Vision Train Acc": get(vision_m,"train_acc",i),
            "Vision Val Loss": get(vision_m,"val_loss",i), "Vision Val Acc": get(vision_m,"val_acc",i),
            "Vision Test Loss": get(vision_m,"test_loss",i), "Vision Test Acc": get(vision_m,"test_acc",i),
            "Partial Train Loss": get(partial_m,"train_loss",i), "Partial Train Acc": get(partial_m,"train_acc",i),
            "Partial Val Loss": get(partial_m,"val_loss",i), "Partial Val Acc": get(partial_m,"val_acc",i),
            "Partial Test Loss": get(partial_m,"test_loss",i), "Partial Test Acc": get(partial_m,"test_acc",i),
            "LoRA Train Loss": get(lora_m,"train_loss",i), "LoRA Train Acc": get(lora_m,"train_acc",i),
            "LoRA Val Loss": get(lora_m,"val_loss",i), "LoRA Val Acc": get(lora_m,"val_acc",i),
            "LoRA Test Loss": get(lora_m,"test_loss",i), "LoRA Test Acc": get(lora_m,"test_acc",i),
        })
    def avg(m,k):
        v=m.get(k, []); return float(np.mean(v)) if v else 0.0
    rows.append({
        "Fold":"Average",
        "Full Train Loss":avg(full_m,"train_loss"), "Full Train Acc":avg(full_m,"train_acc"),
        "Full Val Loss":avg(full_m,"val_loss"), "Full Val Acc":avg(full_m,"val_acc"),
        "Full Test Loss":avg(full_m,"test_loss"), "Full Test Acc":avg(full_m,"test_acc"),
        "Text Train Loss":avg(text_m,"train_loss"), "Text Train Acc":avg(text_m,"train_acc"),
        "Text Val Loss":avg(text_m,"val_loss"), "Text Val Acc":avg(text_m,"val_acc"),
        "Text Test Loss":avg(text_m,"test_loss"), "Text Test Acc":avg(text_m,"test_acc"),
        "Vision Train Loss":avg(vision_m,"train_loss"), "Vision Train Acc":avg(vision_m,"train_acc"),
        "Vision Val Loss":avg(vision_m,"val_loss"), "Vision Val Acc":avg(vision_m,"val_acc"),
        "Vision Test Loss":avg(vision_m,"test_loss"), "Vision Test Acc":avg(vision_m,"test_acc"),
        "Partial Train Loss":avg(partial_m,"train_loss"), "Partial Train Acc":avg(partial_m,"train_acc"),
        "Partial Val Loss":avg(partial_m,"val_loss"), "Partial Val Acc":avg(partial_m,"val_acc"),
        "Partial Test Loss":avg(partial_m,"test_loss"), "Partial Test Acc":avg(partial_m,"test_acc"),
        "LoRA Train Loss":avg(lora_m,"train_loss"), "LoRA Train Acc":avg(lora_m,"train_acc"),
        "LoRA Val Loss":avg(lora_m,"val_loss"), "LoRA Val Acc":avg(lora_m,"val_acc"),
        "LoRA Test Loss":avg(lora_m,"test_loss"), "LoRA Test Acc":avg(lora_m,"test_acc"),
    })
    df = pd.DataFrame(rows)
    save_csv(df, "fold_comparison_table.csv")
    return df

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    freeze_support()
    overall_start = time.time()
    print(f"\nStart: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_start))}")

    device = setup_environment()
    clip_model, _, _ = load_clip_model(device, "ViT-B-32", "openai")
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    def init_fold_metrics():
        return {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[],
                "test_loss":[], "test_acc":[], "precision":[], "recall":[], "f1":[]}

    full_fold   = init_fold_metrics()
    text_fold   = init_fold_metrics()
    vision_fold = init_fold_metrics()
    partial_fold= init_fold_metrics()
    lora_fold   = init_fold_metrics()

    timing = {"Full Fine-Tuning":[], "Text Encoder Fine-Tuning":[],
              "Vision-Only Fine-Tuning":[], "Partial Fine-Tuning":[], "LoRA Adapters":[]}

    ckpt_path = os.path.join(BASE_DIR, "checkpoint.json")
    checkpoint = {"current_fold": 0, "completed_approaches": {}}
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path,"r") as f: checkpoint = json.load(f)
            print(f"Resume from fold {checkpoint['current_fold']+1}")
            for k in ["full_fold","text_fold","vision_fold","partial_fold","lora_fold","timing"]:
                if k in checkpoint:
                    locals()[k] = checkpoint[k]
        except Exception as e:
            print("Checkpoint load failed:", e)

    num_workers = min(8, os.cpu_count()); print("Workers:", num_workers)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        if fold < checkpoint["current_fold"]:
            print(f"Skip Fold {fold+1}"); continue

        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        tv_size = len(train_val_idx); train_size = int(0.8 * tv_size)
        train_idx = train_val_idx[:train_size]; val_idx = train_val_idx[train_size:]

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

        # 1) Full FT
        if "full" not in checkpoint["completed_approaches"][str(fold)]:
            print("\n[1/5] Full Fine-Tuning")
            start = time.time()
            model = FullFineTunedCLIP(clip_model, len(subcategories), num_layers=1, freeze_encoder=False).to(device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
            opt  = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
            log_dir = os.path.join(BASE_DIR, "raw_full_finetune_fold", f"{fold+1}")
            met = train_model(model, train_loader, val_loader, test_loader, crit, opt, 4,
                              subcategories, device, log_dir, fold+1, 4, 2, "Full Fine-Tuning")
            best = os.path.join(BEST_DIR, f"best_full_fine-tuning_fold{fold+1}.pth")
            if os.path.exists(best): model.load_state_dict(torch.load(best, map_location=device))
            v = evaluate_model_with_metrics(model, val_loader, crit, subcategories, device, fold+1, 999,
                                            "Full Fine-Tuning","Validation")
            t = evaluate_model_with_metrics(model, test_loader, crit, subcategories, device, fold+1, 999,
                                            "Full Fine-Tuning","Test")
            full_fold["train_loss"].append(met["train_loss"][-1] if met["train_loss"] else 0)
            full_fold["train_acc"].append(met["train_acc"][-1] if met["train_acc"] else 0)
            full_fold["val_loss"].append(v[0]); full_fold["val_acc"].append(v[1])
            full_fold["test_loss"].append(t[0]); full_fold["test_acc"].append(t[1])
            full_fold["precision"].append(v[2]); full_fold["recall"].append(v[3]); full_fold["f1"].append(v[4])
            save_logs_as_zip(log_dir, f"full_finetune_fold_{fold+1}_logs")
            timing["Full Fine-Tuning"].append(time.time()-start)
            checkpoint["completed_approaches"][str(fold)].append("full")
            checkpoint["current_fold"]=fold; checkpoint["full_fold"]=full_fold; checkpoint["timing"]=timing
            with open(ckpt_path,"w") as f: json.dump(checkpoint,f)

        # 2) Text FT
        if "text" not in checkpoint["completed_approaches"][str(fold)]:
            print("\n[2/5] Text Encoder Fine-Tuning")
            start=time.time()
            model = TextEncoderFineTunedCLIP(clip_model, subcategories, device, freeze_visual=True).to(device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
            opt  = optim.AdamW(model.text_encoder.parameters(), lr=1e-4)
            log_dir = os.path.join(BASE_DIR, "raw_text_finetune_fold", f"{fold+1}")
            met = train_model(model, train_loader, val_loader, test_loader, crit, opt, 4,
                              subcategories, device, log_dir, fold+1, 4, 2, "Text Encoder Fine-Tuning")
            best = os.path.join(BEST_DIR, f"best_text_encoder_fine-tuning_fold{fold+1}.pth")
            if os.path.exists(best): model.load_state_dict(torch.load(best, map_location=device))
            v = evaluate_model_with_metrics(model, val_loader, crit, subcategories, device, fold+1, 999,
                                            "Text Encoder Fine-Tuning","Validation")
            t = evaluate_model_with_metrics(model, test_loader, crit, subcategories, device, fold+1, 999,
                                            "Text Encoder Fine-Tuning","Test")
            text_fold["train_loss"].append(met["train_loss"][-1] if met["train_loss"] else 0)
            text_fold["train_acc"].append(met["train_acc"][-1] if met["train_acc"] else 0)
            text_fold["val_loss"].append(v[0]); text_fold["val_acc"].append(v[1])
            text_fold["test_loss"].append(t[0]); text_fold["test_acc"].append(t[1])
            text_fold["precision"].append(v[2]); text_fold["recall"].append(v[3]); text_fold["f1"].append(v[4])
            save_logs_as_zip(log_dir, f"text_finetune_fold_{fold+1}_logs")
            timing["Text Encoder Fine-Tuning"].append(time.time()-start)
            checkpoint["completed_approaches"][str(fold)].append("text")
            checkpoint["current_fold"]=fold; checkpoint["text_fold"]=text_fold; checkpoint["timing"]=timing
            with open(ckpt_path,"w") as f: json.dump(checkpoint,f)

        # 3) Vision-only FT
        if "vision" not in checkpoint["completed_approaches"][str(fold)]:
            print("\n[3/5] Vision-Only Fine-Tuning")
            start=time.time()
            model = VisionOnlyFineTunedCLIP(clip_model, len(subcategories), num_layers=1).to(device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
            opt  = optim.AdamW(model.parameters(), lr=1e-4)
            log_dir = os.path.join(BASE_DIR, "raw_vision_finetune_fold", f"{fold+1}")
            met = train_model(model, train_loader, val_loader, test_loader, crit, opt, 4,
                              subcategories, device, log_dir, fold+1, 4, 2, "Vision-Only Fine-Tuning")
            best = os.path.join(BEST_DIR, f"best_vision-only_fine-tuning_fold{fold+1}.pth")
            if os.path.exists(best): model.load_state_dict(torch.load(best, map_location=device))
            v = evaluate_model_with_metrics(model, val_loader, crit, subcategories, device, fold+1, 999,
                                            "Vision-Only Fine-Tuning","Validation")
            t = evaluate_model_with_metrics(model, test_loader, crit, subcategories, device, fold+1, 999,
                                            "Vision-Only Fine-Tuning","Test")
            vision_fold["train_loss"].append(met["train_loss"][-1] if met["train_loss"] else 0)
            vision_fold["train_acc"].append(met["train_acc"][-1] if met["train_acc"] else 0)
            vision_fold["val_loss"].append(v[0]); vision_fold["val_acc"].append(v[1])
            vision_fold["test_loss"].append(t[0]); vision_fold["test_acc"].append(t[1])
            vision_fold["precision"].append(v[2]); vision_fold["recall"].append(v[3]); vision_fold["f1"].append(v[4])
            save_logs_as_zip(log_dir, f"vision_finetune_fold_{fold+1}_logs")
            timing["Vision-Only Fine-Tuning"].append(time.time()-start)
            checkpoint["completed_approaches"][str(fold)].append("vision")
            checkpoint["current_fold"]=fold; checkpoint["vision_fold"]=vision_fold; checkpoint["timing"]=timing
            with open(ckpt_path,"w") as f: json.dump(checkpoint,f)

        # 4) Partial FT
        if "partial" not in checkpoint["completed_approaches"][str(fold)]:
            print("\n[4/5] Partial Fine-Tuning (last 30% both encoders)")
            start=time.time()
            model = PartialFineTunedCLIP(clip_model, len(subcategories), freeze_percentage=0.7).to(device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
            params = [p for p in model.parameters() if p.requires_grad]
            opt  = optim.AdamW(params, lr=1e-4)
            log_dir = os.path.join(BASE_DIR, "raw_partial_finetune_fold", f"{fold+1}")
            met = train_model(model, train_loader, val_loader, test_loader, crit, opt, 4,
                              subcategories, device, log_dir, fold+1, 4, 2, "Partial Fine-Tuning")
            best = os.path.join(BEST_DIR, f"best_partial_fine-tuning_fold{fold+1}.pth")
            if os.path.exists(best): model.load_state_dict(torch.load(best, map_location=device))
            v = evaluate_model_with_metrics(model, val_loader, crit, subcategories, device, fold+1, 999,
                                            "Partial Fine-Tuning","Validation")
            t = evaluate_model_with_metrics(model, test_loader, crit, subcategories, device, fold+1, 999,
                                            "Partial Fine-Tuning","Test")
            partial_fold["train_loss"].append(met["train_loss"][-1] if met["train_loss"] else 0)
            partial_fold["train_acc"].append(met["train_acc"][-1] if met["train_acc"] else 0)
            partial_fold["val_loss"].append(v[0]); partial_fold["val_acc"].append(v[1])
            partial_fold["test_loss"].append(t[0]); partial_fold["test_acc"].append(t[1])
            partial_fold["precision"].append(v[2]); partial_fold["recall"].append(v[3]); partial_fold["f1"].append(v[4])
            save_logs_as_zip(log_dir, f"partial_finetune_fold_{fold+1}_logs")
            timing["Partial Fine-Tuning"].append(time.time()-start)
            checkpoint["completed_approaches"][str(fold)].append("partial")
            checkpoint["current_fold"]=fold; checkpoint["partial_fold"]=partial_fold; checkpoint["timing"]=timing
            with open(ckpt_path,"w") as f: json.dump(checkpoint,f)

        # 5) LoRA (custom, both encoders)
        if "lora" not in checkpoint["completed_approaches"][str(fold)]:
            print("\n[5/5] LoRA Adapters (custom; both encoders) ...")
            start=time.time()
            model = LoRAClipBothEncoders(clip_model, subcategories, device,
                                         lora_r=8, lora_alpha=16, lora_dropout=0.05).to(device)
            crit = nn.CrossEntropyLoss(weight=class_weights)
            params = [p for p in model.parameters() if p.requires_grad]  # only LoRA params
            opt  = optim.AdamW(params, lr=2e-4)
            log_dir = os.path.join(BASE_DIR, "raw_lora_finetune_fold", f"{fold+1}")
            met = train_model(model, train_loader, val_loader, test_loader, crit, opt, 4,
                              subcategories, device, log_dir, fold+1, 4, 2, "LoRA Adapters")
            best = os.path.join(BEST_DIR, f"best_lora_adapters_fold{fold+1}.pth")
            if os.path.exists(best): model.load_state_dict(torch.load(best, map_location=device))
            v = evaluate_model_with_metrics(model, val_loader, crit, subcategories, device, fold+1, 999,
                                            "LoRA Adapters","Validation")
            t = evaluate_model_with_metrics(model, test_loader, crit, subcategories, device, fold+1, 999,
                                            "LoRA Adapters","Test")
            lora_fold["train_loss"].append(met["train_loss"][-1] if met["train_loss"] else 0)
            lora_fold["train_acc"].append(met["train_acc"][-1] if met["train_acc"] else 0)
            lora_fold["val_loss"].append(v[0]); lora_fold["val_acc"].append(v[1])
            lora_fold["test_loss"].append(t[0]); lora_fold["test_acc"].append(t[1])
            lora_fold["precision"].append(v[2]); lora_fold["recall"].append(v[3]); lora_fold["f1"].append(v[4])
            save_logs_as_zip(log_dir, f"lora_finetune_fold_{fold+1}_logs")
            timing["LoRA Adapters"].append(time.time()-start)
            checkpoint["completed_approaches"][str(fold)].append("lora")
            checkpoint["current_fold"]=fold; checkpoint["lora_fold"]=lora_fold; checkpoint["timing"]=timing
            with open(ckpt_path,"w") as f: json.dump(checkpoint,f)

    print("\nLengths before plotting:")
    for name,d in [("Full",full_fold),("Text",text_fold),("Vision",vision_fold),("Partial",partial_fold),("LoRA",lora_fold)]:
        print(name,{k:len(v) for k,v in d.items()})

    # Per-approach plots
    def plot_fold_wrapper(name, data):
        plot_fold_metrics(data, name)
    plot_fold_wrapper("Full Fine-Tuning",    full_fold)
    plot_fold_wrapper("Text Encoder Fine-Tuning", text_fold)
    plot_fold_wrapper("Vision-Only Fine-Tuning",  vision_fold)
    plot_fold_wrapper("Partial Fine-Tuning",      partial_fold)
    plot_fold_wrapper("LoRA Adapters",            lora_fold)

    # Comparisons
    compare_metrics(full_fold, text_fold, vision_fold, partial_fold, lora_fold, k_folds)
    plot_full_vs_partial_vs_lora(full_fold, partial_fold, lora_fold, k_folds)

    # CSV dumps
    def dump(name, d): save_csv(d, f"{name}.csv")
    dump("kfold_full_finetune_metrics",    full_fold)
    dump("kfold_text_encoder_metrics",     text_fold)
    dump("kfold_vision_only_metrics",      vision_fold)
    dump("kfold_partial_finetune_metrics", partial_fold)
    dump("kfold_lora_adapters_metrics",    lora_fold)

    create_fold_comparison_table(full_fold, text_fold, vision_fold, partial_fold, lora_fold, k_folds)

    # Timing CSV
    timing_rows=[]
    for approach, durations in timing.items():
        if durations:
            for i, dur in enumerate(durations, 1):
                h, rem = divmod(dur,3600); m, s = divmod(rem,60)
                timing_rows.append({"Approach":approach,"Fold":i,"Duration (seconds)":dur,
                                    "Formatted Duration":f"{int(h):02d}:{int(m):02d}:{int(s):02d}"})
        else:
            for i in range(1, k_folds+1):
                timing_rows.append({"Approach":approach,"Fold":i,"Duration (seconds)":0,"Formatted Duration":"N/A"})
    save_csv(pd.DataFrame(timing_rows), "execution_timing.csv")

    overall_end = time.time()
    h, rem = divmod(overall_end-overall_start,3600); m, s = divmod(rem,60)
    print(f"\nFinished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(overall_end))}")
    print(f"Total Duration: {int(h):02d}:{int(m):02d}:{int(s):02d}")