# -*- coding: utf-8 -*-
import os, json, time, shutil, warnings
from zipfile import ZipFile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# AMP that works for PT1.x / PT2.x
try:
    from torch.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PT 2.x
    AMP_IS_V2 = True
except Exception:
    from torch.cuda.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PT 1.x
    AMP_IS_V2 = False

def make_scaler(device): return GradScalerAmp(enabled=(device.type == "cuda"))

@contextmanager
def autocast_ctx(device):
    if AMP_IS_V2:
        with autocast_amp(device_type=device.type, enabled=(device.type != "cpu")):
            yield
    else:
        with autocast_amp(enabled=torch.cuda.is_available()):
            yield

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from datasets import load_dataset
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType

# ------------------------------
# Output dirs
# ------------------------------
BASE_DIR = os.path.join("logs", "main3-5b")
PNG_DIR  = os.path.join(BASE_DIR, "png")
CSV_DIR  = os.path.join(BASE_DIR, "csv")
BEST_DIR = os.path.join(BASE_DIR, "best")
for d in [PNG_DIR, CSV_DIR, BEST_DIR]:
    os.makedirs(d, exist_ok=True)

warnings.filterwarnings("ignore")

# ------------------------------
# Dataset -> we keep raw PIL images and batch-process with CLIPProcessor
# ------------------------------
class FashionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, subcategories):
        self.dataset = dataset
        self.subcategories = subcategories

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")      # PIL Image
        label = self.subcategories.index(item["subCategory"])
        return image, label

def collate_fn_processor(batch, processor, device):
    images, labels = zip(*batch)
    enc = processor(images=images, return_tensors="pt")
    pixel_values = enc["pixel_values"]  # (B, 3, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return pixel_values.to(device), labels.to(device)

# ------------------------------
# Utilities
# ------------------------------
def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def load_fashion_dataset():
    dataset = load_dataset("ceyda/fashion-products-small")
    train_data = dataset["train"]
    subcategories = sorted(set(item["subCategory"] for item in train_data))
    print(f"Loaded dataset with {len(subcategories)} subcategories and {len(train_data)} samples.")
    return train_data, subcategories

def compute_class_weights(dataset, subcategories):
    labels = [subcategories.index(item["subCategory"]) for item in dataset]
    counts = np.bincount(labels, minlength=len(subcategories))
    total = len(labels)
    weights = total / (len(subcategories) * np.clip(counts, 1, None))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float)

def save_csv(data, filename):
    out = os.path.join(CSV_DIR, filename)
    pd.DataFrame(data).to_csv(out, index=False)
    print(f"CSV saved: {out}")
    return out

def save_png_current_figure(filename):
    out = os.path.join(PNG_DIR, filename)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"PNG saved: {out}")
    return out

def plot_confusion_matrix(cm, classes, model_type, fold, epoch, split_type="Validation"):
    import seaborn as sns
    n = cm.shape[0]
    labels = classes[:n] if n <= len(classes) else [f"Class_{i}" for i in range(n)]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {model_type} (Fold {fold}, Epoch {epoch}, {split_type})")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    save_png_current_figure(f"{model_type.replace(' ', '_').lower()}_fold{fold}_epoch{epoch}_{split_type.lower()}_confmat.png")

def analyze_misclassifications(cm, classes, model_type, fold, epoch, split_type="Validation"):
    rows = []
    n = cm.shape[0]
    for t in range(n):
        tot = np.sum(cm[t])
        if tot == 0: continue
        for p in range(n):
            if t != p and cm[t][p] > 0:
                cnt = int(cm[t][p])
                rows.append({
                    "True Class": classes[t] if t < len(classes) else f"Class_{t}",
                    "Predicted Class": classes[p] if p < len(classes) else f"Class_{p}",
                    "Count": cnt,
                    "Percentage of True Class": 100.0 * cnt / tot
                })
    if rows:
        df = pd.DataFrame(rows).sort_values("Count", ascending=False).head(20)
        out = os.path.join(CSV_DIR, f"{model_type.replace(' ', '_').lower()}_fold{fold}_epoch{epoch}_{split_type.lower()}_misclass.csv")
        df.to_csv(out, index=False)
        print(f"Misclassifications saved: {out}")

# ------------------------------
# CLIP + PEFT LoRA wrapper
# ------------------------------
class CLIPWithPEFT(nn.Module):
    """
    Wraps transformers' CLIPModel and applies PEFT LoRA to vision_model and text_model.
    Trains with classification via CLIP similarity against class prompts.
    """
    def __init__(self, model_name, device, subcategories, lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.subcategories = subcategories
        self.device = device

        self.clip = self.clip.to(device)

        # Prepare class prompts tokenizer (will be used frequently)
        self.class_prompts = [f"a photo of {c}" for c in subcategories]

        if lora:
            # Apply LoRA to both encoders (vision and text)
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            # Wrap submodules in-place
            self.clip.vision_model = get_peft_model(self.clip.vision_model, cfg)
            self.clip.text_model   = get_peft_model(self.clip.text_model, cfg)
            # Only LoRA params trainable (base frozen). If you want base to train too, remove this:
            for n, p in self.clip.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False

    def forward(self, pixel_values, text_inputs=None):
        """
        Return logits over classes for each image in `pixel_values`.
        If `text_inputs` (tokenized prompts) not provided, will tokenize class prompts.
        """
        # Compute image features
        image_features = self.clip.get_image_features(pixel_values=pixel_values)  # (B, D)
        # Tokenize prompts if not provided
        if text_inputs is None:
            text_inputs = self.processor.tokenizer(self.class_prompts, padding=True, return_tensors="pt").to(self.device)
        text_features = self.clip.get_text_features(**text_inputs)                # (C, D)

        # Normalize and compute similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features /  text_features.norm(dim=-1,  keepdim=True)
        # Use CLIP learnable logit scale
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.t())               # (B, C)
        return logits

# ------------------------------
# Evaluate & Train
# ------------------------------
def evaluate(model, loader, criterion, subcats, device, fold, epoch, model_type, split_type="Validation"):
    model.eval()
    tot_loss, all_preds, all_labels = 0.0, [], []

    # Prepare text tokenization ONCE per evaluation (grad not needed)
    with torch.no_grad():
        prompt_inputs = model.processor.tokenizer(model.class_prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        for pixel_values, labels in loader:
            outputs = model(pixel_values, text_inputs=prompt_inputs)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = tot_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                               average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, subcats, model_type, fold, epoch, split_type)
    analyze_misclassifications(cm, subcats, model_type, fold, epoch, split_type)

    print(f"{model_type} {split_type} (Fold {fold}, Epoch {epoch}): "
          f"Loss={avg_loss:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    return avg_loss, acc, precision, recall, f1, cm

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer,
                num_epochs, subcats, device, fold, model_type="PEFT LoRA",
                accumulation_steps=4, validate_every=1):
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

        # Text tokenization each epoch (with grad if text LoRA is active)
        prompt_inputs = model.processor.tokenizer(model.class_prompts, padding=True, return_tensors="pt").to(device)

        with tqdm(train_loader, desc=f"{model_type} | Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for bidx, (pixel_values, labels) in enumerate(pbar):
                with autocast_ctx(device):
                    logits = model(pixel_values, text_inputs=prompt_inputs)
                    loss = criterion(logits, labels) / accumulation_steps

                scaler.scale(loss).backward()
                if (bidx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

                total_train_loss += loss.item() * accumulation_steps
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == labels).sum().item()
                total_train   += labels.size(0)

                pbar.set_postfix({
                    "loss": f"{(total_train_loss / (bidx+1)):.4f}",
                    "acc":  f"{(correct_train / max(total_train,1)):.4f}"
                })

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        train_acc = correct_train / max(total_train, 1)

        # Validate + Test
        v_loss, v_acc, v_p, v_r, v_f1, _ = evaluate(
            model, val_loader, criterion, subcats, device, fold, epoch+1, model_type, "Validation")
        t_loss, t_acc, t_p, t_r, t_f1, _ = evaluate(
            model, test_loader, criterion, subcats, device, fold, epoch+1, model_type, "Test")

        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(avg_train_loss); metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(v_loss); metrics["val_acc"].append(v_acc)
        metrics["test_loss"].append(t_loss); metrics["test_acc"].append(t_acc)
        metrics["precision_val"].append(v_p); metrics["recall_val"].append(v_r); metrics["f1_val"].append(v_f1)
        metrics["precision_test"].append(t_p); metrics["recall_test"].append(t_r); metrics["f1_test"].append(t_f1)

        # Early stopping based on Val F1
        if v_f1 > best_f1:
            best_f1 = v_f1; patience_counter = 0
            best_path = os.path.join(BEST_DIR, f"best_{model_type.replace(' ','_').lower()}_fold{fold}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved: {best_path} (Val F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")

    # Per-epoch CSV
    save_csv(metrics, f"{model_type.replace(' ','_').lower()}_fold{fold}_per_epoch.csv")
    return metrics

# ------------------------------
# Simple fold plots
# ------------------------------
def plot_fold_summary(metrics, title_suffix="PEFT LoRA"):
    def g(k): return metrics.get(k, [])
    n = len(g("val_loss"))
    if n == 0: return
    xs = list(range(1, n+1))
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1); plt.plot(xs, g("val_loss"),  marker="o", label="Val Loss");  plt.title(f"Val Loss — {title_suffix}");  plt.legend()
    plt.subplot(2,2,2); plt.plot(xs, g("test_loss"), marker="o", label="Test Loss"); plt.title(f"Test Loss — {title_suffix}"); plt.legend()
    plt.subplot(2,2,3); plt.plot(xs, g("val_acc"),   marker="o", label="Val Acc");   plt.title(f"Val Acc — {title_suffix}");   plt.legend()
    plt.subplot(2,2,4); plt.plot(xs, g("test_acc"),  marker="o", label="Test Acc");  plt.title(f"Test Acc — {title_suffix}");  plt.legend()
    save_png_current_figure(f"fold_metrics_{title_suffix.replace(' ','_')}.png")

# ------------------------------
# Main (5-fold CV with PEFT LoRA)
# ------------------------------
if __name__ == "__main__":
    overall_start = time.time()
    device = setup_environment()

    # Data
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories).to(device)

    # 5-fold split (80/20 train+val | test). Inside train+val, we split 80/20 into train/val.
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Track per-fold summary
    fold_rows = []

    # Training config
    NUM_EPOCHS = 4
    BATCH_SIZE = 32
    ACC_STEPS  = 4
    LR         = 2e-4

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(dataset), start=1):
        print(f"\n--- Fold {fold}/{k_folds} ---")
        tv_size = len(train_val_idx)
        train_size = int(0.8 * tv_size)
        train_idx = train_val_idx[:train_size]
        val_idx   = train_val_idx[train_size:]

        # Subsets
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset   = Subset(dataset, val_idx.tolist())
        test_subset  = Subset(dataset, test_idx.tolist())

        # Datasets
        train_ds = FashionDataset(train_subset, subcategories)
        val_ds   = FashionDataset(val_subset,   subcategories)
        test_ds  = FashionDataset(test_subset,  subcategories)

        # Build a temporary model to get the processor for the collate_fn
        tmp_model = CLIPWithPEFT("openai/clip-vit-base-patch32", device, subcategories, lora=False)
        processor = tmp_model.processor
        del tmp_model

        # DataLoaders (use processor in collate_fn)
        def collate_train(b): return collate_fn_processor(b, processor, device)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=min(8, os.cpu_count()), collate_fn=collate_train, pin_memory=True)

        def collate_eval(b): return collate_fn_processor(b, processor, device)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=min(8, os.cpu_count()), collate_fn=collate_eval, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=min(8, os.cpu_count()), collate_fn=collate_eval, pin_memory=True)

        # ----- Build PEFT LoRA model -----
        model = CLIPWithPEFT(
            model_name="openai/clip-vit-base-patch32",
            device=device,
            subcategories=subcategories,
            lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05
        )

        # Loss & Optim
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        params = [p for p in model.parameters() if p.requires_grad]  # only LoRA params
        optimizer = optim.AdamW(params, lr=LR)

        # Train
        metrics = train_model(
            model, train_loader, val_loader, test_loader, criterion, optimizer,
            NUM_EPOCHS, subcategories, device, fold,
            model_type="PEFT LoRA", accumulation_steps=ACC_STEPS, validate_every=1
        )

        # Record fold summary
        fold_rows.append({
            "Fold": fold,
            "Val Loss":  metrics["val_loss"][-1] if metrics["val_loss"] else 0.0,
            "Val Acc":   metrics["val_acc"][-1]  if metrics["val_acc"]  else 0.0,
            "Test Loss": metrics["test_loss"][-1]if metrics["test_loss"] else 0.0,
            "Test Acc":  metrics["test_acc"][-1] if metrics["test_acc"]  else 0.0,
        })

        # Fold plots
        plot_fold_summary(metrics, title_suffix=f"PEFT LoRA (fold {fold})")

    # Save per-fold summary + average
    df_folds = pd.DataFrame(fold_rows)
    if not df_folds.empty:
        avg_row = {"Fold": "Average"}
        for k in ["Val Loss","Val Acc","Test Loss","Test Acc"]:
            avg_row[k] = df_folds[k].mean()
        df_folds = pd.concat([df_folds, pd.DataFrame([avg_row])], ignore_index=True)
    df_folds.to_csv(os.path.join(CSV_DIR, "peft_lora_fold_summary.csv"), index=False)
    print(f"Saved: {os.path.join(CSV_DIR, 'peft_lora_fold_summary.csv')}")

    total_s = int(time.time() - overall_start)
    h, rem = divmod(total_s, 3600); m, s = divmod(rem, 60)
    print(f"\nFinished. Total time: {h:02d}:{m:02d}:{s:02d}")
