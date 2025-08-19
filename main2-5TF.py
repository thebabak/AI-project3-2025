# -*- coding: utf-8 -*-
import os
import io
import json
import shutil
from multiprocessing import freeze_support
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

# -----------------------------
# Env setup
# -----------------------------
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
torch.backends.cudnn.benchmark = True

# ---- AMP compatibility (PyTorch 1.x & 2.x) ----
try:
    from torch.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PyTorch 2.x
    AMP_IS_V2 = True
except Exception:
    from torch.cuda.amp import GradScaler as GradScalerAmp, autocast as autocast_amp  # PyTorch 1.x
    AMP_IS_V2 = False

def autocast_ctx(device: torch.device):
    if AMP_IS_V2:
        return autocast_amp(device_type=device.type, enabled=(device.type != "cpu"))
    else:
        # cuda-only autocast in 1.x
        return autocast_amp(enabled=torch.cuda.is_available())

def make_scaler(device: torch.device):
    return GradScalerAmp(enabled=(device.type == "cuda"))

# ---- TensorFlow (for TF summaries only; keep TF on CPU) ----
try:
    import tensorflow as tf
    try:
        # Hide GPUs from TF so it won't allocate VRAM
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
        print("TF GPU visibility config:", e)
    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed; TF summaries disabled:", e)
    TF_AVAILABLE = False

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Device / environment
# -----------------------------
def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"GPU cache clear error: {e}")
            device = torch.device("cpu")
    return device

# -----------------------------
# Model (OpenCLIP)
# -----------------------------
import open_clip

def load_clip_model(device, model_name="ViT-B-32", pretrained_weights="openai"):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained_weights, device=device
    )
    print("\nModel structure:")
    print(model)
    return model, preprocess_train, preprocess_val

class CustomCLIPFineTuner(nn.Module):
    def __init__(self, base_model, num_classes, num_layers=1):
        super().__init__()
        self.visual_encoder = base_model.visual
        # Freeze most layers (keep last blocks trainable)
        for name, p in self.visual_encoder.named_parameters():
            p.requires_grad = False
        # You can selectively unfreeze final block(s) if desired:
        for name, p in self.visual_encoder.named_parameters():
            if any(k in name.lower() for k in ["ln_post", "proj", "attn", "mlp", "resblocks.11", "resblocks.10"]):
                p.requires_grad = True
        # Probe output dim
        dev = next(base_model.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]
        # Simple classifier head (num_layers=1 by default)
        layers = []
        in_dim = out_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Dropout(0.2)]
        layers += [nn.Linear(in_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, images):
        feats = self.visual_encoder(images)
        return self.classifier(feats)

class NonFineTunedCLIP(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.visual_encoder = base_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = False
        dev = next(base_model.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, device=dev)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, images):
        feats = self.visual_encoder(images)
        return self.classifier(feats)

# -----------------------------
# Dataset
# -----------------------------
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
        self.aug = transforms.Compose([
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
            image = self.aug(image)
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
    return torch.tensor(weights, dtype=torch.float32)

# -----------------------------
# Metrics / Visualization
# -----------------------------
def compute_metrics(true_labels, predictions, subcategories, return_cm=True):
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(true_labels, predictions) if return_cm else None
    return precision, recall, f1, cm

def fig_from_confusion(cm, subcategories, title="Confusion Matrix"):
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    fig.tight_layout()
    return fig

def tf_log_figure(tf_writer, tag, fig, step):
    if not TF_AVAILABLE or tf_writer is None:
        return
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    with tf_writer.as_default():
        tf.summary.image(tag, image, step=step)

# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint(state, ckpt_path):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)

def load_checkpoint(ckpt_path, map_location=None):
    return torch.load(ckpt_path, map_location=map_location)

# -----------------------------
# Zip logs
# -----------------------------
def zip_dir(src_dir, zip_stem):
    if os.path.exists(src_dir):
        os.makedirs(os.path.dirname(zip_stem), exist_ok=True)
        shutil.make_archive(zip_stem, 'zip', src_dir)
        print(f"Zipped logs to: {zip_stem}.zip")

# -----------------------------
# Train / Evaluate (with TB & TF logging + checkpointing)
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.count = 0
        self.stop = False
    def __call__(self, score):
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.count = 0
        else:
            self.count += 1
            if self.count >= self.patience:
                self.stop = True

def evaluate_model_with_metrics(model, loader, criterion, subcategories, device, step_for_plots=None,
                                tb_writer: SummaryWriter=None, tf_writer=None, model_tag="Eval"):
    model.eval()
    tot_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    avg_loss = tot_loss / max(len(loader), 1)
    precision, recall, f1, cm = compute_metrics(y_true, y_pred, subcategories, return_cm=True)

    # Log a confusion matrix image
    if step_for_plots is not None and cm is not None:
        fig = fig_from_confusion(cm, subcategories, f"{model_tag} Confusion Matrix")
        if tb_writer is not None:
            tb_writer.add_figure(f"{model_tag}/ConfusionMatrix", fig, global_step=step_for_plots)
        tf_log_figure(tf_writer, f"{model_tag}/ConfusionMatrix", fig, step_for_plots)

    return avg_loss, precision, recall, f1

def train_model_with_logging(model, train_loader, val_loader, optimizer_type, model_type,
                             subcategories, class_weights, device,
                             num_epochs=3, lr=1e-4, base_log_dir="logs/part2-5-4/", fold_index=0,
                             resume=True):
    # Dirs
    run_dir = os.path.join(base_log_dir, f"{model_type}_{optimizer_type}_fold_{fold_index}")
    tb_torch_dir = os.path.join(run_dir, "tb_torch")
    tb_tf_dir = os.path.join(run_dir, "tb_tf")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(tb_torch_dir, exist_ok=True)
    os.makedirs(tb_tf_dir, exist_ok=True)

    # Writers
    tb_writer = SummaryWriter(log_dir=tb_torch_dir)
    tf_writer = tf.summary.create_file_writer(tb_tf_dir) if TF_AVAILABLE else None

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = make_scaler(device)

    # Checkpoints
    ckpt_last = os.path.join(run_dir, "checkpoint_last.pth")
    ckpt_best = os.path.join(run_dir, "checkpoint_best.pth")
    start_epoch = 0
    best_f1 = -1.0

    if resume and os.path.exists(ckpt_last):
        print(f"[Resume] Loading checkpoint: {ckpt_last}")
        data = load_checkpoint(ckpt_last, map_location=device)
        try:
            model.load_state_dict(data["model"])
            optimizer.load_state_dict(data["optimizer"])
            scaler.load_state_dict(data["scaler"])
            start_epoch = data.get("epoch", 0) + 1
            best_f1 = data.get("best_f1", -1.0)
            print(f"Resumed from epoch {start_epoch}, best_f1={best_f1:.4f}")
        except Exception as e:
            print("Resume failed (fresh start):", e)

    early = EarlyStopping(patience=3)
    history = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"{model_type} {optimizer_type} Fold {fold_index} | Epoch {epoch+1}/{num_epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}", "acc": f"{(100.0*correct/max(total,1)):.2f}"})

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(total, 1)

        # Validation + logs
        step = epoch
        val_loss, precision, recall, f1 = evaluate_model_with_metrics(
            model, val_loader, criterion, subcategories, device,
            step_for_plots=step, tb_writer=tb_writer, tf_writer=tf_writer,
            model_tag=f"{model_type}/{optimizer_type}/Val"
        )

        # Torch TB scalars
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Train/Loss", train_loss, step)
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Train/Accuracy", train_acc, step)
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Val/Loss", val_loss, step)
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Val/Precision", precision, step)
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Val/Recall", recall, step)
        tb_writer.add_scalar(f"{model_type}/{optimizer_type}/Val/F1", f1, step)
        # LR & GradNorm (best-effort)
        try:
            for gi, pg in enumerate(optimizer.param_groups):
                tb_writer.add_scalar(f"{model_type}/{optimizer_type}/LR/group_{gi}", pg.get("lr", 0.0), step)
        except Exception:
            pass

        # TF summaries (mirror)
        if TF_AVAILABLE and tf_writer is not None:
            with tf_writer.as_default():
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Train/Loss", train_loss, step=step)
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Train/Accuracy", train_acc, step=step)
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Val/Loss", val_loss, step=step)
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Val/Precision", precision, step=step)
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Val/Recall", recall, step=step)
                tf.summary.scalar(f"{model_type}/{optimizer_type}/Val/F1", f1, step=step)

        # Save "last" checkpoint
        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1
        }, ckpt_last)

        # Save "best" checkpoint (by F1)
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_f1": best_f1
            }, ckpt_best)

        early(f1)
        history.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        print(f"[{model_type}/{optimizer_type}] Epoch {epoch+1}: "
              f"TrainLoss={train_loss:.4f} Acc={train_acc:.2f} | ValLoss={val_loss:.4f} F1={f1:.4f}")

        if early.stop:
            print("Early stopping triggered.")
            break

    tb_writer.close()
    if TF_AVAILABLE and tf_writer is not None:
        tf_writer.flush()

    # Zip logs (both writers)
    zip_dir(run_dir, os.path.join(base_log_dir, f"{model_type}_{optimizer_type}_fold_{fold_index}_logs"))

    return history, os.path.join(run_dir, "checkpoint_best.pth")

# -----------------------------
# Plotting & CSV
# -----------------------------
def plot_metrics(metrics, k_folds, title_suffix=""):
    folds = range(1, k_folds + 1)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    for k in metrics:
        plt.plot(folds[:len(metrics[k]['val_loss'])], metrics[k]["val_loss"], marker="o", label=f"{k} ValLoss")
    plt.xlabel("Fold"); plt.ylabel("Loss"); plt.title("Validation Loss"); plt.legend()

    plt.subplot(2, 2, 2)
    for k in metrics:
        plt.plot(folds[:len(metrics[k]['precision'])], metrics[k]["precision"], marker="o", label=f"{k} Precision")
    plt.xlabel("Fold"); plt.ylabel("Precision"); plt.title("Validation Precision"); plt.legend()

    plt.subplot(2, 2, 3)
    for k in metrics:
        plt.plot(folds[:len(metrics[k]['recall'])], metrics[k]["recall"], marker="o", label=f"{k} Recall")
    plt.xlabel("Fold"); plt.ylabel("Recall"); plt.title("Validation Recall"); plt.legend()

    plt.subplot(2, 2, 4)
    for k in metrics:
        plt.plot(folds[:len(metrics[k]['f1'])], metrics[k]["f1"], marker="o", label=f"{k} F1")
    plt.xlabel("Fold"); plt.ylabel("F1"); plt.title("Validation F1"); plt.legend()
    plt.tight_layout()
    out = "metrics_plot.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out}")

def save_metrics_nested_to_csv(metrics_nested, out_csv):
    # Flatten nested dict: {key: {metric: [per fold]}}
    rows = []
    # infer num folds
    num_folds = max(len(v.get("val_loss", [])) for v in metrics_nested.values())
    for i in range(num_folds):
        row = {"fold": i+1}
        for key, d in metrics_nested.items():
            for met in ["val_loss", "precision", "recall", "f1"]:
                vals = d.get(met, [])
                row[f"{key} {met}"] = vals[i] if i < len(vals) else None
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    freeze_support()
    set_seed(42)
    device = setup_environment()

    clip_model, preprocess_train, preprocess_val = load_clip_model(
        device, model_name="ViT-B-32", pretrained_weights="openai"
    )
    dataset, subcategories = load_fashion_dataset()
    class_weights = compute_class_weights(dataset, subcategories)

    # DataLoader workers / pin_memory
    num_workers = 0 if os.name == "nt" else min(8, os.cpu_count() or 0)
    pin_mem = (device.type == "cuda")

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    metrics = {
        "adam_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adamw_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adam_non_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []},
        "adamw_non_fine_tuned": {"val_loss": [], "precision": [], "recall": [], "f1": []}
    }

    base_log = "logs/part2-5-4/"
    os.makedirs(base_log, exist_ok=True)

    for optimizer_type in ["adam", "adamw"]:
        for fold_index, (train_idx, val_idx) in enumerate(kf.split(dataset), start=1):
            print(f"\n=== {optimizer_type.upper()} | Fold {fold_index}/{k_folds} ===")
            train_subset = Subset(dataset, train_idx.tolist())
            val_subset   = Subset(dataset, val_idx.tolist())

            train_ds = FashionDataset(train_subset, subcategories, augment=True)
            val_ds   = FashionDataset(val_subset,   subcategories)

            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                      num_workers=num_workers, pin_memory=pin_mem)
            val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin_mem)

            # --- Fine-Tuned model ---
            print(f"Training Fine-Tuned model ({optimizer_type}) ...")
            fine_model = CustomCLIPFineTuner(clip_model, len(subcategories), num_layers=1).to(device)
            hist_ft, best_ckpt_ft = train_model_with_logging(
                fine_model, train_loader, val_loader, optimizer_type=optimizer_type, model_type="fine_tuned",
                subcategories=subcategories, class_weights=class_weights, device=device,
                num_epochs=3, lr=1e-4, base_log_dir=base_log, fold_index=fold_index, resume=True
            )
            # Take last epoch metrics from history
            last = hist_ft[-1] if hist_ft else {"val_loss": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
            metrics[f"{optimizer_type}_fine_tuned"]["val_loss"].append(last["val_loss"])
            metrics[f"{optimizer_type}_fine_tuned"]["precision"].append(last["precision"])
            metrics[f"{optimizer_type}_fine_tuned"]["recall"].append(last["recall"])
            metrics[f"{optimizer_type}_fine_tuned"]["f1"].append(last["f1"])

            # --- Non-Fine-Tuned model ---
            print(f"Training Non-Fine-Tuned model ({optimizer_type}) ...")
            non_model = NonFineTunedCLIP(clip_model, len(subcategories)).to(device)
            hist_nft, best_ckpt_nft = train_model_with_logging(
                non_model, train_loader, val_loader, optimizer_type=optimizer_type, model_type="non_fine_tuned",
                subcategories=subcategories, class_weights=class_weights, device=device,
                num_epochs=3, lr=1e-4, base_log_dir=base_log, fold_index=fold_index, resume=True
            )
            last = hist_nft[-1] if hist_nft else {"val_loss": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}
            metrics[f"{optimizer_type}_non_fine_tuned"]["val_loss"].append(last["val_loss"])
            metrics[f"{optimizer_type}_non_fine_tuned"]["precision"].append(last["precision"])
            metrics[f"{optimizer_type}_non_fine_tuned"]["recall"].append(last["recall"])
            metrics[f"{optimizer_type}_non_fine_tuned"]["f1"].append(last["f1"])

    # Plots & CSV
    plot_metrics(metrics, k_folds)
    save_metrics_nested_to_csv(metrics, os.path.join(base_log, "metrics_summary.csv"))

    print("\nDone. TensorBoard logs are under each run dir:")
    print(" - PyTorch:   .../tb_torch/")
    print(" - TensorFlow .../tb_tf/")
    print("Launch: tensorboard --logdir logs/part2-5-4")
