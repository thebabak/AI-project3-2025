# %% [markdown]
## Image Classification with ConvNeXt: Full Workflow + TensorFlow Logging
# - Manual/local weight loading option
# - GPU utilization printouts
# - Batch visualization
# - Confusion matrix
# - TensorFlow summary logging (scalars + images) per fold

# %%
# Imports
import os
import io
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter

# ---- Quiet some logs; keep oneDNN on (default) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---- Optional: TensorFlow logging (mirrors metrics & images) ----
try:
    import tensorflow as tf
    TF_OK = True
except Exception:
    TF_OK = False
    tf = None

# Where TensorBoard logs will be written (scalars + images)
BASE_TF_LOGDIR = "logs_convnext"

# %% [markdown]
### 1. Initialize Environment

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    if TF_OK:
        print(f"TensorFlow version: {tf.__version__}")
    else:
        print("TensorFlow not found; tf.summary logging will be skipped.")
    return device

device = setup_environment()

# %% [markdown]
### 2. Load ConvNeXt Model (OpenCLIP) with Optional Local Weights

def load_and_print_convnext_model(device, model_name="convnext_base_w",
                                  pretrained_weights="laion2b_s13b_b82k_augreg",
                                  local_weights_path=None):
    """
    Load ConvNeXt (OpenCLIP) model and return model + preprocess transforms.
    - Set local_weights_path to load a local checkpoint instead of downloading.
    """
    try:
        if local_weights_path:
            print(f"Loading model weights from local path: {local_weights_path}")
            model, preproc_train, preproc_val = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=None, device=device
            )
            state = torch.load(local_weights_path, map_location=device)
            model.load_state_dict(state)
        else:
            model, preproc_train, preproc_val = open_clip.create_model_and_transforms(
                model_name=model_name, pretrained=pretrained_weights, device=device
            )

        print("\nModel structure (truncated):")
        print(repr(model)[:1500])  # avoid huge dumps
        print("\nPreprocessing (train):", preproc_train)
        print("\nPreprocessing (val):", preproc_val)
        return model, preproc_train, preproc_val
    except Exception as e:
        print(f"Error loading ConvNeXt model: {e}")
        raise

convnext_model, preprocess_train, preprocess_val = load_and_print_convnext_model(
    device,
    model_name="convnext_base_w",
    pretrained_weights="laion2b_s13b_b82k_augreg",
    local_weights_path=None,  # set a path to load your own weights
)

# %% [markdown]
### 3. Load & Explore Dataset

def load_fashion_dataset():
    ds = load_dataset("ceyda/fashion-products-small")
    dtrain = ds["train"]
    # Peek a sample
    sample = dtrain[0]
    print("\nSample keys:", list(sample.keys()))
    for k, v in sample.items():
        if k != "image":
            print(f"{k}: {v}")
    # Quick display
    plt.figure(figsize=(4, 4))
    plt.imshow(sample["image"])
    plt.title(f"Sample: {sample['subCategory']}")
    plt.axis("off")
    plt.show()

    subcats = sorted(set(dtrain["subCategory"]))
    print(f"Found {len(subcats)} subcategories; first 5:", subcats[:5])
    return dtrain, subcats

dataset, subcategories = load_fashion_dataset()

# %% [markdown]
### 4. Inspect Class Distribution

def inspect_class_distribution(dataset):
    counts = Counter(dataset["subCategory"])
    print("Class distribution (top 20):")
    for label, count in counts.most_common(20):
        print(f"{label:30s} {count}")
    print(f"Total samples: {sum(counts.values())}")

inspect_class_distribution(dataset)

# %% [markdown]
### 5. Class Weights

def compute_class_weights(dataset, subcategories, device):
    counts = Counter(dataset["subCategory"])
    total = sum(counts.values())
    # weight = total / count (inverse frequency)
    weights = [total / max(counts.get(sc, 1), 1) for sc in subcategories]
    return torch.tensor(weights, dtype=torch.float, device=device)

class_weights = compute_class_weights(dataset, subcategories, device)

# %% [markdown]
### 6. Custom Dataset + Augmentations

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

class FashionDataset(Dataset):
    def __init__(self, data, subcategories, transform=None, augment=False):
        self.data = data
        self.subcategories = subcategories
        self.augment = augment
        self.base_transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = int(idx)
        item = self.data[idx]
        img = item["image"]
        if self.augment:
            img = self.aug_transform(img)
        img = self.base_transform(img)
        label = self.subcategories.index(item["subCategory"])
        return img, label

# %% [markdown]
### 7. TensorFlow Summary Helpers (images + scalars)

def _figure_to_tf_image(fig):
    """Convert a Matplotlib figure to a 4D uint8 tensor (1,H,W,C) for tf.summary.image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = tf.image.decode_png(buf.getvalue(), channels=4)  # RGBA
    img = tf.expand_dims(img, 0)
    return img

def tf_log_scalar(writer, tag, value, step):
    if writer is None:
        return
    with writer.as_default():
        tf.summary.scalar(tag, float(value), step=step)

def tf_log_figure(writer, tag, fig, step):
    if writer is None:
        return
    img = _figure_to_tf_image(fig)
    with writer.as_default():
        tf.summary.image(tag, img, step=step)

def tf_log_images_grid(writer, tag, tensor_4d_chw, mean=CLIP_MEAN, std=CLIP_STD, step=0, max_images=10):
    """
    tensor_4d_chw: (B,C,H,W) torch tensor in normalized space -> will be denormalized for logging
    """
    if writer is None:
        return
    imgs = tensor_4d_chw.detach().cpu().clone()
    # denormalize
    for c in range(3):
        imgs[:, c] = imgs[:, c] * std[c] + mean[c]
    imgs = torch.clamp(imgs, 0, 1)  # safety
    # Make a matplotlib grid for consistency
    k = min(len(imgs), max_images)
    fig = plt.figure(figsize=(2*k, 2))
    for i in range(k):
        ax = fig.add_subplot(1, k, i+1)
        ax.imshow(np.transpose(imgs[i].numpy(), (1, 2, 0)))
        ax.axis("off")
    plt.suptitle(tag)
    tf_log_figure(writer, tag, fig, step)

# %% [markdown]
### 8. Batch Visualization

def visualize_batch_samples(loader, subcategories, title="Batch Samples", tf_writer=None, step=0):
    # Get a batch
    images, labels = next(iter(loader))
    # CPU for plotting + TF
    den_mean = torch.tensor(CLIP_MEAN)
    den_std  = torch.tensor(CLIP_STD)

    dn = images.clone()
    for i in range(3):
        dn[:, i] = dn[:, i] * den_std[i] + den_mean[i]

    plt.figure(figsize=(15, 5))
    for i in range(min(len(images), 10)):
        plt.subplot(1, 10, i + 1)
        plt.imshow(dn[i].permute(1, 2, 0).numpy())
        plt.title(subcategories[int(labels[i])], fontsize=8)
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    fig = plt.gcf()
    plt.show()

    # Also log to TensorBoard (TF) as image
    if TF_OK and tf_writer is not None:
        tf_log_figure(tf_writer, f"Samples/{title}", fig, step)

# %% [markdown]
### 9. Model, Train, Evaluate

class ConvNeXtFineTuner(nn.Module):
    """
    Wrap the OpenCLIP ConvNeXt visual with a small linear head.
    Freezes the base (you can unfreeze if you want).
    Dynamically detects feature dim.
    """
    def __init__(self, base_model, num_classes, device):
        super().__init__()
        self.convnext = base_model
        for p in self.convnext.parameters():
            p.requires_grad = False

        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224, device=device)
            feat_dim = self.convnext.encode_image(dummy).shape[-1]
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, images):
        feats = self.convnext.encode_image(images)
        return self.classifier(feats)

@torch.no_grad()
def evaluate_model(model, loader, subcategories, device, tf_writer=None, step=0, tag_prefix="Val"):
    model.eval()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    acc = 100.0 * correct / max(total, 1)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"[{tag_prefix}] Acc: {acc:.2f}% | P: {precision:.3f} R: {recall:.3f} F1: {f1:.3f}")

    # Confusion matrix (plot & TF)
    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'{tag_prefix} Confusion Matrix')
    plt.tight_layout()
    plt.show()

    if TF_OK and tf_writer is not None:
        tf_log_scalar(tf_writer, f"{tag_prefix}/Accuracy",  acc,    step)
        tf_log_scalar(tf_writer, f"{tag_prefix}/Precision", precision, step)
        tf_log_scalar(tf_writer, f"{tag_prefix}/Recall",    recall,    step)
        tf_log_scalar(tf_writer, f"{tag_prefix}/F1",        f1,        step)
        tf_log_figure(tf_writer, f"{tag_prefix}/ConfusionMatrix", fig, step)

    return acc

def train_model(model, train_loader, val_loader, class_weights, num_epochs=3, lr=1e-4,
                device=torch.device("cpu"), tf_writer=None, fold_idx=0):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    # Log a first grid of training samples
    if TF_OK and tf_writer is not None:
        visualize_batch_samples(train_loader, subcategories,
                                title=f"Training Batch (Fold {fold_idx+1})", tf_writer=tf_writer, step=0)

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        prog = tqdm(train_loader, desc=f'Fold {fold_idx+1} | Epoch {epoch+1}/{num_epochs}')
        for step, (images, labels) in enumerate(prog, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            prog.set_postfix({'loss': f"{running/step:.4f}"})

        avg_loss = running / max(1, len(train_loader))
        print(f"Train loss: {avg_loss:.4f}")

        if TF_OK and tf_writer is not None:
            tf_log_scalar(tf_writer, "Train/Loss", avg_loss, epoch+1)

        # Validate & log
        val_acc = evaluate_model(model, val_loader, subcategories, device,
                                 tf_writer=tf_writer, step=epoch+1, tag_prefix="Val")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_convnext_model_fold{fold_idx+1}.pth')
            print(f"Saved best model for fold {fold_idx+1}: acc={best_acc:.2f}%")

    return model

# %% [markdown]
### 10. K-Fold Cross-Validation (with TF logging per fold)

def k_fold_cross_validation(base_model, dataset, subcategories, k_folds=5, batch_size=32, num_epochs=3, lr=1e-4):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n=== Fold {fold + 1}/{k_folds} ===")

        # Create subsets (ensure Python ints)
        train_subset = Subset(dataset, [int(i) for i in train_idx])
        val_subset   = Subset(dataset, [int(i) for i in val_idx])

        # Datasets & loaders
        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset   = FashionDataset(val_subset,   subcategories, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)

        # TensorFlow writer for this fold
        if TF_OK:
            fold_logdir = os.path.join(BASE_TF_LOGDIR, f"fold_{fold+1}")
            os.makedirs(fold_logdir, exist_ok=True)
            tf_writer = tf.summary.create_file_writer(fold_logdir)
            # Write an init marker
            with tf_writer.as_default():
                tf.summary.text("Run/Info", f"ConvNeXt fold {fold+1}", step=0)
                tf.summary.scalar("Run/InitMarker", 0.0, step=0)
        else:
            tf_writer = None

        # Fresh fine-tuner per fold
        model_ft = ConvNeXtFineTuner(base_model, len(subcategories), device).to(device)

        # Train
        model_ft = train_model(model_ft, train_loader, val_loader, class_weights,
                               num_epochs=num_epochs, lr=lr, device=device,
                               tf_writer=tf_writer, fold_idx=fold)

        # Final validation for the fold
        val_acc = evaluate_model(model_ft, val_loader, subcategories, device,
                                 tf_writer=tf_writer, step=num_epochs, tag_prefix="ValFinal")
        results[fold] = val_acc
        print(f"Fold {fold + 1} Validation Accuracy: {val_acc:.2f}%")

        # Close TF writer
        if TF_OK and tf_writer is not None:
            tf_writer.flush()
            tf_writer.close()

    avg_acc = sum(results.values()) / max(1, k_folds)
    print(f"\nAverage Validation Accuracy across {k_folds} folds: {avg_acc:.2f}%")
    return results

# %% [markdown]
### 11. Run it

# (Re)load dataset & subcategories (idempotent if already loaded)
dataset, subcategories = load_fashion_dataset()

# Do K-fold with TensorFlow logging
results = k_fold_cross_validation(
    convnext_model,
    dataset,
    subcategories,
    k_folds=5,
    batch_size=32,
    num_epochs=3,
    lr=1e-4
)

print("Per-fold results:", results)
