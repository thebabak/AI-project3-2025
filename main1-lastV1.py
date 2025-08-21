# %% [markdown]
# Image Classification with ConvNeXt + CLIP (Windows-safe, Full, TensorBoard)
# - Installs (run once):
# !pip install -U open_clip_torch datasets torchvision matplotlib seaborn tqdm scikit-learn tensorboard

# %%
import os
import time
import math
import random
from collections import Counter
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from torchvision import transforms
from torchvision.utils import make_grid

import open_clip

# -------------------- Quiet logs --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# -------------------- Reproducibility & Device --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  CUDA: {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return device

# -------------------- Model helpers --------------------
def list_convnext_pretrained(limit=20):
    print("\n=== Some ConvNeXt pretrained pairs in open_clip ===")
    shown = 0
    for name, tag in open_clip.list_pretrained():
        if "convnext" in name.lower():
            print(f"{name:22s} -> {tag}")
            shown += 1
            if shown >= limit:
                break
    if shown == 0:
        print("No ConvNeXt entries displayed (try printing all list_pretrained()).")

def load_convnext_base_w(device, pretrained="laion2b_s13b_b82k_augreg"):
    print("\nLoading convnext_base_w...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name="convnext_base_w",
        pretrained=pretrained,
        device=device
    )
    print("\n=== Model (truncated) ===")
    print(model)
    print("\n=== Train preprocess ===")
    print(preprocess_train)
    print("\n=== Val preprocess ===")
    print(preprocess_val)
    return model, preprocess_train, preprocess_val

# -------------------- Data & visualization helpers --------------------
def show_raw_samples(hf_dataset, n=8):
    idxs = random.sample(range(len(hf_dataset)), k=int(n))  # Python ints only
    batch = hf_dataset.select(idxs)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3))
    for ax, im, lab in zip(axes, batch["image"], batch["subCategory"]):
        ax.imshow(im)
        ax.set_title(lab, fontsize=9)
        ax.axis('off')
    plt.tight_layout(); plt.show()

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3,1,1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3,1,1)

def show_preprocessed_batch(loader, n=16):
    xb, yb = next(iter(loader))
    xb = xb[:n].cpu()
    xb_dn = xb * CLIP_STD + CLIP_MEAN
    xb_dn = torch.clamp(xb_dn, 0.0, 1.0)  # manual clamp for older torchvision
    grid = make_grid(xb_dn, nrow=int(math.sqrt(n)))
    plt.figure(figsize=(6,6)); plt.imshow(grid.permute(1,2,0)); plt.axis('off'); plt.show()

# -------------------- Dataset wrapper --------------------
class FashionDataset(Dataset):
    """
    Wrap a HuggingFace Dataset and apply:
      - optional light augmentation (PIL-level)
      - CLIP preprocess (from open_clip)
      - compute per-split class weights
    """
    def __init__(self, hf_data, class_names, preprocess, device=None, augment=False):
        self.hf_data = hf_data
        self.class_names = class_names
        self.preprocess = preprocess
        self.augment = augment

        self.extra_aug = transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomHorizontalFlip(p=0.5),
        ], p=0.4)

        counts = Counter(hf_data["subCategory"])
        total = sum(counts.values())
        self.class_weights = torch.tensor(
            [total / (counts[c] + 1e-6) for c in class_names],
            dtype=torch.float32
        )
        if device is not None:
            self.class_weights = self.class_weights.to(device)

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[int(idx)]
        img = item["image"]
        if self.augment:
            img = self.extra_aug(img)
        img = self.preprocess(img)
        label = self.class_names.index(item["subCategory"])
        return img, label

def make_loader(ds, batch_size=64, shuffle=False, num_workers=4):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

# -------------------- Linear Probe --------------------
class LinearProbe(nn.Module):
    def __init__(self, base_model, num_classes, embed_dim):
        super().__init__()
        self.base = base_model
        for p in self.base.parameters():
            p.requires_grad = False
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feats = self.base.encode_image(x)
        return self.classifier(feats)

# -------------------- TensorBoard helper --------------------
def fig_to_tensorboard(writer, fig, tag, global_step):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = plt.imread(buf)
    img = torch.from_numpy(img).permute(2,0,1)  # HWC->CHW
    writer.add_image(tag, img, global_step)

# -------------------- Evaluation --------------------
def evaluate_model(model, loader, class_names, device, writer=None, global_step=None, split_name="val", amp=True):
    model.eval()
    preds, labels = [], []
    amp_ctx = autocast if (amp and device.type == "cuda") else nullcontext
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device)
            with amp_ctx():
                out = model(xb)
            pred = out.argmax(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    precision = precision_score(labels, preds, average="weighted", zero_division=0)
    recall    = recall_score(labels, preds, average="weighted", zero_division=0)
    f1        = f1_score(labels, preds, average="weighted", zero_division=0)
    acc       = (np.array(preds) == np.array(labels)).mean()

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    fig = plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{split_name} Confusion Matrix")
    plt.xticks(rotation=90); plt.yticks(rotation=0); plt.tight_layout()

    if writer is not None and global_step is not None:
        writer.add_scalar(f"{split_name}/accuracy", acc, global_step)
        writer.add_scalar(f"{split_name}/precision_w", precision, global_step)
        writer.add_scalar(f"{split_name}/recall_w", recall, global_step)
        writer.add_scalar(f"{split_name}/f1_w", f1, global_step)
        fig_to_tensorboard(writer, fig, f"{split_name}/confusion_matrix", global_step)

    plt.close(fig)
    return acc, precision, recall, f1

# -------------------- Training --------------------
def train_model(model, train_loader, val_loader, class_weights, device, class_names,
                epochs=5, lr=1e-4, run_name="run", amp=True):
    logdir = os.path.join("runs2", run_name)
    writer = SummaryWriter(log_dir=logdir)
    print(f"\nTensorBoard: {logdir}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler(enabled=(amp and device.type == "cuda"))
    amp_ctx = autocast if (amp and device.type == "cuda") else nullcontext

    best_val_acc = 0.0
    patience = 0
    max_patience = 2
    global_step = 0

    # Log a sample batch grid
    xb, yb = next(iter(train_loader))
    xb_dn = xb[:16].cpu() * CLIP_STD + CLIP_MEAN
    xb_dn = torch.clamp(xb_dn, 0.0, 1.0)
    grid = make_grid(xb_dn, nrow=4)
    writer.add_image("train/sample_batch", grid, 0)

    for epoch in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, device=device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(xb)
                loss = criterion(out, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/loss", float(loss.item()), global_step)
            global_step += 1

        val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            model, val_loader, class_names=class_names, device=device,
            writer=writer, global_step=global_step, split_name="val", amp=amp
        )
        print(f"Epoch {epoch}: val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(logdir, "best_model.pth"))
        else:
            patience += 1
            if patience >= max_patience:
                print("Early stopping.")
                break

    writer.close()
    return os.path.join(logdir, "best_model.pth")

# -------------------- K-Fold (HF Datasets .select, no torch.Subset) --------------------
def k_fold_cross_validation(hf_dataset, class_names, preprocess_train, preprocess_val,
                            base_model, embed_dim, device,
                            k_folds=5, batch_size=64, epochs=3, lr=1e-4, amp=True):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    all_acc = []
    indices = np.arange(len(hf_dataset))

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(indices), start=1):
        print(f"\n{'='*40}\nFold {fold}/{k_folds}\n{'='*40}")
        tr_idx = [int(i) for i in tr_idx]
        va_idx = [int(i) for i in va_idx]

        tr_hf = hf_dataset.select(tr_idx)
        va_hf = hf_dataset.select(va_idx)

        tr_ds = FashionDataset(tr_hf, class_names, preprocess_train, device=device, augment=True)
        va_ds = FashionDataset(va_hf, class_names, preprocess_val,   device=device, augment=False)

        tr_loader = make_loader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        va_loader = make_loader(va_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        print("Fold sample batch:")
        show_preprocessed_batch(tr_loader, n=16)

        probe = LinearProbe(base_model, len(class_names), embed_dim).to(device)
        best_path = train_model(probe, tr_loader, va_loader, tr_ds.class_weights, device, class_names,
                                epochs=epochs, lr=lr, run_name=f"kfold_fold{fold}", amp=amp)

        best = LinearProbe(base_model, len(class_names), embed_dim).to(device)
        best.load_state_dict(torch.load(best_path, map_location=device))
        acc, *_ = evaluate_model(best, va_loader, class_names, device, writer=None, global_step=None,
                                 split_name=f"fold{fold}_val", amp=amp)
        print(f"Fold {fold} val_acc={acc:.3f}")
        all_acc.append(acc)

    print(f"\nK-Fold accuracy: {np.mean(all_acc):.3f} ± {np.std(all_acc):.3f}")
    return all_acc

# -------------------- Cosine Similarity (8 images vs custom descriptions) --------------------
def compute_cosine_similarity(model, preprocess, images, descriptions, device, amp=True):
    model.eval()
    amp_ctx = autocast if (amp and device.type == "cuda") else nullcontext
    with torch.no_grad(), amp_ctx():
        imgs = torch.stack([preprocess(im) for im in images]).to(device)
        img_feats = model.encode_image(imgs)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        toks = open_clip.tokenize(descriptions).to(device)
        txt_feats = model.encode_text(toks)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
    sim = (img_feats @ txt_feats.T).detach().cpu().numpy()
    return sim

def visualize_similarity(hf_dataset, class_names, preprocess, device, base_model, n=8, amp=True):
    idxs = random.sample(range(len(hf_dataset)), k=int(n))
    batch = hf_dataset.select(idxs)
    images = list(batch["image"])
    descriptions = [f"A high-quality {sc} product photo" for sc in batch["subCategory"]]

    sims = compute_cosine_similarity(base_model, preprocess, images, descriptions, device, amp=amp)

    plt.figure(figsize=(10,8))
    sns.heatmap(sims, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[f"Desc {i+1}" for i in range(n)],
        yticklabels=[f"Img {i+1}" for i in range(n)])
    plt.title("Image–Text Cosine Similarity"); plt.tight_layout(); plt.show()

    for i in range(n):
        plt.figure(figsize=(3.5,3.5))
        plt.imshow(images[i]); plt.axis('off')
        plt.title(f"{descriptions[i]}\nSim: {sims[i,i]:.2f}")
        plt.tight_layout(); plt.show()

# -------------------- Main (Windows-safe) --------------------
def main():
    set_seed(42)
    device = setup_environment()
    amp = (device.type == "cuda")

    # List and load model
    list_convnext_pretrained()
    convnext_model, preprocess_train, preprocess_val = load_convnext_base_w(device)

    # Load dataset
    ds = load_dataset("ceyda/fashion-products-small")
    raw = ds["train"]
    subcategories = sorted(list(set(raw["subCategory"])))
    num_classes = len(subcategories)
    print(f"\nClasses ({num_classes}): {subcategories}")

    # Raw samples (no DataLoader here)
    print("\nShowing a few raw samples...")
    show_raw_samples(raw, n=8)

    # Split 80/10/10
    splits = raw.train_test_split(test_size=0.2, seed=42)
    train_full, test_split = splits["train"], splits["test"]
    splits2 = train_full.train_test_split(test_size=0.125, seed=42)  # 0.125 of 0.8 = 0.1
    train_split, val_split = splits2["train"], splits2["test"]

    # Wrap into our Dataset
    train_ds = FashionDataset(train_split, subcategories, preprocess_train, device=device, augment=True)
    val_ds   = FashionDataset(val_split,   subcategories, preprocess_val,   device=device, augment=False)
    test_ds  = FashionDataset(test_split,  subcategories, preprocess_val,   device=device, augment=False)

    # DataLoaders (created & iterated inside main -> Windows-safe)
    train_loader = make_loader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader   = make_loader(val_ds,   batch_size=64, shuffle=False, num_workers=4)
    test_loader  = make_loader(test_ds,  batch_size=64, shuffle=False, num_workers=4)

    # Show a preprocessed batch (safe: inside main)
    print("\nShowing a preprocessed training batch...")
    show_preprocessed_batch(train_loader, n=16)

    # Derive embedding dimension from actual preprocess
    with torch.no_grad():
        sample_img = preprocess_val(raw[0]["image"]).unsqueeze(0).to(device)
        embed_dim = convnext_model.encode_image(sample_img).shape[-1]
    print(f"\nDerived image embedding dimension: {embed_dim}")

    # Train linear probe
    probe = LinearProbe(convnext_model, num_classes, embed_dim).to(device)
    run_name = f"convnext_fashion_{int(time.time())}"
    best_ckpt = train_model(
        probe, train_loader, val_loader, class_weights=train_ds.class_weights, device=device,
        class_names=subcategories, epochs=5, lr=1e-4, run_name=run_name, amp=amp
    )

    # Evaluate best on test
    best_probe = LinearProbe(convnext_model, num_classes, embed_dim).to(device)
    best_probe.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_acc, test_prec, test_rec, test_f1 = evaluate_model(
        best_probe, test_loader, subcategories, device,
        writer=None, global_step=None, split_name="test", amp=amp
    )
    print(f"\n== Test Results ==  acc={test_acc:.3f}  prec_w={test_prec:.3f}  rec_w={test_rec:.3f}  f1_w={test_f1:.3f}")

    # Optional: k-fold over the whole dataset (reduce epochs to control time)
    kfold_results = k_fold_cross_validation(
        raw, subcategories, preprocess_train, preprocess_val,
        base_model=convnext_model, embed_dim=embed_dim, device=device,
        k_folds=5, batch_size=64, epochs=3, lr=1e-4, amp=amp
    )

    # Cosine similarity demo
    visualize_similarity(raw, subcategories, preprocess_val, device, convnext_model, n=8, amp=amp)

    print("\nTo view TensorBoard, run:\n  tensorboard --logdir runs2\n")

# Windows-safe entrypoint: NO DataLoader iteration above this guard
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
