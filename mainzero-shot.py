# -*- coding: utf-8 -*-
"""
Zero-shot vs Full Fine-Tuning on ceyda/fashion-products-small using OpenCLIP.

- Uses CLIP's own preprocessors (create_model_and_transforms)
- Mixed precision + safe gradient accumulation (flush remainder)
- Windows-friendly DataLoader defaults, pin_memory when CUDA
- Caches zero-shot text features
- Clean TensorBoard logging and denormalized image previews
"""

import os
# ---- Keep TF quiet before any imports that could pull it in (like datasets) ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")        # hide TF INFO/WARN
# os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")     # uncomment if TF still prints oneDNN notes

import time
start_time = time.time()

import warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version", module="google.protobuf.runtime_version")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
from tqdm import tqdm
import open_clip
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, List


# ------------------------- Configuration -------------------------
class Config:
    BASE_DIR = "logs/clip_comparison"
    K_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 4
    VALIDATE_EVERY = 2
    ACCUMULATION_STEPS = 4
    LR = 1e-5
    WEIGHT_DECAY = 1e-4
    SEED = 42

    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


Config.setup_dirs()
set_seed(Config.SEED)


# ------------------------- Dataset -------------------------
class FashionDataset(torch.utils.data.Dataset):
    """
    Wraps HuggingFace dataset split with optional light augmentation
    and always finishes with a provided 'transform' (use CLIP preprocessors).
    """
    def __init__(self, dataset, subcategories: List[str], augment: bool = False, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.subcategories = subcategories
        self.transform = transform
        self.augment_xform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]) if augment else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        if self.augment_xform:
            image = self.augment_xform(image)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.subcategories.index(item['subCategory'])
        return image, label


# ------------------------- Models -------------------------
class ZeroShotCLIP(nn.Module):
    """
    CLIP zero-shot head that caches normalized text features once.
    """
    def __init__(self, base_model, subcategories: List[str], device: torch.device):
        super().__init__()
        self.model = base_model
        self.subcategories = subcategories
        self.device = device
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        with torch.no_grad():
            text_inputs = self.tokenizer([f"a photo of {c}" for c in self.subcategories]).to(self.device)
            txt = self.model.encode_text(text_inputs)
            self.text_features = txt / txt.norm(dim=-1, keepdim=True)  # [C, D]

    def forward(self, images, text_inputs=None):
        with torch.no_grad():
            img = self.model.encode_image(images)                      # [B, D]
            img = img / img.norm(dim=-1, keepdim=True)
        # similarity logits scaled like CLIP does
        return (img @ self.text_features.T) * 100.0                    # [B, C]


class FullFineTunedCLIP(nn.Module):
    """
    Vision-encoder + small classifier head for supervised finetuning.
    """
    def __init__(self, base_model, num_classes: int):
        super().__init__()
        self.visual_encoder = base_model.visual

        device = next(base_model.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            out_dim = self.visual_encoder(dummy).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, *_):
        feats = self.visual_encoder(images)        # [B, D]
        return self.classifier(feats)              # [B, C]


# ------------------------- Experiment -------------------------
class CLIPExperiment:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nRunning on device: {self.device}")

        # Create model + CLIP preprocessors
        self.clip_model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.clip_model = self.clip_model.to(self.device).eval()

        # Load dataset
        ds_dict = load_dataset("ceyda/fashion-products-small")
        self.dataset = ds_dict["train"]
        self.subcategories = sorted(set(item["subCategory"] for item in self.dataset))

        # Class weights for imbalanced loss
        self.class_weights = self._compute_class_weights().to(self.device)

        # Metrics container
        self.metrics = {
            "zero_shot": {"test_acc": [], "test_f1": []},
            "full_ft": {"train_loss": [], "val_acc": [], "val_f1": []}
        }

    def _compute_class_weights(self) -> torch.Tensor:
        labels = [self.subcategories.index(item['subCategory']) for item in self.dataset]
        counts = np.bincount(labels, minlength=len(self.subcategories))
        weights = len(labels) / (len(self.subcategories) * np.clip(counts, 1, None))
        return torch.tensor(weights, dtype=torch.float)

    def _make_loaders(
        self,
        train_subset,
        val_subset,
        test_subset
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        nw = 0 if os.name == "nt" else min(8, os.cpu_count() or 0)
        pin = self.device.type == "cuda"

        train_ds = FashionDataset(train_subset, self.subcategories, augment=True,  transform=self.preprocess_train)
        val_ds   = FashionDataset(val_subset,   self.subcategories, augment=False, transform=self.preprocess_val)
        test_ds  = FashionDataset(test_subset,  self.subcategories, augment=False, transform=self.preprocess_val)

        train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                                  num_workers=nw, pin_memory=pin, persistent_workers=False)
        val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False,
                                  num_workers=nw, pin_memory=pin, persistent_workers=False)
        test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False,
                                  num_workers=nw, pin_memory=pin, persistent_workers=False)
        return train_loader, val_loader, test_loader

    @staticmethod
    def _denorm_for_tb(imgs: torch.Tensor) -> torch.Tensor:
        # imgs were normalized by CLIP; bring them back to [0,1] for nice TB previews
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(imgs.device)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(imgs.device)
        x = imgs * std + mean
        return x.clamp(0, 1)

    def run(self):
        kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=Config.SEED)

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(self.dataset)):
            print(f"\n{'='*40}\nFold {fold+1}/{Config.K_FOLDS}\n{'='*40}")

            writer = SummaryWriter(log_dir=os.path.join(Config.BASE_DIR, f"fold_{fold+1}"))

            # Split indices
            train_val_subset = Subset(self.dataset, train_val_idx.tolist())
            test_subset      = Subset(self.dataset, test_idx.tolist())

            # 80/20 inside train_val for val
            val_size = int(0.2 * len(train_val_subset))
            train_subset = Subset(train_val_subset, range(len(train_val_subset) - val_size))
            val_subset   = Subset(train_val_subset, range(len(train_val_subset) - val_size, len(train_val_subset)))

            # --- Zero-shot ---
            self.run_zero_shot(fold, test_subset, writer)

            # --- Full FT ---
            self.run_full_finetuning(fold, train_subset, val_subset, test_subset, writer)

            writer.close()

        self._save_results()
        print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
        print(f"Results saved to: {Config.BASE_DIR}")

    def run_zero_shot(self, fold: int, test_subset, writer: SummaryWriter):
        print("\n[1/2] Zero-Shot CLIP Evaluation")
        test_ds = FashionDataset(test_subset, self.subcategories, transform=self.preprocess_val)
        nw = 0 if os.name == "nt" else min(8, os.cpu_count() or 0)
        pin = self.device.type == "cuda"
        test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                                 num_workers=nw, pin_memory=pin)

        model = ZeroShotCLIP(self.clip_model, self.subcategories, self.device)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        test_loss, test_acc, _, _, test_f1 = self.evaluate(model, test_loader, criterion, "zero_shot", writer)
        writer.add_scalar("Zero_Shot/Test_Accuracy", test_acc, 0)
        writer.add_scalar("Zero_Shot/Test_F1", test_f1, 0)

        self.metrics["zero_shot"]["test_acc"].append(test_acc)
        self.metrics["zero_shot"]["test_f1"].append(test_f1)

    def run_full_finetuning(self, fold: int, train_subset, val_subset, test_subset, writer: SummaryWriter):
        print("\n[2/2] Full Fine-Tuning")

        train_loader, val_loader, test_loader = self._make_loaders(train_subset, val_subset, test_subset)

        model = FullFineTunedCLIP(self.clip_model, num_classes=len(self.subcategories)).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        # Try logging a graph (can fail depending on ops / custom modules)
        try:
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            writer.add_graph(model, dummy_input)
        except Exception:
            pass

        best_f1 = 0.0
        validations_per_fold = 0

        for epoch in range(Config.NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            accum = 0
            optimizer.zero_grad(set_to_none=True)

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}", leave=False)
            for batch_idx, (images, labels) in enumerate(loop):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / Config.ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                accum += 1

                # Step every ACCUMULATION_STEPS micro-batches
                if accum % Config.ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss.item() * Config.ACCUMULATION_STEPS

                if batch_idx == 0 and epoch == 0:
                    # Show a few denormalized samples in TB
                    writer.add_images("Training_samples",
                                      self._denorm_for_tb(images[:4].detach()),
                                      global_step=0)

            # Flush remainder if not divisible
            if accum % Config.ACCUMULATION_STEPS != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_train_loss = total_loss / max(1, len(train_loader))
            writer.add_scalar("Train/Loss", avg_train_loss, epoch)
            self.metrics["full_ft"]["train_loss"].append(avg_train_loss)

            # Validation
            if (epoch + 1) % Config.VALIDATE_EVERY == 0:
                val_loss, val_acc, _, _, val_f1 = self.evaluate(model, val_loader, criterion, "val", writer, epoch)
                validations_per_fold += 1

                writer.add_scalar("Val/Accuracy", val_acc, epoch)
                writer.add_scalar("Val/F1", val_f1, epoch)
                self.metrics["full_ft"]["val_acc"].append(val_acc)
                self.metrics["full_ft"]["val_f1"].append(val_f1)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save(model.state_dict(), os.path.join(Config.BASE_DIR, f"best_model_fold{fold+1}.pth"))

        # Final test using best checkpoint
        best_path = os.path.join(Config.BASE_DIR, f"best_model_fold{fold+1}.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=self.device))

        test_loss, test_acc, _, _, test_f1 = self.evaluate(model, test_loader, criterion, "test", writer, Config.NUM_EPOCHS)

        # Log a small embedding projector sample
        self.log_embeddings(model, test_loader, writer, epoch=Config.NUM_EPOCHS)

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, criterion, phase: str, writer: Optional[SummaryWriter], epoch: Optional[int] = None):
        model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        avg_loss = total_loss / max(1, len(loader))
        acc = accuracy_score(all_labels, all_preds)
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)

        if writer and epoch is not None:
            writer.add_scalar(f"{phase}/Loss", avg_loss, epoch)
            writer.add_scalar(f"{phase}/Accuracy", acc, epoch)
            writer.add_scalar(f"{phase}/F1", f1, epoch)

        return avg_loss, acc, None, None, f1

    @torch.no_grad()
    def log_embeddings(self, model: nn.Module, loader: DataLoader, writer: SummaryWriter, epoch: int):
        """
        Log a small batch of features + images to TensorBoard Projector.
        """
        features = []
        labels = []
        images = []

        model.eval()
        for batch_idx, (imgs, lbls) in enumerate(loader):
            if batch_idx > 3:  # limit
                break
            imgs = imgs.to(self.device, non_blocking=True)
            feats = model.visual_encoder(imgs) if hasattr(model, "visual_encoder") else self.clip_model.visual(imgs)
            features.append(feats.detach().cpu())
            labels.extend(lbls.cpu().numpy().tolist())
            images.append(self._denorm_for_tb(imgs).cpu())

        if not features:
            return

        features = torch.cat(features, dim=0)
        images = torch.cat(images, dim=0)

        # limit to 100 to keep projector snappy
        features = features[:100]
        images = images[:100]
        labels = labels[:100]
        labels_str = [str(l) for l in labels]

        writer.add_embedding(features, metadata=labels_str, label_img=images, global_step=epoch, tag="Image_Embeddings")

    def _save_results(self):
        # Prepare per-fold best val F1 from the recorded sequence
        # With NUM_EPOCHS=4 and VALIDATE_EVERY=2, each fold contributes 2 val F1s.
        val_f1_all = self.metrics["full_ft"]["val_f1"]
        if len(val_f1_all) == Config.K_FOLDS * (Config.NUM_EPOCHS // Config.VALIDATE_EVERY):
            per_fold_best_val_f1 = [max(chunk) for chunk in np.array_split(val_f1_all, Config.K_FOLDS)]
        else:
            # Fallback in case of early exits; group as evenly as possible
            splits = np.array_split(val_f1_all, Config.K_FOLDS)
            per_fold_best_val_f1 = [max(chunk) if len(chunk) else float('nan') for chunk in splits]

        metrics_df = pd.DataFrame({
            "fold": list(range(1, Config.K_FOLDS + 1)),
            "zero_shot_acc": self.metrics["zero_shot"]["test_acc"],
            "zero_shot_f1": self.metrics["zero_shot"]["test_f1"],
            "full_ft_val_f1": per_fold_best_val_f1,
        })
        metrics_df.to_csv(os.path.join(Config.BASE_DIR, "metrics.csv"), index=False)

        # Plot comparison
        plt.figure(figsize=(10, 5))
        x = list(range(1, Config.K_FOLDS + 1))
        plt.plot(x, self.metrics["zero_shot"]["test_f1"], 'o-', label="Zero-Shot")
        plt.plot(x, per_fold_best_val_f1, 's-', label="Full FT (best Val F1)")
        plt.xlabel("Fold")
        plt.ylabel("F1 Score")
        plt.title("Zero-Shot vs Full Fine-Tuning Performance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.BASE_DIR, "comparison.png"))
        plt.close()


# ------------------------- Main -------------------------
if __name__ == "__main__":
    experiment = CLIPExperiment()
    experiment.run()
