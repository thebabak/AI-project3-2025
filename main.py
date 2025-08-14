# %% [markdown]
## Image Classification with CLIP: Dataset Loading, Fine-Tuning, and Evaluation

# %%
import os
import torch
import open_clip
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# Suppress warnings and set log directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

LOG_DIR = "logs/main1/"
os.makedirs(LOG_DIR, exist_ok=True)

# %% [markdown]
### 1. Initialize and Verify Environment

def setup_environment():
    """Initialize device and verify requirements"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device

device = setup_environment()

# %% [markdown]
### 2. Load and Explore Dataset

def load_fashion_dataset():
    """Load fashion dataset and visualize samples"""
    try:
        ds = load_dataset("ceyda/fashion-products-small")
        dataset = ds['train']

        # Display sample
        sample = dataset[0]
        print("\nSample data structure:")
        for k, v in sample.items():
            if k != 'image':
                print(f"{k}: {v}")

        # Visualize sample and save to log directory
        plt.figure(figsize=(5, 5))
        plt.imshow(sample['image'])
        plt.title(f"Sample: {sample['subCategory']}")
        plt.axis('off')
        plt.savefig(os.path.join(LOG_DIR, "sample_data.png"))
        plt.show()
        plt.close()

        # Get unique subcategories
        subcategories = sorted(list(set(dataset['subCategory'])))
        print(f"\nFound {len(subcategories)} subcategories")
        print("First 5:", subcategories[:5])

        return dataset, subcategories

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

dataset, subcategories = load_fashion_dataset()

# %% [markdown]
### 3. CLIP Model Setup and Zero-Shot Evaluation

def initialize_clip_model(device):
    """Initialize CLIP model with preprocessing and tokenizer"""
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=device
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise

def evaluate_zero_shot(model, tokenizer, dataset, subcategories, preprocess, indices=None):
    """Evaluate CLIP's zero-shot performance and save plot"""
    indices = indices or [0, 100, 200]
    text_inputs = torch.cat([tokenizer(f"a photo of {c}") for c in subcategories]).to(device)

    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        entry = dataset[idx]
        image = entry['image']
        true_label = entry['subCategory']

        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Normalize and calculate similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, pred_idx = similarity[0].topk(1)
        predicted_label = subcategories[pred_idx[0]]

        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "zero_shot_predictions.png"))
    plt.show()
    plt.close()

model, preprocess, tokenizer = initialize_clip_model(device)
evaluate_zero_shot(model, tokenizer, dataset, subcategories, preprocess)

# %% [markdown]
### 4. Dataset Preparation for Fine-Tuning

class FashionDataset(Dataset):
    """Custom dataset for fashion products"""

    def __init__(self, data, subcategories, transform=None):
        self.data = data
        self.subcategories = subcategories
        self.transform = transform or transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            self.transform(item['image']),
            self.subcategories.index(item['subCategory'])
        )

def prepare_dataloaders(dataset, subcategories, batch_size=32, test_ratio=0.15):
    """Prepare train/val/test dataloaders"""
    val_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - 2 * val_size
    train_data, val_data, test_data = random_split(
        dataset,
        [train_size, val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = FashionDataset(train_data, subcategories)
    val_dataset = FashionDataset(val_data, subcategories)
    test_dataset = FashionDataset(test_data, subcategories)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = prepare_dataloaders(dataset, subcategories)

# %% [markdown]
### 5. Fine-Tuning CLIP Model

class CLIPFineTuner(nn.Module):
    """Fine-tuning wrapper for CLIP model"""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.clip = base_model
        for param in self.clip.parameters():  # Freeze CLIP parameters
            param.requires_grad = False
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images):
        features = self.clip.encode_image(images)
        return self.classifier(features)

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    """Training loop with validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': train_loss / (progress_bar.n + 1)})

        # Validation phase
        val_acc = evaluate_model(model, val_loader)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'best_clip_model.pth'))

    return model, best_acc

def evaluate_model(model, loader):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

model_ft = CLIPFineTuner(model, len(subcategories)).to(device)
model_ft, best_val_acc = train_model(model_ft, train_loader, val_loader)

# Evaluate on test set
model_ft.load_state_dict(torch.load(os.path.join(LOG_DIR, 'best_clip_model.pth')))
test_acc = evaluate_model(model_ft, test_loader)
print(f'\nTest Accuracy: {test_acc:.2f}%')

# Save metrics to log directory
with open(os.path.join(LOG_DIR, "metrics.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")

# %% [markdown]
### 6. Visualize Predictions

def visualize_predictions(model, dataset, subcategories, preprocess, indices=None):
    """Visualize model predictions and save plot"""
    indices = indices or [0, 100, 200]
    model.eval()
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 5))
    if len(indices) == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        entry = dataset[idx]
        image = entry['image']
        true_label = entry['subCategory']

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = subcategories[predicted[0]]

        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "finetuned_predictions.png"))
    plt.show()
    plt.close()

visualize_predictions(model_ft, dataset, subcategories, preprocess)