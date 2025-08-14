# %% [markdown]
## Image Classification with ConvNeXt (CLIPFineTuner): Dataset Splitting, Loading, Fine-Tuning, and Evaluation

# %%
# Import required libraries
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
from sklearn.metrics import precision_score, recall_score, f1_score

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Optional: Create log directory for saving figures and metrics
LOG_DIR = "logs/main1/"
os.makedirs(LOG_DIR, exist_ok=True)

# %% [markdown]
### 1. Initialize Environment

def setup_environment():
    """Initialize device and verify environment"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    return device

device = setup_environment()

# %% [markdown]
### 2. Load ConvNeXt Model and Print Structure

def load_and_print_convnext_model(device, model_name="convnext_base_w", pretrained_weights="laion2b_s13b_b82k_augreg"):
    """Load ConvNeXt model and print its structure and preprocessing transforms"""
    try:
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_weights,
            device=device
        )
        print("\nModel structure:")
        print(model)
        print("\nPreprocessing transforms (for training):")
        print(preprocess_train)
        print("\nPreprocessing transforms (for validation):")
        print(preprocess_val)
        return model, preprocess_train, preprocess_val
    except Exception as e:
        print(f"Error loading ConvNeXt model: {e}")
        raise

# Load the ConvNeXt model
convnext_model, preprocess_train, preprocess_val = load_and_print_convnext_model(device)

# %% [markdown]
### 3. Load and Explore Dataset

def load_fashion_dataset():
    """Load fashion dataset and visualize samples"""
    try:
        ds = load_dataset("ceyda/fashion-products-small")
        dataset = ds['train']

        # Display a sample
        sample = dataset[0]
        print("\nSample data structure:")
        for k, v in sample.items():
            if k != 'image':
                print(f"{k}: {v}")

        # Visualize the sample and save/show
        plt.figure(figsize=(5, 5))
        plt.imshow(sample['image'])
        plt.title(f"Sample: {sample['subCategory']}")
        plt.axis('off')
        plt.savefig(os.path.join(LOG_DIR, "sample_data.png"))
        plt.show()
        plt.close()

        # Extract unique subcategories
        subcategories = sorted(list(set(dataset['subCategory'])))
        print(f"\nFound {len(subcategories)} subcategories")
        print("First 5:", subcategories[:5])

        return dataset, subcategories
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

dataset, subcategories = load_fashion_dataset()

# %% [markdown]
### 4. Dataset Splitting and DataLoader Preparation

class FashionDataset(Dataset):
    """Custom dataset for fashion products with preprocessing transforms"""

    def __init__(self, data, subcategories, transform=None):
        self.data = data
        self.subcategories = subcategories
        self.transform = transform or preprocess_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item['image'])
        label = self.subcategories.index(item['subCategory'])
        return image, label

def prepare_dataloaders(dataset, subcategories, batch_size=32, test_ratio=0.15):
    """
    Split dataset into train, validation, and test sets, and create DataLoaders
    """
    val_size = int(test_ratio * len(dataset))
    test_size = val_size
    train_size = len(dataset) - val_size - test_size

    train_data, val_data, test_data = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataset = FashionDataset(train_data, subcategories)
    val_dataset = FashionDataset(val_data, subcategories)
    test_dataset = FashionDataset(test_data, subcategories)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = prepare_dataloaders(dataset, subcategories)

# %% [markdown]
### 5. Fine-Tuning ConvNeXt Model (renamed as CLIPFineTuner)

class CLIPFineTuner(nn.Module):
    """Fine-tuning wrapper for ConvNeXt model"""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.convnext = base_model
        for param in self.convnext.parameters():  # Freeze ConvNeXt parameters
            param.requires_grad = False
        self.classifier = nn.Linear(640, num_classes)  # Adjust input size for classifier

    def forward(self, images):
        features = self.convnext.encode_image(images)
        return self.classifier(features)

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4):
    """Training loop with validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0.0

    for epoch in range(num_epochs):
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

        # Validation
        val_acc = evaluate_model(model, val_loader)
        print(f'Validation Accuracy: {val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(LOG_DIR, 'best_convnext_model.pth'))

    return model, best_acc

def evaluate_model(model, loader):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    return 100 * correct / total

model_ft = CLIPFineTuner(convnext_model, len(subcategories)).to(device)
model_ft, best_val_acc = train_model(model_ft, train_loader, val_loader)

# Evaluate on the test set
model_ft.load_state_dict(torch.load(os.path.join(LOG_DIR, 'best_convnext_model.pth')))
test_acc = evaluate_model(model_ft, test_loader)
print(f'\nTest Accuracy: {test_acc:.2f}%')

# Save metrics to log directory
with open(os.path.join(LOG_DIR, "convnext_metrics.txt"), "w") as f:
    f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
    f.write(f"Test Accuracy: {test_acc:.2f}%\n")

# %% [markdown]
### 6. Visualize Predictions

def visualize_predictions(model, dataset, subcategories, preprocess, indices=None):
    """Visualize model predictions and save/show plot"""
    indices = indices or torch.randint(0, len(dataset), (3,)).tolist()
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
        axes[i].set_title(f'True: {true_label}\nPredicted: {predicted_label}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "convnext_predictions.png"))
    plt.show()
    plt.close()

visualize_predictions(model_ft, dataset, subcategories, preprocess_train)