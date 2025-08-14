# %% [markdown]
## Image Classification with CLIP: Dataset Loading, Fine-Tuning, and Evaluation

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
### 2. List Available Pretrained CLIP Models
def list_pretrained_clip_models():
    """List all available pretrained CLIP models"""
    clip_models = open_clip.list_pretrained()  # Get the list of available models
    print("\nAvailable pretrained CLIP models:")
    for i, model in enumerate(clip_models):
        print(f"{i + 1}. {model}")
    print(f"Number of available pre-trained CLIP models: {len(clip_models)}")

# Call the function to list pretrained CLIP models
list_pretrained_clip_models()

# %% [markdown]
### 3. Load and Explore Dataset

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

        # Visualize sample
        plt.figure(figsize=(5, 5))
        plt.imshow(sample['image'])
        plt.title(f"Sample: {sample['subCategory']}")
        plt.axis('off')
        plt.show()

        # Get unique subcategories
        subcategories = sorted(list(set(dataset['subCategory'])))
        print(f"\nFound {len(subcategories)} subcategories")
        print("First 5:", subcategories[:5])

        return dataset, subcategories

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

# Load the dataset and extract subcategories
dataset, subcategories = load_fashion_dataset()

# %% [markdown]
### 4. CLIP Model Setup with User Input

def initialize_clip_model(device, model_name="ViT-B-32", pretrained_weights="laion2b_s34b_b79k"):
    """Initialize CLIP model with preprocessing and tokenizer"""
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_weights,
            device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        print(f"Loaded model: {model_name} with weights: {pretrained_weights}")
        return model, preprocess, tokenizer
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise

# Allow user to choose the model and weights
print("\nEnter model name (e.g., ViT-B-32, ViT-L-14, RN50):")
chosen_model = input("Model name: ") or "ViT-B-32"
print("Enter pretrained weights (e.g., laion2b_s34b_b79k, laion400m_e32):")
chosen_weights = input("Pretrained weights: ") or "laion2b_s34b_b79k"

# Initialize CLIP model
model, preprocess, tokenizer = initialize_clip_model(device, chosen_model, chosen_weights)

# %% [markdown]
### 5. Zero-Shot Evaluation

def evaluate_zero_shot(model, tokenizer, dataset, subcategories, indices=None):
    """Evaluate CLIP's zero-shot performance"""
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
        axes[i].set_title(f'True: {true_label}\nPredicted: {predicted_label}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Evaluate zero-shot performance
evaluate_zero_shot(model, tokenizer, dataset, subcategories)

# %% [markdown]
### 6. Dataset Preparation for Fine-Tuning

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

# Prepare dataloaders
train_loader, val_loader, test_loader = prepare_dataloaders(dataset, subcategories)

# %% [markdown]
### 7. Fine-Tuning CLIP Model

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
            torch.save(model.state_dict(), 'best_clip_model.pth')

    return model

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

            # Collect predictions and labels for metrics calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate precision, recall, and F1-score
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    return 100 * correct / total

# Fine-tune and evaluate model
model_ft = CLIPFineTuner(model, len(subcategories)).to(device)
model_ft = train_model(model_ft, train_loader, val_loader)

# Evaluate on test set
model_ft.load_state_dict(torch.load('best_clip_model.pth'))
test_acc = evaluate_model(model_ft, test_loader)
print(f'\nTest Accuracy: {test_acc:.2f}%')

# %% [markdown]
### 8. Visualize Predictions

def visualize_predictions(model, dataset, subcategories, indices=None):
    """Visualize model predictions"""
    indices = indices or torch.randint(0, len(dataset), (3,)).tolist()  # Randomly select indices if none are provided
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
    plt.show()

# Visualize predictions
visualize_predictions(model_ft, dataset, subcategories)