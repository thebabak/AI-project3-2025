# %% [markdown]
## Image Classification with ConvNeXt: Full Workflow with Manual Weight Loading, GPU Utilization, Batch Visualization, Confusion Matrix, and Cosine Similarity

# %%
# Import required libraries
import os
import torch
import open_clip
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
from collections import Counter
import numpy as np

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'


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
### 2. Load ConvNeXt Model with Manual Weight Loading

def load_and_print_convnext_model(device, model_name="convnext_base_w", pretrained_weights=None,
                                  local_weights_path=None):
    """
    Load ConvNeXt model and print its structure and preprocessing transforms.
    Allows manual loading of pretrained weights if `local_weights_path` is provided.
    """
    try:
        if local_weights_path:
            print(f"Loading model weights from local path: {local_weights_path}")
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=None,  # Skip downloading
                device=device
            )
            model.load_state_dict(torch.load(local_weights_path, map_location=device))
        else:
            model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained_weights,
                device=device
            )

        # Print the model structure and preprocessing transforms
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


# Load the ConvNeXt model, use local weights if automatic download fails
convnext_model, preprocess_train, preprocess_val = load_and_print_convnext_model(
    device,
    model_name="convnext_base_w",
    pretrained_weights="laion2b_s13b_b82k_augreg",
    local_weights_path=None  # Set to a valid path if you want to load manually
)


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

        # Visualize the sample
        plt.figure(figsize=(5, 5))
        plt.imshow(sample['image'])
        plt.title(f"Sample: {sample['subCategory']}")
        plt.axis('off')
        plt.show()

        # Extract unique subcategories
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
### 4. Inspect Class Distribution

def inspect_class_distribution(dataset):
    """Inspect the distribution of classes in the dataset"""
    labels = [item['subCategory'] for item in dataset]
    class_counts = Counter(labels)
    print("Class distribution:")
    for label, count in class_counts.items():
        print(f"{label}: {count}")


# Inspect class distribution
inspect_class_distribution(dataset)


# %% [markdown]
### 5. Compute Class Weights

def compute_class_weights(dataset, subcategories):
    """Compute class weights based on the dataset distribution"""
    labels = [item['subCategory'] for item in dataset]
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / class_counts[subcategory] for subcategory in subcategories]
    return torch.tensor(class_weights, dtype=torch.float).to(device)


class_weights = compute_class_weights(dataset, subcategories)


# %% [markdown]
### 6. Custom PyTorch Dataset with Augmentations

class FashionDataset(Dataset):
    """Custom dataset for fashion products with preprocessing and augmentations"""

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

        # Augmentations for training (if enabled)
        self.augment = augment
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert index to Python int for compatibility with Hugging Face Dataset
        idx = int(idx)
        item = self.data[idx]
        image = item['image']

        # Apply augmentations if enabled
        if self.augment:
            image = self.augmentation_transforms(image)

        # Apply preprocessing
        image = self.transform(image)
        label = self.subcategories.index(item['subCategory'])
        return image, label


# %% [markdown]
### 7. Fine-Tuning ConvNeXt Model

class ConvNeXtFineTuner(nn.Module):
    """Fine-tuning wrapper for ConvNeXt model"""

    def __init__(self, base_model, num_classes):
        super().__init__()
        self.convnext = base_model
        for param in self.convnext.parameters():  # Freeze ConvNeXt parameters
            param.requires_grad = False
        self.classifier = nn.Linear(640, num_classes)  # Adjust input size to match ConvNeXt output

    def forward(self, images):
        features = self.convnext.encode_image(images)
        return self.classifier(features)


# %% [markdown]
### 8. Compute Cosine Similarity Between Images and Descriptions

def compute_cosine_similarity(model, preprocess, images, descriptions, device):
    """
    Compute cosine similarity between images and descriptions using CLIP embeddings.
    Args:
        model: CLIP model instance.
        preprocess: Preprocessing function for images.
        images: List of PIL images.
        descriptions: List of textual descriptions.
        device: Device (CPU/GPU) to run computations on.
    Returns:
        similarities: Cosine similarity scores between each image and description.
    """
    model.eval()  # Set model to evaluation mode

    # Preprocess images and move to device
    preprocessed_images = torch.stack([preprocess(image) for image in images]).to(device)

    # Encode images
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

    # Encode descriptions
    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize(descriptions).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

    # Compute cosine similarity
    similarities = (image_features @ text_features.T).cpu().numpy()  # Matrix multiplication for similarity
    return similarities


# %% [markdown]
### 9. Select Images and Compute Cosine Similarity

def select_images_and_compute_similarity(dataset, subcategories, num_samples=8):
    """
    Select random images from the dataset, generate descriptions, and compute cosine similarity.
    Args:
        dataset: The dataset containing images and labels.
        subcategories: List of subcategory names corresponding to labels.
        num_samples: Number of images to select and describe (default is 8).
    """
    # Randomly select images
    selected_indices = [int(i) for i in np.random.choice(len(dataset), num_samples, replace=False)]
    selected_samples = [dataset[i] for i in selected_indices]
    images = [sample['image'] for sample in selected_samples]

    # Generate descriptions based on subcategories
    descriptions = [f"This is a {subcategories.index(sample['subCategory'])}, suitable for various occasions."
                    for sample in selected_samples]

    # Compute cosine similarity
    similarities = compute_cosine_similarity(convnext_model, preprocess_val, images, descriptions, device)

    # Display images, descriptions, and similarity scores
    for i, (image, description) in enumerate(zip(images, descriptions)):
        similarity_score = similarities[i][i]  # Extract diagonal value for image-text pair
        print(f"Image {i + 1}:")
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Description: {description}\nSimilarity: {similarity_score:.2f}")
        plt.show()

select_images_and_compute_similarity(dataset, subcategories, num_samples=8)

# %% [markdown]
### 10. K-Fold Cross-Validation

def k_fold_cross_validation(model, dataset, subcategories, k_folds=5, batch_size=32, num_epochs=3, lr=1e-4):
    """Perform k-fold cross-validation on the dataset"""
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")

        # Create train and validation subsets
        train_subset = Subset(dataset, [int(i) for i in train_idx])  # Convert indices to Python int
        val_subset = Subset(dataset, [int(i) for i in val_idx])      # Convert indices to Python int

        # Wrap subsets in FashionDataset
        train_dataset = FashionDataset(train_subset, subcategories, augment=True)
        val_dataset = FashionDataset(val_subset, subcategories)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        model_ft = ConvNeXtFineTuner(model, len(subcategories)).to(device)

        # Train the model
        model_ft = train_model(model_ft, train_loader, val_loader, num_epochs=num_epochs, lr=lr)

        # Evaluate the model
        val_acc = evaluate_model(model_ft, val_loader, subcategories)
        results[fold] = val_acc

        print(f"Fold {fold + 1} Validation Accuracy: {val_acc:.2f}%")

    # Print overall results
    avg_acc = sum(results.values()) / k_folds
    print(f"\nAverage Validation Accuracy across {k_folds} folds: {avg_acc:.2f}%")
    return results


# %% [markdown]
### 11. Training Loop and Evaluation

def train_model(model, train_loader, val_loader, num_epochs=3, lr=1e-4):
    """Training loop with validation"""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        val_acc = evaluate_model(model, val_loader, subcategories)
        print(f'Validation Accuracy: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_convnext_model.pth')

    return model


def evaluate_model(model, loader, subcategories):
    """Evaluate model performance and visualize the confusion matrix"""
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

    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

    # Confusion matrix visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subcategories, yticklabels=subcategories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    return 100 * correct / total


# %% [markdown]
### 12. Main Workflow

# Perform k-fold cross-validation
results = k_fold_cross_validation(convnext_model, dataset, subcategories, k_folds=5, batch_size=32, num_epochs=3, lr=1e-4)

# Compute cosine similarity between selected images and descriptions
select_images_and_compute_similarity(dataset, subcategories, num_samples=8)