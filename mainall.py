# --- 1. Install and Import Dependencies ---

# Install if running in a fresh Colab environment
# !pip install open_clip_torch datasets torch torchvision tqdm matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from datasets import load_dataset
import open_clip
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# --- 2. Load and Explore the Dataset ---

ds = load_dataset('ceyda/fashion-products-small')
dataset = ds['train']

# Get all unique subcategories
subcategories = sorted(list(set([item['subCategory'] for item in dataset])))
subcategory_to_idx = {sc: idx for idx, sc in enumerate(subcategories)}
idx_to_subcategory = {idx: sc for sc, idx in subcategory_to_idx.items()}

# --- 3. Split Data into Train/Val/Test ---

total_len = len(dataset)
test_len = int(0.15 * total_len)
val_len = int(0.15 * total_len)
train_len = total_len - val_len - test_len

train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42))

# --- 4. Load CLIP Model and Preprocessing ---

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)

# --- 5. Custom Dataset Class ---

class FashionDataset(Dataset):
    def __init__(self, data, subcategory_to_idx, preprocess):
        self.data = data
        self.subcategory_to_idx = subcategory_to_idx
        self.transform = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        subcategory = item['subCategory']
        label = self.subcategory_to_idx[subcategory]
        return self.transform(image), label

# --- 6. DataLoaders ---

batch_size = 32
train_loader = DataLoader(FashionDataset(train_data, subcategory_to_idx, preprocess), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(FashionDataset(val_data, subcategory_to_idx, preprocess), batch_size=batch_size)
test_loader = DataLoader(FashionDataset(test_data, subcategory_to_idx, preprocess), batch_size=batch_size)

# --- 7. Model Modification for Classification ---

class CLIPFineTuner(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():  # Freeze the CLIP visual backbone
            features = self.clip_model.encode_image(x)
        return self.classifier(features)

num_classes = len(subcategories)
model_ft = CLIPFineTuner(model, num_classes).to(device)

# --- 8. Loss Function & Optimizer ---

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

# --- 9. Training and Validation Loop ---

num_epochs = 5
for epoch in range(num_epochs):
    model_ft.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Training loss: {epoch_loss:.4f}")

    # Validation
    model_ft.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total
    print(f"Validation accuracy: {val_acc:.4f}")

# --- 10. Save the Fine-Tuned Model ---

torch.save(model_ft.state_dict(), 'clip_finetuned.pth')

# --- 11. Evaluate on Test Set ---

model_ft.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_ft(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")

# --- 12. Visualize Predictions on Random Samples ---

import random

indices = random.sample(range(len(test_data)), 3)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, idx in enumerate(indices):
    item = test_data[idx]
    image = item['image']
    true_label = item['subCategory']
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_ft(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        predicted_label = idx_to_subcategory[pred_idx]

    axes[i].imshow(image)
    axes[i].set_title(f"True: {true_label}\nPred: {predicted_label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()