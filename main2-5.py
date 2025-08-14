# Install the necessary packages (run this in your terminal, not in your script)
# pip install open_clip_torch datasets torchvision matplotlib seaborn torchmetrics

import open_clip
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


def setup_environment():
    """Initialize device and verify environment for AMD GPU with ROCm."""
    if torch.cuda.is_available():
        device = torch.device("cuda")  # ROCm uses CUDA interface
        print(f"Using device: {device} (AMD GPU with ROCm support)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU. No compatible GPU found.")

    print(f"PyTorch version: {torch.__version__}")
    return device


# Step 1: Setup environment
device = setup_environment()

# Step 2: Load the Dataset
dataset = load_dataset('ceyda/fashion-products-small')

# Inspect the dataset structure
print(dataset)
print(dataset['train'][0])  # Print the first entry to see its structure


# Function to visualize sample images
def show_sample_images(dataset):
    sample_indices = [0, 1, 2, 3, 4]
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i + 1)
        plt.imshow(dataset['train'][idx]['image'])  # Display the image
        plt.title(dataset['train'][idx]['subCategory'])  # Use 'subCategory' or 'masterCategory' as the label
        plt.axis('off')
    plt.show()


# Visualize sample images
show_sample_images(dataset)

# Step 3: Explore Pre-Trained Models
# Instantiate the ViT-B-32 model with pretrained='openai'
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
print(model)
print(preprocess)

# Freeze all parameters except the final classifier layer
for param in model.parameters():
    param.requires_grad = False


# Define QuickGELU activation
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

class CustomClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1):
        super(CustomClassifier, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(QuickGELU())  # Use custom QuickGELU activation
        layers.append(nn.Linear(input_dim, output_dim))  # Final output layer
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


# Define the number of output classes based on your dataset
num_classes = len(set(dataset['train'][i]['subCategory'] for i in range(len(dataset['train']))))

# Step 5: Dataset Preparation
# Split the dataset into train and test sets using the Hugging Face method
train_val = dataset['train'].train_test_split(test_size=0.2, seed=42)  # 20% for testing
train = train_val['train']
test = train_val['test']

# Further split the training data into training and validation sets
train, val = train.train_test_split(test_size=0.2, seed=42).values()  # 20% of the train for validation

# Fit the label encoder on training labels
label_encoder = LabelEncoder()
label_encoder.fit([item['subCategory'] for item in train])  # Use your label key here


# Custom PyTorch Dataset class
class FashionDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['subCategory']  # Use 'subCategory' or 'masterCategory' as the label
        if self.transform:
            image = self.transform(image)
        # Convert label to numerical format using label encoder
        label = label_encoder.transform([label])[0]  # Transform and get the first element
        return image, torch.tensor(label)  # Return as tensor


# Define transformations
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Create datasets
train_dataset = FashionDataset(train, transform)
val_dataset = FashionDataset(val, transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Step 6: New Model Definition with Custom Classifier
class CLIPWithCustomClassifier(nn.Module):
    def __init__(self, clip_model, custom_classifier):
        super(CLIPWithCustomClassifier, self).__init__()
        self.clip_model = clip_model
        self.custom_classifier = custom_classifier

    def forward(self, images):
        # Forward pass through the CLIP model
        features = self.clip_model.encode_image(images)
        # Forward pass through the custom classifier
        outputs = self.custom_classifier(features)
        return outputs


# Step 7: Training and Validation Function
def train_and_validate(num_layers, num_epochs=3):
    # Instantiate the model
    clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

    # Freeze parameters
    for param in clip_model.parameters():
        param.requires_grad = False

    # Create a new custom classifier
    custom_classifier = CustomClassifier(input_dim=clip_model.visual.proj.shape[1], output_dim=num_classes,
                                         num_layers=num_layers)

    # Create a new model that combines CLIP and the custom classifier
    model = CLIPWithCustomClassifier(clip_model, custom_classifier)

    # Move model to device (check if AMD GPU is available)
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(custom_classifier.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # Ensure both are tensors
            optimizer.zero_grad()
            outputs = model(images)  # This will use the custom classifier
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    return train_accuracy


# Store results for each configuration
results = {}

# Train models with different numbers of layers (1, 2, and 3)
for layers in range(1, 4):
    accuracy = train_and_validate(num_layers=layers, num_epochs=3)
    results[layers] = accuracy * 100  # Convert to percentage
    print(f'Model with {layers} layer(s): Train Accuracy: {results[layers]:.2f}%')  # Display as percentage

# Plotting the results
plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'orange', 'green'])
plt.xlabel('Number of Linear Layers in Classifier')
plt.ylabel('Training Accuracy (%)')
plt.title('Comparison of Classifier Performance with Different Number of Linear Layers')
plt.xticks(list(results.keys()))
plt.ylim(0, 100)  # Set y-limit to 100 for percentage
plt.show()