
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# 1. DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
MODEL_PATH = "fashion_mnist_cnn.pth"

# 2. CLASS NAMES (10 CLASSES)
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

NUM_CLASSES = 10

# 3. DATA TRANSFORMS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 4. LOAD FULL DATASET & TAKE FIRST 8000 SAMPLES
full_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Take first 8000 samples
train_indices = list(range(6000))      
test_indices = list(range(6000, 8000))  
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Training samples:", len(train_dataset))
print("Testing samples :", len(test_dataset))

# 6. CNN MODEL (HIGH ACCURACY)
class StrongCNN(nn.Module):
    def __init__(self):
        super(StrongCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = StrongCNN().to(device)

# 7. LOSS FUNCTION & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8. TRAIN MODEL (EPOCH = 20)
num_epochs = 20

if os.path.exists(MODEL_PATH):
    print("\nSaved model found. Loading model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    print("\nNo saved model found. Training model...\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {running_loss/len(train_loader):.4f} "
              f"Train Accuracy: {train_accuracy:.2f}%")

    # 💾 Save trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n💾 Model saved as '{MODEL_PATH}'")


# 9. OVERALL TEST ACCURACY
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

overall_accuracy = 100 * correct / total

print("\n===================================")
print(f"OVERALL TEST ACCURACY: {overall_accuracy:.2f}%")
print("===================================\n")

# DISPLAY PREDICTION RESULTS

# 10. DISPLAY RANDOM 5 TEST PREDICTIONS (AFTER TRAINING)
import random

model.eval()

# Get random indices from test dataset
random_indices = random.sample(range(len(test_dataset)), 5)

plt.figure(figsize=(10,4))

for i, idx in enumerate(random_indices):
    image, label = test_dataset[idx]
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    plt.subplot(1,5,i+1)
    plt.imshow(image.cpu().squeeze(), cmap='gray')
    plt.title(f"P: {class_names[predicted.item()]}\nA: {class_names[label]}")
    plt.axis('off')

plt.suptitle(f"Random Test Predictions (Accuracy: {overall_accuracy:.2f}%)")
plt.show()