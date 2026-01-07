import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#  SETTINGS 
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
MODEL_PATH = os.path.join(BASE, "drone_bird_cnn.pth")

IMG_SIZE = 24
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

#  DATASET 
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transforms)
classes = full_dataset.classes
print("Classes:", classes)

train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

test_dataset.dataset.transform = test_transforms

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# MODEL 
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = CNNModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#  TRAIN 
if os.path.exists(MODEL_PATH):
    print("\nSaved model found. Loading model...\n")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
else:
    print("\nTraining Started\n")

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Epoch", epoch + 1,
            "Loss", round(loss_sum / len(train_loader), 4),
            "Accuracy", round(100 * correct / total, 2), "%"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel trained and saved successfully")



# TEST 
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("\nTest Accuracy:", round(100 * correct / total, 2), "%")

# DISPLAY TEST IMAGES 
NUM_IMAGES = 5

rows = 2
cols = 3
fig, axes = plt.subplots(rows, cols, figsize=(8,4))
axes = axes.flatten()

indices = random.sample(range(len(test_dataset)), NUM_IMAGES)

print("\nSample Predictions\n")

for i, idx in enumerate(indices):
    img_tensor, _ = test_dataset[idx]
    img_input = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_input)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probabilities, 1)

    img_path = test_dataset.dataset.samples[test_dataset.indices[idx]][0]
    image = Image.open(img_path)

    label = classes[prediction.item()]
    confidence_value = confidence.item() * 100

    print(
        "Image", i + 1,
        "Predicted", label.upper(),
        "Confidence", round(confidence_value, 2), "%"
    )

    color = "green" if label.lower() == "birds" else "red"

    axes[i].imshow(image)
    axes[i].set_title(label.upper(), fontsize=14, fontweight="bold", color=color)
    axes[i].axis("off")

    rect = patches.Rectangle(
        (0, 0),
        image.size[0],
        image.size[1],
        linewidth=3,
        edgecolor=color,
        facecolor="none"
    )
    axes[i].add_patch(rect)

# Hide unused subplot
for j in range(NUM_IMAGES, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()