import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# CONFIG
MODEL_PATH = "mnist_cnn_full.pth"
EPOCHS = 20
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# TRANSFORM
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# DATASET (FULL MNIST)
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Test samples : {len(test_dataset)}")

# MODEL
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = MNIST_CNN().to(device)

# LOSS & OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# TRAIN ONLY ONCE
if not os.path.exists(MODEL_PATH):
    print("\nTraining started...\n")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel trained & saved ✔\n")

else:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("\nModel loaded (No retraining) ✔\n")

# TEST ACCURACY
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

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# BEAUTIFUL OUTPUT – 10 CLASS VISUALIZATION
shown = []
images_list = []
preds_list = []
labels_list = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = torch.argmax(outputs, 1)

        for i in range(len(images)):
            lbl = labels[i].item()
            if lbl not in shown:
                shown.append(lbl)
                images_list.append(images[i])
                preds_list.append(preds[i].item())
                labels_list.append(lbl)

            if len(shown) == 10:
                break
        if len(shown) == 10:
            break

plt.figure(figsize=(15, 4))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(images_list[i].squeeze(), cmap="gray")
    plt.title(f"P:{preds_list[i]}\nT:{labels_list[i]}")
    plt.axis("off")

plt.suptitle("MNIST – 10 Class Predictions (Full Dataset)", fontsize=16)
plt.show()
