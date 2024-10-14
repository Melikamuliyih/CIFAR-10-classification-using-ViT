!pip install timm
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from timm import create_model  # For Vision Transformer in Task 2

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define batch size and data transformations
batch_size = 128

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Function to display images
def imshow(img, lab):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(' '.join(lab))
    plt.show()

# Visualizing some images from the training set
dataiter = iter(trainloader)
images, labels = next(dataiter)
labs = ['%5s' % classes[labels[j]] for j in range(batch_size)]
imshow(torchvision.utils.make_grid(images), labs)


# Task 2: Vision Transformer (ViT) Model for CIFAR-10 Classification
# Download and set up Vision Transformer model from timm
vit_model = create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
vit_model = vit_model.to(device)

# Define optimizer and criterion for Vision Transformer
vit_optimizer = torch.optim.Adam(vit_model.parameters(), lr=0.0001)
vit_criterion = nn.CrossEntropyLoss()

# Training the Vision Transformer model
vit_loss_history = []
for epoch in range(10):  # Train for 10 epochs
    vit_model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        vit_optimizer.zero_grad()

        outputs = vit_model(inputs)
        loss = vit_criterion(outputs, labels)
        loss.backward()
        vit_optimizer.step()

        running_loss += loss.item()

    vit_loss_history.append(running_loss / len(trainloader))
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}')

print('Finished Vision Transformer Training')

# Plot the training loss for Vision Transformer
plt.plot(range(1, len(vit_loss_history) + 1), vit_loss_history, 'b.-')
plt.title('Training Loss per Epoch (ViT)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Testing the Vision Transformer model
correct = 0
total = 0
with torch.no_grad():
    vit_model.eval()
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = vit_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of Vision Transformer on 10000 test images: {100 * correct / total:.2f}%')
