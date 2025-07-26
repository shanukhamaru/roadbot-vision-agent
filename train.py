import torch
from torchvision import datasets, transforms
from models.resnet_model import get_resnet18_model
import torch.optim as optim

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load Dataset using ImageFolder
train_data = datasets.ImageFolder(root='data/raw/Train', transform=transform)

# Create DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# Print class labels mapping
print(train_data.class_to_idx)



# Get number of classes
num_classes = len(train_data.classes)

# Load ResNet18 model
model = get_resnet18_model(num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training Loop
for epoch in range(5):  # Start with 5 epochs
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'models/roadbot_resnet18.pth')

# Save class labels mapping
import json
with open('labels.json', 'w') as f:
    json.dump(train_data.class_to_idx, f)
