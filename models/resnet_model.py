import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_model(num_classes):
    model = models.resnet18(pretrained=True)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Modify final fully connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model
