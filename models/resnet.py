import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)