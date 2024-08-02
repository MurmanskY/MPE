'''
for downloading pretrained parameters
'''
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights


resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

