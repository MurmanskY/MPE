"""
downloading alexnet structrue
"""
from torchvision.models import alexnet, AlexNet_Weights

alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
