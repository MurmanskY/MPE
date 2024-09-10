"""
downloading
"""
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
print(model)