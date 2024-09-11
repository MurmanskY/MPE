'''
for downloading pretrained parameters
'''
from torchvision.models import vit_b_16, ViT_B_16_Weights

vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
