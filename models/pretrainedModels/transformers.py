'''
for downloading pretrained parameters
'''
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import vit_h_14, ViT_H_14_Weights
from torchvision.models import swin_b, Swin_B_Weights
from torchvision.models import swin_v2_b, Swin_V2_B_Weights

vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)  # weights='IMAGENET1K_SWAG_E2E_V1', 518,518
vit_h_14(weights=ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1)  # weights='IMAGENET1K_SWAG_LINEAR_V1' 224,224
vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
