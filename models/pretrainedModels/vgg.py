'''
for downloading pretrained parameters
'''
from torchvision.models import vgg11, VGG11_Weights
from torchvision.models import vgg13, VGG13_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from torchvision.models import vgg19_bn, VGG19_BN_Weights


vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)