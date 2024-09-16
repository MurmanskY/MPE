from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import densenet201, DenseNet201_Weights

model1 = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)




model2 = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
