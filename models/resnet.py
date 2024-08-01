import torch
from torch import nn
from torch.nn import functional as F


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

model.eval()