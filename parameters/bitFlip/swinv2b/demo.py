import torch
import torchvision

model = torchvision.models.swin_v2_b()
model.load_state_dict(torch.load('../../bitFlip/swinv2b/bitFlip/frac_1.pth'))