import torch
import numpy as np
from bitstring import BitArray


path1 = "../parameters/init/resnet50-11ad3fa6.pth"

path2 = "../parameters/retrained_ImageNet2PCAM/resnet50re_10.pth"
pathT1 = "../parameters/expReplacement/resnet50PCAM2Init1.pth"
pathT2 = "../parameters/expReplacement/resnet50PCAM2Init2.pth"
pathT3 = "../parameters/expReplacement/resnet50PCAM2Init3.pth"
pathT4 = "../parameters/expReplacement/resnet50PCAM2Init4.pth"
pathT5 = "../parameters/expReplacement/resnet50PCAM2Init5.pth"

path3 = "../parameters/retrained_ImageNet2FGVCAircraft/resnet50re_20.pth"
pathX1 = "../parameters/expReplacement/resnet50FGVCAircraft2Init1.pth"
pathX2 = "../parameters/expReplacement/resnet50FGVCAircraft2Init2.pth"


para1 = torch.load(path1)
para2 = torch.load(path3, map_location=torch.device("mps"))
para1["conv1.weight"].data = para2["conv1.weight"].data
para1["layer1.0.conv1.weight"].data = para2["layer1.0.conv1.weight"].data
# para1["layer1.0.conv2.weight"].data = para2["layer1.0.conv2.weight"].data
# para1["layer1.0.conv3.weight"].data = para2["layer1.0.conv3.weight"].data

torch.save(para1, pathX2)

