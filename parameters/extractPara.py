"""code cache"""
import torch


resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'

para_resnet18 = torch.load(resnet18InitParaPath)
para_resnet50 = torch.load(resnet50InitParaPath)
para_resnet101 = torch.load(resnet101InitParaPath)


print(para_resnet18["conv1.weight"].data)
print(para_resnet50["conv1.weight"].data)
print(para_resnet101["conv1.weight"].data)
