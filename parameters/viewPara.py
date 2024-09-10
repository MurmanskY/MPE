import torch
import torchvision.models as models


resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'
alexnetInitParaPath = './init/alexnet-owt-7be5be79.pth'

def showParaStructure(paraPath):
    """
    根据pth文件，显示参数的类型
    :param paraPath: pth文件路径
    :return: 打印pth文件类型，长度，key
    """
    model_pth = torch.load(paraPath)
    print("pth文件的类型是："+str(type(model_pth)))
    print("pth文件的字典长度是："+str(len(model_pth)))
    print("------pth文件的字典key包含------")
    print("key:")
    for k in model_pth.keys():
        print(k)
    print("------------------------------")

def showParaValue(paraPath):
    """
    根据pth文件，显示参数的值
    :param paraPath: pth文件路径
    :return: 打印pth文件中参数的值
    """
    model_pth = torch.load(paraPath)
    for k in model_pth:
        print(k, model_pth[k])

# print("value:")
# # 查看模型字典里面的value
# for k in model_pth:
#     print(k,model_pth[k])


if __name__ == "__main__":
    '''for test'''

    model = models.alexnet()
    model.load_state_dict(torch.load(alexnetInitParaPath))
    print(model)
    showParaStructure(alexnetInitParaPath)
    # showParaValue(paraPath)