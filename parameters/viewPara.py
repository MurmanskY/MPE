import torch
import torchvision.models as models


resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg16BNInitParaPath = './init/vgg16_bn-6c64b313.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'
vgg19BNInitParaPath = './init/vgg19_bn-c79401a0.pth'
alexnetInitParaPath = './init/alexnet-owt-7be5be79.pth'
convnextInitParaPath = './init/convnext_base-6075fbad.pth'
vitb16InitParaPath = './init/vit_b_16-c867db91.pth'
swinbInitParaPath = './init/swin_b-68c6b09e.pth'
inceptionV3InitParaPath = './init/inception_v3_google-0cc3c7bd.pth'
googlenetInitParaPath = './init/googlenet-1378be20.pth'
densenet121InitParaPath = './init/densenet121-a639ec97.pth'
densenet201InitParaPath = './init/densenet201-c1103571.pth'
vith14_lcswagInitParaPath = './init/vit_h_14_lc_swag-c1eb923e.pth'
vith14_swagInitParaPath = './init/vit_h_14_swag-80465313.pth'
convnext_large_InitParaPath = './init/convnext_large-ea097f82.pth'
swinv2bInitParaPath = './init/swin_v2_b-781e5279.pth'

device = torch.device("mps")

def showParaStructure(paraPath):
    """
    根据pth文件，显示参数的类型
    :param paraPath: pth文件路径
    :return: 打印pth文件类型，长度，key
    """
    model_pth = torch.load(paraPath, map_location=device)
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


'''处理densenee预训练参数中的key'''
# # 加载原始 state_dict
# state_dict = torch.load(densenet201InitParaPath)
#
# # 创建一个新的 state_dict，修复不匹配的键
# new_state_dict = {}
# for key in state_dict.keys():
#     # 修复键名，将 'norm.1' 改为 'norm1'，类似地处理其他键名
#     new_key = key.replace('norm.1', 'norm1').replace('conv.1', 'conv1') \
#                  .replace('norm.2', 'norm2').replace('conv.2', 'conv2')
#     new_state_dict[new_key] = state_dict[key]
#
# # 保存修正后的 state_dict 为新的 pth 文件
# torch.save(new_state_dict, densenet201InitParaPath)


if __name__ == "__main__":
    '''for test'''




    model = models.swin_v2_b()
    model.load_state_dict(torch.load(swinv2bInitParaPath))
    print(model)
    showParaStructure(swinv2bInitParaPath)
    # showParaValue(paraPath)