import os
import torch
import pandas as pd
from bitstring import BitArray

device = torch.device("mps")

'''原始pth文件'''
resnet18InitParaPath = '../init/resnet18-f37072fd.pth'
resnet50InitParaPath = '../init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = '../init/resnet101-cd907fc2.pth'
vgg11InitParaPath = '../init/vgg11-8a719046.pth'
vgg13InitParaPath = '../init/vgg13-19584684.pth'
vgg16InitParaPath = '../init/vgg16-397923af.pth'
vgg19InitParaPath = '../init/vgg19-dcbb9e9d.pth'
'''用于存储bit翻转率的结果'''
outputFileRoot = './result/'

'''for xor substitute testing'''
bit_1 = torch.tensor(1, dtype=torch.int32)
bit_2 = torch.tensor(3, dtype=torch.int32)
bit_3 = torch.tensor(7, dtype=torch.int32)
bit_4 = torch.tensor(15, dtype=torch.int32)
bit_5 = torch.tensor(31, dtype=torch.int32)
bit_6 = torch.tensor(63, dtype=torch.int32)
bit_7 = torch.tensor(127, dtype=torch.int32)
bit_8 = torch.tensor(255, dtype=torch.int32)
bit_9 = torch.tensor(511, dtype=torch.int32)
bit_10 = torch.tensor(1023, dtype=torch.int32)
bit_11 = torch.tensor(2047, dtype=torch.int32)
bit_12 = torch.tensor(4095, dtype=torch.int32)
bit_13 = torch.tensor(8191, dtype=torch.int32)
bit_14 = torch.tensor(16383, dtype=torch.int32)
bit_15 = torch.tensor(32767, dtype=torch.int32)
bit_16 = torch.tensor(65535, dtype=torch.int32)
bit_17 = torch.tensor(131071, dtype=torch.int32)
bit_18 = torch.tensor(262143, dtype=torch.int32)
bit_19 = torch.tensor(524287, dtype=torch.int32)
bit_20 = torch.tensor(1048575, dtype=torch.int32)
bit_21 = torch.tensor(2097151, dtype=torch.int32)
bit_22 = torch.tensor(4194303, dtype=torch.int32)
bit_23 = torch.tensor(8388607, dtype=torch.int32)
bit_24 = torch.tensor(16777215, dtype=torch.int32)
bit_25 = torch.tensor(33554431, dtype=torch.int32)
bit_26 = torch.tensor(67108863, dtype=torch.int32)
bit_27 = torch.tensor(134217727, dtype=torch.int32)
bit_28 = torch.tensor(268435455, dtype=torch.int32)
bit_29 = torch.tensor(536870911, dtype=torch.int32)
bit_30 = torch.tensor(1073741823, dtype=torch.int32)


def getPthKeys(paraPath):
    """
    返回layers
    :param paraPath: 待获得的参数pth
    :return:
    """
    return torch.load(paraPath).keys()


def getBitFlipNum(str1, str2):
    """
    对比两个字符串，返回两个字符串中不同的比特数
    :param str1:
    :param str2:
    :return: 返回两个字符串中不同的比特数
    """
    ret = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            ret += 1
    return ret


def showBitFlip(initParaPath, retrainParaPath, bitStartIdx, bitEndIdx, outputFile):
    """
    记录在[bitStartIdx,bitEndIdx)之间有多少比特发生了翻转
    记录所有层的信息在结果文件中
    :param initPara:
    :param retrainPara:
    :param bitStartIdx:
    :param bitEndIdx:
    :param layer:
    :return:
    """
    data = {}  # 用于存储结果的字典
    initPara = torch.load(initParaPath, map_location=device)
    retrainPara = torch.load(retrainParaPath, map_location=device)

    for key in initPara.keys():  # 遍历所有键值

        if len(initPara[key].data.shape) < 1:
            continue  # 只比较参数二维及以上维度可能嵌入的层

        initLayerTensor = initPara[key].data.flatten()  # 将多维向量平铺
        retrainLayerTensor = retrainPara[key].data.flatten()

        if len(initLayerTensor) != len(retrainLayerTensor):
            continue  # 如果张量的形状不同，说明已经在最后一个全连接层，可以直接跳过

        paraNum = len(initLayerTensor)
        bitFlipNum = 0

        for idx in range(paraNum):
            initLayerEleStr = BitArray(int=initLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
            retrainLayerEleStr = BitArray(int=retrainLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
            # print(initLayerEleStr, retrainLayerEleStr)
            bitFlipNum += getBitFlipNum(initLayerEleStr, retrainLayerEleStr)

        data[key] = bitFlipNum / (paraNum * (bitEndIdx - bitStartIdx))

        print(key, paraNum, data[key])

    df = pd.DataFrame([data])

    df.to_csv(outputFile, index=False)
    return


def layerBitFLip(initParaPath, flipParaPath, bit_n, *layers):
    """
    翻转pth的layers层的低n bit
    :param initParaPath: 原始参数pth
    :param flipParaPath: 翻转之后的参数pth
    :param bit_n: 翻转低多少bit
    :return: void
    """
    para = torch.load(initParaPath)

    for layer in layers:  # layers数组中的所有layer
        if len(para[layer].data.shape) < 1:
            continue  # 单值除去
        # print(layer, type(layer))
        layerTensor = para[layer].data
        # print(layerTensor.shape)
        layerTensor_initView = layerTensor.view(torch.int32)
        # print(format(layerTensor_initView[0][0][0][0], '032b'), layerTensor[0][0][0][0])
        layerTensor_embedded_int = layerTensor_initView ^ bit_n
        layerTensor_embedded = layerTensor_embedded_int.view(torch.float32)
        # print(format(layerTensor_embedded_int[0][0][0][0], '032b'), layerTensor_embedded[0][0][0][0])

        para[layer].data = layerTensor_embedded

    torch.save(para, flipParaPath)
    return


def func(pth1, pth2, *layers):
    para1 = torch.load(pth1)
    para2 = torch.load(pth2)
    for layer in layers:
        if len(para1[layer].data.shape) <= 1:
            continue  # 只比较参数二维及以上维度可能嵌入的层
        print(layer, para1[layer].data.shape)
        para1Tensor = para1[layer].data.flatten()
        para2Tensor = para2[layer].data.flatten()
        print(format(para1Tensor[0].view(torch.int32), '032b')[9:32])
        print(format(para2Tensor[0].view(torch.int32), '032b')[9:32], "\n")

    return


if __name__ == "__main__":
    '''resnet50 bit flip'''
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_1.pth", bit_1, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_4.pth", bit_4, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_8.pth", bit_8, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_12.pth", bit_12, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_16.pth", bit_16, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_20.pth", bit_20, *getPthKeys(resnet50InitParaPath))
    # layerBitFLip(resnet50InitParaPath, "./resnet50/bitFlip/frac_23.pth", bit_23, *getPthKeys(resnet50InitParaPath))
    '''resnet50_2_CFIAR100'''
    showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2CIFAR100/frac_1_ep_5.pth", 31, 32,
                "./resnet50/2CIFAR100/result/frac_1_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2CIFAR100/frac_4_ep_5.pth", 28, 32,
                "./resnet50/2CIFAR100/result/frac_4_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2CIFAR100/frac_8_ep_5.pth", 24, 32,
                "./resnet50/2CIFAR100/result/frac_8_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2CIFAR100/frac_12_ep_5.pth", 20, 32,
                "./resnet50/2CIFAR100/result/frac_12_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2CIFAR100/frac_16_ep_5.pth", 16, 32,
                "./resnet50/2CIFAR100/result/frac_16_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2CIFAR100/frac_20_ep_5.pth", 12, 32,
                "./resnet50/2CIFAR100/result/frac_20_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2CIFAR100/frac_23_ep_10.pth", 9, 32,
                "./resnet50/2CIFAR100/result/frac_23_ep_10.csv")

    # func(resnet50InitParaPath, "./resnet50/bitFlip/frac_23.pth", *getPthKeys(resnet50InitParaPath))


