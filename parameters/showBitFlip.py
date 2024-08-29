import os
import torch
import pandas as pd
from bitstring import BitArray

device = torch.device("mps")

resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'

outputFileRoot = './bitFlip/result/'


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

        if len(initPara[key].data.shape) <= 1:
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

        print(key, data[key], paraNum)

    df = pd.DataFrame([data])

    df.to_csv(outputFile, index=False)
    return



if __name__ == "__main__":
    '''resnet50_2_CFIAR100'''
    # showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
    #             0, 1, "./bitFlip/result/resnet502CFIAR100_low1Bit.csv" )
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 4, "./bitFlip/result/resnet502CFIAR100_low4Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 8, "./bitFlip/result/resnet502CFIAR100_low8Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 12, "./bitFlip/result/resnet502CFIAR100_low12Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 16, "./bitFlip/result/resnet502CFIAR100_low16Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 20, "./bitFlip/result/resnet502CFIAR100_low20Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                0, 23, "./bitFlip/result/resnet502CFIAR100_low23Bit.csv")
    showBitFlip(resnet50InitParaPath, "./retrained_ImageNet2CIFAR100/res50_ep5.pth",
                23, 26, "./bitFlip/result/resnet502CFIAR100_exp_low3Bit.csv")


