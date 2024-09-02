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


def pthBitFLip(initParaPath, resnet50InitParaPath, )


if __name__ == "__main__":
    '''resnet50 bit flip'''
    pthBitFLip()
    '''resnet50_2_CFIAR100'''



