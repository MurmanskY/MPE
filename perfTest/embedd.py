"""
Embedd the malware into several layers in a AI model
"""
import torch
import random
import struct
import pandas as pd
from bitstring import BitArray
from fileProcess.fileProcess import split_file, merge_file
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math


device = torch.device("mps")


'''原始pth文件'''
resnet18InitParaPath = '../parameters/init/resnet18-f37072fd.pth'
resnet50InitParaPath = '../parameters/init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = '../parameters/init/resnet101-cd907fc2.pth'
vgg11InitParaPath = '../parameters/init/vgg11-8a719046.pth'
vgg13InitParaPath = '../parameters/init/vgg13-19584684.pth'
vgg16InitParaPath = '../parameters/init/vgg16-397923af.pth'
vgg16BNInitParaPath = '../parameters/init/vgg16_bn-6c64b313.pth'
vgg19InitParaPath = '../parameters/init/vgg19-dcbb9e9d.pth'
vgg19BNInitParaPath = '../parameters/init/vgg19_bn-c79401a0.pth'
alexnetInitParaPath = '../parameters/init/alexnet-owt-7be5be79.pth'
convnextInitParaPath = '../parameters/init/convnext_base-6075fbad.pth'
convnext_largeInitParaPath = '../parameters/init/convnext_large-ea097f82.pth'
googlenetInitParaPath = '../parameters/init/googlenet-1378be20.pth'
inceptionV3InitParaPath = '../parameters/init/inception_v3_google-0cc3c7bd.pth'
vitb16InitParaPath = '../parameters/init/vit_b_16-c867db91.pth'
vith14_lcswagInitParaPath = '../parameters/init/vit_h_14_lc_swag-c1eb923e.pth'
swinv2bInitParaPath = '../parameters/init/swin_v2_b-781e5279.pth'


def flipBit(ch):
    """
    翻转比特
    :param ch:
    :return: ch
    """
    if ch == "0":
        return "1"
    else:
        return "0"


def encode(ch):
    """
    返回编码结果
    :param ch:
    :return:
    """
    return ch + flipBit(ch) + flipBit(ch)


def getPthKeys(paraPath):
    """
    返回layers
    :param paraPath: 待获得的参数pth
    :return:
    """
    return torch.load(paraPath, map_location=torch.device("mps")).keys()


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

def getExpEmbeddSize(initParaPath, layers, interval, correct):
    """
    返回指数部分最大的嵌入容量，单位是字节Byte
    :param initParaPath:
    :param layers: list
    :param interval: 每interval个中嵌入一个
    :return: list
    """
    para = torch.load(initParaPath, map_location=torch.device("mps"))
    ret = []
    for layer in layers:
        paraTensor = para[layer].data
        paraTensor_flat = paraTensor.flatten()
        # print(initParaPath, layers, paraTensor_flat.size())
        layerSize = len(paraTensor_flat) // (interval * correct * 8)
        # print(layer, len(paraTensor_flat), layerSize)
        ret.append(layerSize)
    return ret


def layerExpBitEmbedd(initParaPath, flipParaPath, layers, malware, interval, correct):
    """
    单线程
    使用编码规则100和011进行嵌入低3bit,嵌入恶意软件，将卷积层展平，使用交错的方式编码
    :param initParaPath:
    :param flipParaPath:
    :param layers: layer list
    :param malware: 需要嵌入的恶意软件
    :param interval: 每interval个中嵌入一个
    :param correct: 冗余多少
    :return:
    """
    sizeList = getExpEmbeddSize(initParaPath, layers, interval, correct)  # 一共可以嵌入的信息数量Byte
    print("sizeList(byte): ", sizeList)
    layerNum = 0
    malware_total_str = BitArray(filename=malware).bin
    print("malware length(byte): ", len(malware_total_str) / 8)
    malwares = []
    if len(malware_total_str) > sum(sizeList) * 8:
        print("恶意软件过大，不能进行嵌入")
    else:
        print("恶意软件可以进行嵌入")
        allocated_mal_bit = 0
        while allocated_mal_bit < len(malware_total_str):
            start = allocated_mal_bit
            end = min(allocated_mal_bit + sizeList[layerNum] * 8, len(malware_total_str))
            malwares.append(malware_total_str[start : end])
            allocated_mal_bit += (sizeList[layerNum] * 8)
            layerNum += 1

    print(layers[:layerNum])  # 返回进行嵌入的层
    print(initParaPath, " 恶意软件为：",malware, " 嵌入了的层数为：", layerNum)

    para = torch.load(initParaPath, map_location=torch.device("mps"))

    for layer, malwareStr in zip(layers, malwares):
        # 对于每个层
        malwareLen = len(malwareStr)
        writePos = 0  # 恶意软件读写指针
        correctionPos = 0  # 冗余指针，标识目前处于第几个冗余
        currentWriteContent = malwareStr[writePos]  # 存储当前需要写入的内容

        paraTensor_flat = para[layer].flatten()

        while writePos < malwareLen:  # 写恶意软件中每一个bit
            while correctionPos < correct:  # 交错位置冗余
                index = writePos + correctionPos * interval * malwareLen
                paraTensor_flat_str = BitArray(int = paraTensor_flat[index].view(torch.int32), length=32).bin
                newParaTensor_flat_str = paraTensor_flat_str[:6] + encode(currentWriteContent) + paraTensor_flat_str[9:32]
                # 判断是否存在溢出
                if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                    newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                    paraTensor_flat[index] = newParaInt.view(torch.float32)
                else:
                    newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                    paraTensor_flat[index] = newParaInt.view(torch.float32)

                correctionPos += 1
                if correctionPos == correct:
                    correctionPos = 0
                    break
            writePos += 1
            if writePos >= malwareLen:
                break
            else:
                currentWriteContent = malwareStr[writePos]

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)

        torch.save(para, flipParaPath)
    return




if __name__ == "__main__":
    """resnet50进行嵌入"""
    layers = ["layer4.0.conv2.weight",
              "layer4.1.conv2.weight",
              "layer4.2.conv2.weight"]
    malware = "./malware/Zherkov.EXE"
    interval = 7
    correct = 11
    # sizeList = getExpEmbeddSize(resnet50InitParaPath, layers, interval, correct)
    # layerExpBitEmbedd(resnet50InitParaPath, "./embeddPara/resnet50_CivilWar.pth",
    #                   layers, "./malware/CivilWar.COM", interval, correct)
    # layerExpBitEmbedd(resnet50InitParaPath, "./embeddPara/resnet50_Tokyo.pth",
    #                   layers, "./malware/Tokyo.EXE", interval, correct)
    # layerExpBitEmbedd(resnet50InitParaPath, "./embeddPara/resnet50_DropBatch.pth",
    #                   layers, "./malware/DropBatch.BAT", interval, correct)

    # layerExpBitEmbedd(resnet101InitParaPath, "./embeddPara/resnet101_CivilWar.pth",
    #                   layers, "./malware/CivilWar.COM", interval, correct)
    # layerExpBitEmbedd(resnet101InitParaPath, "./embeddPara/resnet101_Tokyo.pth",
    #                   layers, "./malware/Tokyo.EXE", interval, correct)
    # layerExpBitEmbedd(resnet101InitParaPath, "./embeddPara/resnet101_DropBatch.pth",
    #                   layers, "./malware/DropBatch.BAT", interval, correct)
    # layerExpBitEmbedd(resnet101InitParaPath, "./embeddPara/resnet101_Zherkov.pth",
    #                   layers, "./malware/Zherkov.EXE", interval, correct)
    # layerExpBitEmbedd(resnet101InitParaPath, "./embeddPara/resnet101_ParityBoot.pth",
    #                   layers, "./malware/ParityBoot.IMA", interval, correct)

    layers = ["features.5.1.mlp.0.weight",
              "features.5.2.mlp.0.weight",
              "features.5.3.mlp.0.weight",
              "features.5.4.mlp.0.weight",
              "features.5.5.mlp.0.weight",
              "features.5.6.mlp.0.weight",
              "features.5.7.mlp.0.weight",
              "features.5.8.mlp.0.weight",
              "features.5.9.mlp.0.weight",
              "features.5.10.mlp.0.weight",
              "features.5.11.mlp.0.weight",
              "features.5.12.mlp.0.weight",
              "features.7.0.mlp.0.weight",
              "features.7.1.mlp.0.weight"]
    interval = 8
    correct = 7
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_CivilWar.pth",
    #                   layers, "./malware/CivilWar.COM", interval, correct)  # 1
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_Tokyo.pth",
    #                   layers, "./malware/Tokyo.EXE", interval, correct)  # 1
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_DropBatch.pth",
    #                   layers, "./malware/DropBatch.BAT", interval, correct)  # 2
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_Zherkov.pth",
    #                   layers, "./malware/Zherkov.EXE", interval, correct)  # 3
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_ParityBoot.pth",
    #                   layers, "./malware/ParityBoot.IMA", interval, correct)  # 5
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_TripleFantasy.pth",
    #                   layers, "./malware/TripleFantasy", interval, correct)  # 8
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_Stuxnet.pth",
    #                   layers, "./malware/Stuxnet.exe", interval, correct)  # 11
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_SekoiaRootkit.pth",
    #                   layers, "./malware/SekoiaRootkit", interval, correct)  # 13
    # layerExpBitEmbedd(swinv2bInitParaPath, "./embeddPara/swinv2b_Lxeshe.pth",
    #                   layers, "./malware/Lxeshe.EXE", interval, correct)  # 14

    layers = ["features.5.10.block.3.weight",
              "features.5.10.block.5.weight",
              "features.5.11.block.3.weight",
              "features.5.11.block.5.weight",
              "features.5.12.block.3.weight",
              "features.5.12.block.5.weight",
              "features.5.13.block.3.weight",
              "features.5.13.block.5.weight",
              "features.5.14.block.3.weight",
              "features.5.14.block.5.weight",
              "features.5.15.block.3.weight",
              "features.5.15.block.5.weight",
              "features.5.16.block.3.weight",
              "features.5.16.block.5.weight",
              "features.5.17.block.3.weight",
              "features.5.17.block.5.weight",
              "features.5.18.block.3.weight",
              "features.5.18.block.5.weight",
              "features.5.19.block.3.weight",
              "features.5.19.block.5.weight",
              "features.5.20.block.3.weight",
              "features.5.20.block.5.weight",
              "features.5.21.block.3.weight",
              "features.5.21.block.5.weight",
              "features.5.22.block.3.weight",
              "features.5.22.block.5.weight",
              "features.5.23.block.3.weight",
              "features.5.23.block.5.weight",
              "features.5.24.block.3.weight",
              "features.5.24.block.5.weight",
              "features.5.25.block.3.weight",
              "features.5.25.block.5.weight",
              "features.5.26.block.3.weight",
              "features.5.26.block.5.weight",
              "features.7.0.block.3.weight",
              "features.7.0.block.5.weight",
              "features.7.1.block.3.weight",
              "features.7.1.block.5.weight",
              "features.7.2.block.3.weight",
              "features.7.2.block.5.weight"]
    interval = 4
    correct = 7
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_CivilWar.pth",
                      layers, "./malware/CivilWar.COM", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_Tokyo.pth",
                      layers, "./malware/Tokyo.EXE", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_DropBatch.pth",
                      layers, "./malware/DropBatch.BAT", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_Zherkov.pth",
                      layers, "./malware/Zherkov.EXE", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_ParityBoot.pth",
                      layers, "./malware/ParityBoot.IMA", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_TripleFantasy.pth",
                      layers, "./malware/TripleFantasy", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_Stuxnet.pth",
                      layers, "./malware/Stuxnet.exe", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convbb_SekoiaRootkit.pth",
                      layers, "./malware/SekoiaRootkit", interval, correct)  #
    layerExpBitEmbedd(convnextInitParaPath, "./embeddPara/convb_Lxeshe.pth",
                      layers, "./malware/Lxeshe.EXE", interval, correct)  #

