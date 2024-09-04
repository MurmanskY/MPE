"""
此代码使用并行的方式用于加速，两个pth文件的对比过程
使用多线程
"""
import torch
import pandas as pd
from bitstring import BitArray
from multiprocessing import Pool, cpu_count

device = torch.device("cpu")

def getBitFlipNum(initBits, retrainBits):
    return sum(1 for i in range(len(initBits)) if initBits[i] != retrainBits[i])

def process_layer(args):
    key, initLayerTensor, retrainLayerTensor, bitStartIdx, bitEndIdx = args
    paraNum = len(initLayerTensor)
    bitFlipNum = 0

    for idx in range(paraNum):
        initLayerEleStr = BitArray(int=initLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
        retrainLayerEleStr = BitArray(int=retrainLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
        bitFlipNum += getBitFlipNum(initLayerEleStr, retrainLayerEleStr)

    bitFlipRate = bitFlipNum / (paraNum * (bitEndIdx - bitStartIdx))
    return key, bitFlipRate

def showBitFlip(initParaPath, retrainParaPath, bitStartIdx, bitEndIdx, outputFile):
    data = {}
    initPara = torch.load(initParaPath, map_location=device)
    retrainPara = torch.load(retrainParaPath, map_location=device)

    tasks = []
    for key in initPara.keys():
        if len(initPara[key].data.shape) < 1:
            continue  # 跳过非二维及以上维度的层

        initLayerTensor = initPara[key].data.flatten()
        retrainLayerTensor = retrainPara[key].data.flatten()

        if len(initLayerTensor) != len(retrainLayerTensor):
            continue  # 跳过形状不同的层

        tasks.append((key, initLayerTensor, retrainLayerTensor, bitStartIdx, bitEndIdx))

    # 使用多进程池进行并行处理
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_layer, tasks)

    # 收集结果
    for key, bitFlipRate in results:
        data[key] = bitFlipRate
        print(key, len(initPara[key].data.flatten()), bitFlipRate)

    df = pd.DataFrame([data])
    df.to_csv(outputFile, index=False)
    return

if __name__ == '__main__':
    '''resnet50_2_CFIAR100'''
    # showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./resnet50/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2CIFAR100/frac_4_ep_5.pth", 28, 32,
    #             "./resnet50/2CIFAR100/result/frac_4_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./resnet50/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2CIFAR100/frac_12_ep_5.pth", 20, 32,
    #             "./resnet50/2CIFAR100/result/frac_12_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./resnet50/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2CIFAR100/frac_20_ep_5.pth", 12, 32,
    #             "./resnet50/2CIFAR100/result/frac_20_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2CIFAR100/frac_23_ep_10.pth", 9, 32,
    #             "./resnet50/2CIFAR100/result/frac_23_ep_10.csv")
    '''resnet50_2_OxfordIIITPet'''
    # showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2OxfordIIITPet/frac_1_ep_10.pth", 31, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_1_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2OxfordIIITPet/frac_4_ep_10.pth", 28, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_4_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2OxfordIIITPet/frac_8_ep_10.pth", 24, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_8_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2OxfordIIITPet/frac_12_ep_10.pth", 20, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_12_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2OxfordIIITPet/frac_16_ep_10.pth", 16, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_16_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2OxfordIIITPet/frac_20_ep_10.pth", 12, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_20_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2OxfordIIITPet/frac_23_ep_10.pth", 9, 32,
    #             "./resnet50/2OxfordIIITPet/result/frac_23_ep_10.csv")
    '''resnet50_2_GTSRB'''
    # showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2GTSRB/frac_1_ep_10.pth", 31, 32,
    #             "./resnet50/2GTSRB/result/frac_1_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2GTSRB/frac_4_ep_10.pth", 28, 32,
    #             "./resnet50/2GTSRB/result/frac_4_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2GTSRB/frac_8_ep_10.pth", 24, 32,
    #             "./resnet50/2GTSRB/result/frac_8_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2GTSRB/frac_12_ep_10.pth", 20, 32,
    #             "./resnet50/2GTSRB/result/frac_12_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2GTSRB/frac_16_ep_10.pth", 16, 32,
    #             "./resnet50/2GTSRB/result/frac_16_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2GTSRB/frac_20_ep_10.pth", 12, 32,
    #             "./resnet50/2GTSRB/result/frac_20_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2GTSRB/frac_23_ep_10.pth", 9, 32,
    #             "./resnet50/2GTSRB/result/frac_23_ep_10.csv")
    '''resnet50_2_PCAM'''
    showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2PCAM/frac_1_ep_5.pth", 31, 32,
                "./resnet50/2PCAM/result/frac_1_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2PCAM/frac_4_ep_5.pth", 28, 32,
                "./resnet50/2PCAM/result/frac_4_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2PCAM/frac_8_ep_5.pth", 24, 32,
                "./resnet50/2PCAM/result/frac_8_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2PCAM/frac_12_ep_5.pth", 20, 32,
                "./resnet50/2PCAM/result/frac_12_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2PCAM/frac_16_ep_5.pth", 16, 32,
                "./resnet50/2PCAM/result/frac_16_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2PCAM/frac_20_ep_5.pth", 12, 32,
                "./resnet50/2PCAM/result/frac_20_ep_5.csv")
    showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2PCAM/frac_23_ep_5.pth", 9, 32,
                "./resnet50/2PCAM/result/frac_23_ep_5.csv")

