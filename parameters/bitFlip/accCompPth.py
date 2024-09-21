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
        # print("多线程：" + key, idx)
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
        # print(key)
        initLayerTensor = initPara[key].data.flatten()
        retrainLayerTensor = retrainPara[key].data.flatten()

        if len(initLayerTensor) != len(retrainLayerTensor):
            continue  # 跳过形状不同的层

        tasks.append((key, initLayerTensor, retrainLayerTensor, bitStartIdx, bitEndIdx))

    # print("开始多线程")
    # 使用多进程池进行并行处理
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_layer, tasks)

    # print("收集结果")

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
    # showBitFlip("./resnet50/bitFlip/exp_3_allFlip.pth", "./resnet50/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./resnet50/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/sign_flip.pth", "./resnet50/2CIFAR100/sign_flip_ep_5.pth", 0, 1,
    #             "./resnet50/2CIFAR100/result/sign_flip.csv")
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
    # showBitFlip("./resnet50/bitFlip/exp_3_convFlip.pth", "./resnet50/2GTSRB/exp_3_convFlip_ep_10.pth", 6, 9,
    #             "./resnet50/2GTSRB/result/exp_3_convFlip_ep_10.csv")
    # showBitFlip("./resnet50/bitFlip/exp_3_allFlip.pth", "./resnet50/2GTSRB/exp_3_allFlip_ep_10.pth", 6, 9,
    #             "./resnet50/2GTSRB/result/exp_3_allFlip_ep_10.csv")
    '''resnet50_2_FGCVAircraft'''
    # showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2FGCVAircraft/frac_1_ep_20.pth", 31, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_1_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2FGCVAircraft/frac_4_ep_20.pth", 28, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_4_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2FGCVAircraft/frac_8_ep_20.pth", 24, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_8_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2FGCVAircraft/frac_12_ep_20.pth", 20, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_12_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2FGCVAircraft/frac_16_ep_20.pth", 16, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_16_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2FGCVAircraft/frac_20_ep_20.pth", 12, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_20_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2FGCVAircraft/frac_23_ep_20.pth", 9, 32,
    #             "./resnet50/2FGCVAircraft/result/frac_23_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/exp_3_convFlip.pth", "./resnet50/2FGCVAircraft/exp_3_convFlip_ep_20.pth", 6, 9,
    #             "./resnet50/2FGCVAircraft/result/exp_3_convFlip_ep_20.csv")
    # showBitFlip("./resnet50/bitFlip/exp_3_allFlip.pth", "./resnet50/2FGCVAircraft/exp_3_allFlip_ep_20.pth", 6, 9,
    #             "./resnet50/2FGCVAircraft/result/exp_3_allFlip_ep_20.csv")
    '''resnet50_2_PCAM'''
    # showBitFlip("./resnet50/bitFlip/frac_1.pth", "./resnet50/2PCAM/frac_1_ep_5.pth", 31, 32,
    #             "./resnet50/2PCAM/result/frac_1_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_4.pth", "./resnet50/2PCAM/frac_4_ep_5.pth", 28, 32,
    #             "./resnet50/2PCAM/result/frac_4_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_8.pth", "./resnet50/2PCAM/frac_8_ep_5.pth", 24, 32,
    #             "./resnet50/2PCAM/result/frac_8_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_12.pth", "./resnet50/2PCAM/frac_12_ep_5.pth", 20, 32,
    #             "./resnet50/2PCAM/result/frac_12_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_16.pth", "./resnet50/2PCAM/frac_16_ep_5.pth", 16, 32,
    #             "./resnet50/2PCAM/result/frac_16_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_20.pth", "./resnet50/2PCAM/frac_20_ep_5.pth", 12, 32,
    #             "./resnet50/2PCAM/result/frac_20_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/frac_23.pth", "./resnet50/2PCAM/frac_23_ep_5.pth", 9, 32,
    #             "./resnet50/2PCAM/result/frac_23_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/exp_3_convFlip.pth", "./resnet50/2PCAM/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./resnet50/2PCAM/result/exp_3_convFlip_ep_5.csv")
    # showBitFlip("./resnet50/bitFlip/exp_3_allFlip.pth", "./resnet50/2PCAM/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./resnet50/2PCAM/result/exp_3_allFlip_ep_5.csv")


    '''resnet101_2_CIFAR100'''
    # showBitFlip("./resnet101/bitFlip/frac_1.pth", "./resnet101/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./resnet101/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./resnet101/bitFlip/frac_8.pth", "./resnet101/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./resnet101/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./resnet101/bitFlip/frac_16.pth", "./resnet101/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./resnet101/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./resnet101/bitFlip/frac_23.pth", "./resnet101/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./resnet101/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./resnet101/bitFlip/exp_3_allFlip.pth", "./resnet101/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./resnet101/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./resnet101/bitFlip/exp_3_convFlip.pth", "./resnet101/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./resnet101/2CIFAR100/result/exp_3_convFlip_ep_5.csv")

    '''vgg16_2_CIFAR100'''
    # showBitFlip("./vgg16/bitFlip/frac_1.pth", "./vgg16/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./vgg16/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./vgg16/bitFlip/frac_8.pth", "./vgg16/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./vgg16/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./vgg16/bitFlip/frac_16.pth", "./vgg16/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./vgg16/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./vgg16/bitFlip/frac_23.pth", "./vgg16/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./vgg16/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./vgg16/bitFlip/exp_3_allFlip.pth", "./vgg16/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./vgg16/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./vgg16/bitFlip/exp_3_convFlip.pth", "./vgg16/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./vgg16/2CIFAR100/result/exp_3_convFlip_ep_5.csv")

    '''vgg19_2_CIFAR100'''
    # showBitFlip("./vgg19/bitFlip/frac_1.pth", "./vgg19/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./vgg19/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./vgg19/bitFlip/frac_8.pth", "./vgg19/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./vgg19/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./vgg19/bitFlip/frac_16.pth", "./vgg19/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./vgg19/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./vgg19/bitFlip/frac_23.pth", "./vgg19/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./vgg19/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./vgg19/bitFlip/exp_3_allFlip.pth", "./vgg19/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./vgg19/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./vgg19/bitFlip/exp_3_convFlip.pth", "./vgg19/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./vgg19/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    '''vgg16bn_2_CIFAR100'''
    # showBitFlip("./vgg16bn/bitFlip/frac_1.pth", "./vgg16bn/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./vgg16bn/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./vgg16bn/bitFlip/frac_8.pth", "./vgg16bn/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./vgg16bn/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./vgg16bn/bitFlip/frac_16.pth", "./vgg16bn/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./vgg16bn/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./vgg16bn/bitFlip/frac_23.pth", "./vgg16bn/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./vgg16bn/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./vgg16bn/bitFlip/exp_3_allFlip.pth", "./vgg16bn/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./vgg16bn/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./vgg16bn/bitFlip/exp_3_convFlip.pth", "./vgg16bn/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./vgg16bn/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    '''vgg19bn_2_CIFAR100'''
    # showBitFlip("./vgg19bn/bitFlip/frac_1.pth", "./vgg19bn/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./vgg19bn/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./vgg19bn/bitFlip/frac_8.pth", "./vgg19bn/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./vgg19bn/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./vgg19bn/bitFlip/frac_16.pth", "./vgg19bn/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./vgg19bn/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./vgg19bn/bitFlip/frac_23.pth", "./vgg19bn/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./vgg19bn/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./vgg19bn/bitFlip/exp_3_allFlip.pth", "./vgg19bn/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./vgg19bn/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./vgg19bn/bitFlip/exp_3_convFlip.pth", "./vgg19bn/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./vgg19bn/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    '''alexnet_2_CIFAR100'''
    # showBitFlip("./alexnet/bitFlip/frac_1.pth", "./alexnet/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./alexnet/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./alexnet/bitFlip/frac_8.pth", "./alexnet/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./alexnet/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./alexnet/bitFlip/frac_16.pth", "./alexnet/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./alexnet/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./alexnet/bitFlip/frac_23.pth", "./alexnet/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./alexnet/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./alexnet/bitFlip/exp_3_allFlip.pth", "./alexnet/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./alexnet/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./alexnet/bitFlip/exp_3_convFlip.pth", "./alexnet/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./alexnet/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    '''convnext_2_CIFAR100'''
    # showBitFlip("./convnext/bitFlip/frac_1.pth", "./convnext/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./convnext/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./convnext/bitFlip/frac_8.pth", "./convnext/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./convnext/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./convnext/bitFlip/frac_16.pth", "./convnext/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./convnext/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./convnext/bitFlip/frac_23.pth", "./convnext/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./convnext/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./convnext/bitFlip/exp_3_allFlip.pth", "./convnext/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./convnext/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./convnext/bitFlip/exp_3_convFlip.pth", "./convnext/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./convnext/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    # '''googlenet_2_CIFAR100'''
    # showBitFlip("./googlenet/bitFlip/frac_1.pth", "./googlenet/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./googlenet/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./googlenet/bitFlip/frac_8.pth", "./googlenet/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./googlenet/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./googlenet/bitFlip/frac_16.pth", "./googlenet/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./googlenet/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./googlenet/bitFlip/frac_23.pth", "./googlenet/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./googlenet/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./googlenet/bitFlip/exp_3_allFlip.pth", "./googlenet/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./googlenet/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./googlenet/bitFlip/exp_3_convFlip.pth", "./googlenet/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./googlenet/2CIFAR100/result/exp_3_convFlip_ep_5.csv")


    # '''inceptionV3_2_CIFAR100'''
    # showBitFlip("./inceptionV3/bitFlip/frac_1.pth", "./inceptionV3/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./inceptionV3/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./inceptionV3/bitFlip/frac_8.pth", "./inceptionV3/2CIFAR100/frac_8_ep_5.pth", 24, 32,
    #             "./inceptionV3/2CIFAR100/result/frac_8_ep_5.csv")
    # showBitFlip("./inceptionV3/bitFlip/frac_16.pth", "./inceptionV3/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./inceptionV3/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./inceptionV3/bitFlip/frac_23.pth", "./inceptionV3/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./inceptionV3/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./inceptionV3/bitFlip/exp_3_allFlip.pth", "./inceptionV3/2CIFAR100/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./inceptionV3/2CIFAR100/result/exp_3_allFlip_ep_5.csv")
    # showBitFlip("./inceptionV3/bitFlip/exp_3_convFlip.pth", "./inceptionV3/2CIFAR100/exp_3_convFlip_ep_5.pth", 6, 9,
    #             "./inceptionV3/2CIFAR100/result/exp_3_convFlip_ep_5.csv")

    # '''convnext_large_2_CIFAR100'''
    # showBitFlip("./convnext_large/bitFlip/frac_1.pth", "./convnext_large/2CIFAR100/frac_1_ep_5.pth", 31, 32,
    #             "./convnext_large/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_16.pth", "./convnext_large/2CIFAR100/frac_16_ep_5.pth", 16, 32,
    #             "./convnext_large/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_23.pth", "./convnext_large/2CIFAR100/frac_23_ep_5.pth", 9, 32,
    #             "./convnext_large/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/exp_3_allFlip.pth", "./convnext_large/2CIFAR100/exp_3_allFlip_ep_5.pth",6, 9,
    #             "./convnext_large/2CIFAR100/result/exp_3_allFlip_ep_5.csv")

    # '''convnext_large_2_FGCVAircraft'''
    # showBitFlip("./convnext_large/bitFlip/frac_1.pth", "./convnext_large/2FGCVAircraft/frac_1_ep_20.pth", 31, 32,
    #             "./convnext_large/2FGCVAircraft/result/frac_1_ep_20.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_16.pth", "./convnext_large/2FGCVAircraft/frac_16_ep_20.pth", 16, 32,
    #             "./convnext_large/2FGCVAircraft/result/frac_16_ep_20.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_23.pth", "./convnext_large/2FGCVAircraft/frac_23_ep_20.pth", 9, 32,
    #             "./convnext_large/2FGCVAircraft/result/frac_23_ep_20.csv")
    # showBitFlip("./convnext_large/bitFlip/exp_3_allFlip.pth", "./convnext_large/2FGCVAircraft/exp_3_allFlip_ep_20.pth",
    #             6, 9,
    #             "./convnext_large/2FGCVAircraft/result/exp_3_allFlip_ep_20.csv")

    # '''convnext_large_2_GTSRB'''
    showBitFlip("./convnext_large/bitFlip/frac_1.pth", "./convnext_large/2GTSRB/frac_1_ep_10.pth", 31, 32,
                "./convnext_large/2GTSRB/result/frac_1_ep_10.csv")
    showBitFlip("./convnext_large/bitFlip/frac_16.pth", "./convnext_large/2GTSRB/frac_16_ep_10.pth", 16, 32,
                "./convnext_large/2GTSRB/result/frac_16_ep_10.csv")
    showBitFlip("./convnext_large/bitFlip/frac_23.pth", "./convnext_large/2GTSRB/frac_23_ep_10.pth", 9, 32,
                "./convnext_large/2GTSRB/result/frac_23_ep_10.csv")
    showBitFlip("./convnext_large/bitFlip/exp_3_allFlip.pth", "./convnext_large/2GTSRB/exp_3_allFlip_ep_10.pth",
                6, 9,
                "./convnext_large/2GTSRB/result/exp_3_allFlip_ep_10.csv")

    # '''convnext_large_2_PCAM'''
    # showBitFlip("./convnext_large/bitFlip/frac_1.pth", "./convnext_large/2PCAM/frac_1_ep_5.pth", 31, 32,
    #             "./convnext_large/2PCAM/result/frac_1_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_16.pth", "./convnext_large/2PCAM/frac_16_ep_5.pth", 16, 32,
    #             "./convnext_large/2PCAM/result/frac_16_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/frac_23.pth", "./convnext_large/2PCAM/frac_23_ep_5.pth", 9, 32,
    #             "./convnext_large/2PCAM/result/frac_23_ep_5.csv")
    # showBitFlip("./convnext_large/bitFlip/exp_3_allFlip.pth", "./convnext_large/2PCAM/exp_3_allFlip_ep_5.pth",6, 9,
    #             "./convnext_large/2PCAM/result/exp_3_allFlip_ep_5.csv")

    # '''vith14_2_CIFAR100'''
    # showBitFlip("./vith14/bitFlip/frac_1.pth", "./vith14/2CIFAR100_right_key/frac_1_ep_5.pth", 31, 32,
    #             "./vith14/2CIFAR100/result/frac_1_ep_5.csv")
    # showBitFlip("./vith14/bitFlip/frac_16.pth", "./vith14/2CIFAR100_right_key/frac_16_ep_5.pth", 16, 32,
    #             "./vith14/2CIFAR100/result/frac_16_ep_5.csv")
    # showBitFlip("./vith14/bitFlip/frac_23.pth", "./vith14/2CIFAR100_right_key/frac_23_ep_5.pth", 9, 32,
    #             "./vith14/2CIFAR100/result/frac_23_ep_5.csv")
    # showBitFlip("./vith14/bitFlip/exp_3_allFlip.pth", "./vith14/2CIFAR100_right_key/exp_3_allFlip_ep_5.pth", 6, 9,
    #             "./vith14/2CIFAR100/result/exp_3_allFlip_ep_5.csv")


