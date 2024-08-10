"""翻转预训练好的参数"""
import torch
import numpy as np
from bitstring import BitArray

'''for xor substitute testing'''
bit_1 = torch.tensor(1, dtype=torch.int32)
bit_2 = torch.tensor(3, dtype=torch.int32)
bit_3 = torch.tensor(7, dtype=torch.int32)
bit_4 = torch.tensor(15, dtype=torch.int32)
bit_5 = torch.tensor(31, dtype=torch.int32)
bit_6 = torch.tensor(63, dtype=torch.int32)
bit_7 = torch.tensor(127, dtype=torch.int32)
bit_8 = torch.tensor(255, dtype=torch.int32)

resnet18InitParaPath = '../parameters/init/resnet18-f37072fd.pth'
resnet50InitParaPath = '../parameters/init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = '../parameters/init/resnet101-cd907fc2.pth'
vgg11InitParaPath = '../parameters/init/vgg11-8a719046.pth'
vgg13InitParaPath = '../parameters/init/vgg13-19584684.pth'
vgg16InitParaPath = '../parameters/init/vgg16-397923af.pth'
vgg19InitParaPath = '../parameters/init/vgg19-dcbb9e9d.pth'

def strFlip(str):
    """
    翻转字符串
    :param str:
    :return:
    """
    flipped_string = ''.join('1' if char == '0' else '0' for char in str)
    return flipped_string


def firstConvExpLastBitXOR(paraPath, Last_N, embeddedParaPath):
    """
    第一个卷积层的参数后Nbit翻转
    :param paraPath:
    :param bitReplacement:
    :param embeddedParaPath:
    :return:
    """
    para = torch.load(paraPath)
    fcWeightsTensor = para["conv1.weight"].data


    dim0, dim1, dim2, dim3 = fcWeightsTensor.shape
    # 随机选择若干个参数更改
    index0 = np.random.choice(np.arange(0, dim0), dim0 // 2, replace=False)
    index1 = np.random.choice(np.arange(0, dim1), dim1, replace=False)
    index2 = np.random.choice(np.arange(0, dim2), dim2, replace=False)
    index3 = np.random.choice(np.arange(0, dim3), dim3 // 2, replace=False)
    for idx0 in index0:
        for idx1 in index1:
            for idx2 in index2:
                for idx3 in index3:
                    paraStr = BitArray(int=fcWeightsTensor[idx0][idx1][idx2][idx3].view(torch.int32), length=32).bin
                    print(fcWeightsTensor[idx0][idx1][idx2][idx3],
                          BitArray(int=fcWeightsTensor[idx0][idx1][idx2][idx3].view(torch.int32), length=32).bin[1:9])

                    newParaStr = paraStr[:9-Last_N] + strFlip(paraStr[9-Last_N:9]) + paraStr[9:32]
                    # 判断是否存在int32溢出的情况
                    if int(newParaStr, 2) >= 2 ** 31:
                        newParaInt = torch.tensor(int(newParaStr, 2) - 2 ** 32, dtype=torch.int32)
                        fcWeightsTensor[idx0][idx1][idx2][idx3] = newParaInt.view(torch.float32)
                    else:
                        newParaInt = torch.tensor(int(newParaStr, 2), dtype=torch.int32)
                        fcWeightsTensor[idx0][idx1][idx2][idx3] = newParaInt.view(torch.float32)
                    print(fcWeightsTensor[idx0][idx1][idx2][idx3],
                          BitArray(int=fcWeightsTensor[idx0][idx1][idx2][idx3].view(torch.int32), length=32).bin[1:9])
                    print("\n")
    para["conv1.weight"].data = fcWeightsTensor
    torch.save(para, embeddedParaPath)

    return

if __name__ == "__main__":
    firstConvExpLastBitXOR(resnet50InitParaPath, 4, "../parameters/expXOR/resnet50FirstConv1_low4.pth")
    # print(strFlip("01010"))