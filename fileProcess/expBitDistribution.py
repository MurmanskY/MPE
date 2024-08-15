"""测试参数的指数部分的比特分布"""
import numpy as np
import torch
from bitstring import BitArray

pth = "../parameters/init/resnet101-cd907fc2.pth"  # 需要查看的pth文件
layer = "conv1.weight"  # 需要查看的参数的层

def show_exp(tens):
    """
    通过一个torch.float32的数据元素，返回它的指数部分
    :param tens:
    :return:
    """
    return(BitArray(int = tens.view(torch.int32), length=32).bin[1:9])


bit1Distr = np.zeros(8)
bit0Distr = np.zeros(8)
paras = torch.load(pth)
paraTensor = paras[layer].data
dim0, dim1, dim2, dim3 = paraTensor.shape
paraNum = dim0 * dim1 * dim2 * dim3

for i in range(dim0):
    for j in range(dim1):
        for k in range(dim2):
            for m in range(dim3):
                tempStr = show_exp(paraTensor[i][j][k][m])
                for idx in range(len(tempStr)):
                    if tempStr[idx] == '1':
                        bit1Distr[idx] += 1
                    else:
                        bit0Distr[idx] += 1

bit1Distr /= paraNum
bit0Distr /= paraNum

print(bit1Distr)
print(bit0Distr)





