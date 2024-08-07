import torch
from bitstring import BitArray
import numpy as np
import matplotlib


def hammingDis(s1, s2):
    """
    计算汉明距离
    :param s1:
    :param s2:
    :return:
    """
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")
    distance = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    return distance


def hamming_distance_numbers(n1, n2):
    """
    计算数字之间的汉明距离
    :param n1:
    :param n2:
    :return:
    """
    """
    Calculate the Hamming distance between two integers.

    Parameters:
    n1 (int): The first integer.
    n2 (int): The second integer.

    Returns:
    int: The Hamming distance between the two integers.
    """
    # XOR the two numbers to find differing bits
    xor_result = n1 ^ n2

    # Count the number of differing bits
    distance = bin(xor_result).count('1')
    return distance

device = torch.device("mps")
para_init = "../parameters/init/resnet50-11ad3fa6.pth"
para_retrained_1 = "../parameters/retrained/resnet50re.pth"
para_retrained_10 = "../parameters/retrained/resnet50re_10.pth"


paraInit = torch.load(para_init)
paraRetrained1 = torch.load(para_retrained_1, map_location=torch.device('mps'))
paraRetrained10 = torch.load(para_retrained_10, map_location=torch.device('mps'))


paraInitTensor = paraInit["conv1.weight"].data
paraRetrained1Tensor = paraRetrained1["conv1.weight"].data
paraRetrained10Tensor = paraRetrained10["conv1.weight"].data
paraInitTensor_intView = paraInitTensor.view(torch.int32)
paraRetrained1Tensor_intView = paraRetrained1Tensor.view(torch.int32)
paraRetrained10Tensor_intView = paraRetrained10Tensor.view(torch.int32)

hamming1 = np.zeros(9)
hamming2 = np.zeros(9)

dim0, dim1, dim2, dim3 = paraInitTensor.shape
for i in range(dim0):
    for j in range(dim1):
        for k in range(dim2):
            for m in range(dim3):
                temp1 = format(paraInitTensor_intView[i][j][k][m], '032b')[1:9]
                temp2 = format(paraRetrained1Tensor_intView[i][j][k][m], '032b')[1:9]
                temp3 = format(paraRetrained10Tensor_intView[i][j][k][m], '032b')[1:9]
                hamming1[hammingDis(temp1, temp2)] += 1
                hamming2[hammingDis(temp1, temp3)] += 1
                print(temp1)
                print(temp2)
                print(temp3)
                print("\n")

print(hamming1)
print(hamming2)


# length = 50000
# # 生成一百个生态分布的随机数
# para = torch.normal(mean=0.0, std=0.3, size=(length,), dtype=torch.float32)
# print(para)
#
# # 获取
# paraExpStrArray = []
# for i in range(length):
#     paraStr = BitArray(int = para.data[i].view(torch.int32), length=32).bin
#     paraExpStr = paraStr[1:9]
#     paraExpStrArray.append(paraExpStr)
# # print(paraExpStrArray)
#
# disDistri = np.zeros(8)
# for i in range(length):
#     for j in range(length):
#         disDistri[hammingDis(paraExpStrArray[i], paraExpStrArray[j])] += 1
# print(disDistri)
#
#
# print(hammingDis("01011", "10100"))