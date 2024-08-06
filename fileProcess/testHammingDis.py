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


length = 50000
# 生成一百个生态分布的随机数
para = torch.normal(mean=0.0, std=0.3, size=(length,), dtype=torch.float32)
print(para)

# 获取
paraExpStrArray = []
for i in range(length):
    paraStr = BitArray(int = para.data[i].view(torch.int32), length=32).bin
    paraExpStr = paraStr[1:9]
    paraExpStrArray.append(paraExpStr)
# print(paraExpStrArray)

disDistri = np.zeros(8)
for i in range(length):
    for j in range(length):
        disDistri[hammingDis(paraExpStrArray[i], paraExpStrArray[j])] += 1
print(disDistri)


print(hammingDis("01011", "10100"))