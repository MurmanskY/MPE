import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import numpy as np
import struct
import sys
import struct

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


def float_to_ieee754_exp_str(number):
    # 将数字转换为IEEE 754标准的32位浮点数
    packed = struct.pack('>f', number)

    # 将浮点数解包为一个整数
    unpacked = struct.unpack('>I', packed)[0]

    # 提取指数部分（位移8位并截取低8位）
    exponent_bits = (unpacked >> 23) & 0xFF

    # 减去127得到偏移后的指数
    exponent = exponent_bits

    return format(exponent, '032b')[24:32]


# 示例数字
hammingD = np.zeros(9)
num = 10000
numbers = np.random.normal(loc=0, scale=10, size=num)
for i in range(len(numbers)-1):
    if hammingDis(float_to_ieee754_exp_str(numbers[i]), float_to_ieee754_exp_str(numbers[i+1])) != 0:
        print(numbers[i], float_to_ieee754_exp_str(numbers[i]))
        print(numbers[i+1], float_to_ieee754_exp_str(numbers[i+1]))
        print(hammingDis(float_to_ieee754_exp_str(numbers[i]), float_to_ieee754_exp_str(numbers[i+1])))
        print("\n")
    hammingD[hammingDis(float_to_ieee754_exp_str(numbers[i]), float_to_ieee754_exp_str(numbers[i+1]))] += 1

hammingD /= num
print(hammingD)
print(hammingD[0] + hammingD[1] + hammingD[2] + hammingD[3])
# 获取指数部分并打印
