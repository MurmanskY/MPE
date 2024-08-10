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
para_retrained_1 = "../parameters/retrained_ImageNet2PCAM/resnet50re_10.pth"
para_retrained_10 = "../parameters/retrained_ImageNet2FGVCAircraft/resnet50re_20.pth"
para_retrained_20 = "../parameters/retrained_ImageNet2FGVCAircraft/resnet50re_30.pth"


paraInit = torch.load(para_init, map_location=torch.device('mps'))
paraRetrained1 = torch.load(para_retrained_1, map_location=torch.device('mps'))
paraRetrained10 = torch.load(para_retrained_10, map_location=torch.device('mps'))
paraRetrained20 = torch.load(para_retrained_20, map_location=torch.device('mps'))


paraPos = "conv1.weight"



paraInitTensor = paraInit[paraPos].data
paraRetrained1Tensor = paraRetrained1[paraPos].data
paraRetrained10Tensor = paraRetrained10[paraPos].data
paraRetrained20Tensor = paraRetrained20[paraPos].data
paraInitTensor_intView = paraInitTensor.view(torch.int32)
paraRetrained1Tensor_intView = paraRetrained1Tensor.view(torch.int32)
paraRetrained10Tensor_intView = paraRetrained10Tensor.view(torch.int32)
paraRetrained20Tensor_intView = paraRetrained20Tensor.view(torch.int32)


hamming1 = np.zeros(9)
hamming10 = np.zeros(9)
hamming20 = np.zeros(9)


'''for conv parameters'''
dim0, dim1, dim2, dim3 = paraInitTensor.shape
paraNum = dim0 * dim1 * dim2 * dim3
for i in range(dim0):
    for j in range(dim1):
        for k in range(dim2):
            for m in range(dim3):
                temp1 = format(paraInitTensor_intView[i][j][k][m], '032b')[1:39]
                temp2 = format(paraRetrained1Tensor_intView[i][j][k][m], '032b')[1:9]
                temp3 = format(paraRetrained10Tensor_intView[i][j][k][m], '032b')[1:9]
                temp4 = format(paraRetrained20Tensor_intView[i][j][k][m], '032b')[1:9]
                hamming1[hammingDis(temp1, temp2)] += 1
                hamming10[hammingDis(temp1, temp3)] += 1
                hamming20[hammingDis(temp1, temp4)] += 1
hamming1 /= paraNum
hamming10 /= paraNum
hamming20 /= paraNum
print(hamming1, hamming1[0] + hamming1[1] + hamming1[2] + hamming1[3])
print(hamming1, hamming10[0] + hamming10[1] + hamming10[2] + hamming10[3])
print(hamming1, hamming20[0] + hamming20[1] + hamming20[2] + hamming20[3])



'''for 2 dim weight'''
# fcdim0, fcdim1 = paraInitTensor.shape
# fcparaNum = fcdim0 * fcdim1
# for i in range(fcdim0):
#     for j in range(fcdim1):
#         temp1 = format(paraInitTensor_intView[i][j], '032b')[1:9]
#         temp2 = format(paraRetrained1Tensor_intView[i][j], '032b')[1:9]
#         temp3 = format(paraRetrained10Tensor_intView[i][j], '032b')[1:9]
#         temp4 = format(paraRetrained20Tensor_intView[i][j], '032b')[1:9]
#         hamming1[hammingDis(temp1, temp2)] += 1
#         hamming10[hammingDis(temp1, temp3)] += 1
#         hamming20[hammingDis(temp1, temp4)] += 1
# hamming1 /= fcparaNum
# hamming10 /= fcparaNum
# hamming20 /= fcparaNum
# print(hamming1, hamming1[0] + hamming1[1] + hamming1[2] + hamming1[3])
# print(hamming1, hamming10[0] + hamming10[1] + hamming10[2] + hamming10[3])
# print(hamming1, hamming20[0] + hamming20[1] + hamming20[2] + hamming20[3])