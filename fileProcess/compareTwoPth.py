"""
此文件用来比较两个pth文件之间的不同
"""
import numpy as np
import torch
from bitstring import BitArray


device = torch.device("mps")

init_path = "../parameters/init/resnet50-11ad3fa6.pth"
path1 = "../parameters/expXOR/resnet50ThirdConv1_low4_method3.pth"  # 首层卷积层的指数部分低Nbit进行翻转
path2 = "../parameters/embeddedRetrainPCAM/resnet50EmbededThirdConv2Low4XOR2PCAM_3_method3.pth"  # 进行重训练后的结果




def show_exp(tens):
    """
    通过一个torch.float32的数据元素，返回它的指数部分
    :param tens:
    :return:
    """
    return(BitArray(int = tens.view(torch.int32), length=32).bin[1:9])




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




para_init = torch.load(init_path, map_location=device)
para1 = torch.load(path1, map_location=device)
para2 = torch.load(path2, map_location=device)

dim0, dim1, dim2, dim3 = para1["layer1.0.conv2.weight"].shape

conv1W1 = para_init["layer1.0.conv2.weight"].data
conv1W2 = para1["layer1.0.conv2.weight"].data
conv1W3 = para2["layer1.0.conv2.weight"].data




HD = np.zeros(9)
count = 0
for i in range(dim0):
    for j in range(dim1):
        for k in range(dim2):
            for m in range(dim3):
                if conv1W1[i][j][k][m] != conv1W2[i][j][k][m]:  # 找到修改参数的位置
                    hD = hammingDis(show_exp(conv1W2[i][j][k][m]), show_exp(conv1W3[i][j][k][m]))
                    print(show_exp(conv1W1[i][j][k][m]), show_exp(conv1W2[i][j][k][m]), show_exp(conv1W3[i][j][k][m]), hD)
                    HD[hD] += 1
                    count += 1

print(count)
print(HD/4096)