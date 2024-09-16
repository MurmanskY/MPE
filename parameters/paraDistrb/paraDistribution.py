import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import entropy

'''原始pth文件'''
resnet18InitParaPath = '../init/resnet18-f37072fd.pth'
resnet50InitParaPath = '../init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = '../init/resnet101-cd907fc2.pth'
vgg11InitParaPath = '../init/vgg11-8a719046.pth'
vgg13InitParaPath = '../init/vgg13-19584684.pth'
vgg16InitParaPath = '../init/vgg16-397923af.pth'
vgg16BNInitParaPath = '../init/vgg16_bn-6c64b313.pth'
vgg19InitParaPath = '../init/vgg19-dcbb9e9d.pth'
vgg19BNInitParaPath = '../init/vgg19_bn-c79401a0.pth'
alexnetInitParaPath = '../init/alexnet-owt-7be5be79.pth'
convnextInitParaPath = '../init/convnext_base-6075fbad.pth'
googlenetInitParaPath = '../init/googlenet-1378be20.pth'
inceptionV3InitParaPath = '../init/inception_v3_google-0cc3c7bd.pth'
vitb16InitParaPath = '../init/vit_b_16-c867db91.pth'
densenet121InitParaPath = '../init/densenet121-a639ec97.pth'
densenet201InitParaPath = '../init/densenet201-c1103571.pth'



'''使用不同编码方式嵌入后的参数路径'''
resnet50_100 = './resnet50/encode_100.pth'
resnet50_010 = './resnet50/encode_010.pth'
resnet50_001 = './resnet50/encode_001.pth'
resnet50_111 = './resnet50/encode_111.pth'
densenet121_100 = './densenet121/encode_100.pth'
densenet121_010 = './densenet121/encode_010.pth'
densenet121_001 = './densenet121/encode_001.pth'
densenet121_111 = './densenet121/encode_111.pth'
convnext_100 = './convnext/encode_100.pth'
convnext_010 = './convnext/encode_010.pth'
convnext_001 = './convnext/encode_001.pth'
convnext_111 = './convnext/encode_111.pth'
paraPath = [convnext_100, convnext_010, convnext_001, convnext_111]




paramsInit = torch.load(convnextInitParaPath)
weightsInit = paramsInit["features.7.0.block.3.weight"].cpu().numpy()


# 定义计算MSE的函数
def calculate_mse(weights1, weights2):
    return np.mean((weights1 - weights2) ** 2)

# 定义计算KL散度的函数
def calculate_kl_divergence(weights1, weights2, bins=100):
    # 计算直方图，并归一化为概率分布
    hist1, bin_edges1 = np.histogram(weights1, bins=bins, density=True)
    hist2, bin_edges2 = np.histogram(weights2, bins=bins, density=True)

    # 避免分布中出现零
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10

    # 计算KL散度
    kl_div = entropy(hist1, hist2)
    return kl_div


# 计算区间概率，并传入相同的bin_edges
def calculate_interval_probabilities(weights, bins, bin_edges=None):
    if bin_edges is None:
        hist, bin_edges = np.histogram(weights, bins=bins, density=False)  # 计算实际频数
    else:
        hist, _ = np.histogram(weights, bins=bin_edges, density=False)  # 使用传入的bin_edges
    probabilities = hist / np.sum(hist)  # 将频数转换为概率
    intervals = [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges) - 1)]
    return intervals, probabilities, bin_edges


# 设定相同的bins数量
bins = 500

# 首先计算初始权重的直方图和区间
intervals_init, probs_init, bin_edges = calculate_interval_probabilities(weightsInit, bins=bins)
# 定义统一的y轴范围
y_max = max(probs_init) * 100  # 统一y轴最大值

for para in paraPath:
    print(para)
    figName = ''
    if para == resnet50_100:
        figName = 'resnet50 with encode_100'
    elif para == resnet50_010:
        figName = 'resnet50 with encode_010'
    elif para == resnet50_001:
        figName = 'resnet50 with encode_001'
    elif para == resnet50_111:
        figName = 'resnet50 with encode_111'
    elif para == densenet121_100:
        figName = 'densenet121 with encode_100'
    elif para == densenet121_010:
        figName = 'densenet121 with encode_010'
    elif para == densenet121_001:
        figName = 'densenet121 with encode_001'
    elif para == densenet121_111:
        figName = 'densenet121 with encode_111'
    elif para == convnext_100:
        figName = 'convnext with encode_100'
    elif para == convnext_010:
        figName = 'convnext with encode_010'
    elif para == convnext_001:
        figName = 'convnext with encode_001'
    elif para == convnext_111:
        figName = 'convnext with encode_111'


    params = torch.load(para)
    weightsPara = params["features.7.0.block.3.weight"].cpu().numpy()

    mse = calculate_mse(weightsInit, weightsPara)
    kl_div = calculate_kl_divergence(weightsInit, weightsPara)

    print("MSE: ", mse)
    print("KL: ", kl_div)

    # 使用相同的bin_edges计算编码后的权重的区间概率
    intervals_para, probs_para, _ = calculate_interval_probabilities(weightsPara, bins=bins, bin_edges=bin_edges)

    # 将概率作为柱状图绘制
    intervals_init_mid = [(i[0] + i[1]) / 2 for i in intervals_init]
    intervals_para_mid = [(i[0] + i[1]) / 2 for i in intervals_para]


    plt.figure(figsize=(8, 5))
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})  # 将所有字体放大为16
    plt.bar(intervals_init_mid, probs_init, width=0.01, alpha=0.7, color='#ff4da6',
            label='Initial Weights Probabilities')
    plt.bar(intervals_para_mid, probs_para, width=0.01, alpha=0.2, color='#66b2ff',
            label='Encoded Weights Probabilities')

    plt.xlim([-0.75, 0.75])  # 根据实际参数的范围调整
    plt.xlabel("Weight values")
    plt.ylim([1e-6, y_max])  # 统一y轴范围，并设置非线性刻度
    plt.yscale('log')  # 设置y轴为对数尺度
    plt.ylabel("Probability")
    plt.title(figName + " Weight Interval Probabilities")
    plt.legend()
    plt.grid()
    plt.savefig('./convnext/pics/' + figName + '.png', bbox_inches='tight')
    plt.show()
