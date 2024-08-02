import torch
import numpy as np
from viewPara import showParaStructure
from viewPara import showParaValue

resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'




def fcWeightsEmbedding(fcWeights, methods = 'LSB'):
    '''
    在全连接层的权重参数实现嵌入
    :param fcWeights: 需要进行嵌入的权重参数
    :param methods: 对权重参数进行嵌入的方法
    :return: 返回一个被嵌入有害信息的权重参数
    '''
    fcWeightsTensor = fcWeights.data
    match methods:
        case 'LSB':
            return
    return


def weightsEmbedding(pth_file):
    return


if __name__ == "__main__":
    # showParaStructure(resnet18InitParaPath)

    parameters = torch.load(vgg11InitParaPath)
    fcWeights = parameters["classifier.6.weight"]
    print(type(fcWeights))  # <class 'torch.nn.parameter.Parameter'>
    print(type(fcWeights.data))  # <class 'torch.Tensor'>
    fcWeightsTensor = fcWeights.data
    print(fcWeightsTensor.size()) # [1000, 512] ...
    print(fcWeightsTensor[0][0]) # tensor(-0.0185)


    dim0 = fcWeightsTensor.shape[0]
    dim1 = fcWeightsTensor.shape[1]
    print(fcWeightsTensor)
    int_view = fcWeightsTensor.view(torch.int32)
    print(int_view)

    # str_view = np.full((dim0, dim1), "", dtype=str)
    # for i in range(dim0):
    #     for j in range(dim1):
    #         # str_view[i][j] = format(int_view[i][j].item(), '032b')
    #         print(format(int_view[i][j].item(), '032b'))


    # 或运算，翻转最后一个比特位
    new_int_view = int_view ^ torch.tensor(11111111, dtype=torch.int32)



    new_float_view = new_int_view.view(torch.float32)
    print(new_float_view)
    print(type(new_float_view))


    # 写入pth文件
    parameters["classifier.6.weight"].data = new_float_view

    torch.save(parameters, "./embedding/vgg11_embedding_8_32.pth")
