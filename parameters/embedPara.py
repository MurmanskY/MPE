import sys
sys.path.append(r'./fileProcess/')
import torch
import numpy as np
from viewPara import showParaStructure
from viewPara import showParaValue
from bitstring import BitArray
from fileProcess.fileProcess import split_file, merge_file


'''for xor substitute testing'''
bit_1 = torch.tensor(1, dtype=torch.int32)
bit_2 = torch.tensor(3, dtype=torch.int32)
bit_3 = torch.tensor(7, dtype=torch.int32)
bit_4 = torch.tensor(15, dtype=torch.int32)
bit_5 = torch.tensor(31, dtype=torch.int32)
bit_6 = torch.tensor(63, dtype=torch.int32)
bit_7 = torch.tensor(127, dtype=torch.int32)
bit_8 = torch.tensor(255, dtype=torch.int32)
bit_9 = torch.tensor(511, dtype=torch.int32)
bit_10 = torch.tensor(1023, dtype=torch.int32)
bit_11 = torch.tensor(2047, dtype=torch.int32)
bit_12 = torch.tensor(4095, dtype=torch.int32)
bit_13 = torch.tensor(8191, dtype=torch.int32)
bit_14 = torch.tensor(16383, dtype=torch.int32)
bit_15 = torch.tensor(32767, dtype=torch.int32)
bit_16 = torch.tensor(65535, dtype=torch.int32)
bit_17 = torch.tensor(131071, dtype=torch.int32)
bit_18 = torch.tensor(262143, dtype=torch.int32)
bit_19 = torch.tensor(524287, dtype=torch.int32)
bit_20 = torch.tensor(1048575, dtype=torch.int32)
bit_21 = torch.tensor(2097151, dtype=torch.int32)
bit_22 = torch.tensor(4194303, dtype=torch.int32)
bit_23 = torch.tensor(8388607, dtype=torch.int32)
bit_24 = torch.tensor(16777215, dtype=torch.int32)
bit_25 = torch.tensor(33554431, dtype=torch.int32)
bit_26 = torch.tensor(67108863, dtype=torch.int32)
bit_27 = torch.tensor(134217727, dtype=torch.int32)
bit_28 = torch.tensor(268435455, dtype=torch.int32)
bit_29 = torch.tensor(536870911, dtype=torch.int32)
bit_30 = torch.tensor(1073741823, dtype=torch.int32)



resnet18InitParaPath = './init/resnet18-f37072fd.pth'
resnet50InitParaPath = './init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = './init/resnet101-cd907fc2.pth'
vgg11InitParaPath = './init/vgg11-8a719046.pth'
vgg13InitParaPath = './init/vgg13-19584684.pth'
vgg16InitParaPath = './init/vgg16-397923af.pth'
vgg19InitParaPath = './init/vgg19-dcbb9e9d.pth'

malware_path = "../malware/test2.jpg"




def fcWeightLowBitXOR(paraPath, bitReplacement, embeddedParaPath):
    '''
    在全连接层的权重参数 最低位取反，将取反后的参数pth文件存储
    :param paraPath: 需要进行嵌入的权重参数
    :param bitReplacement: 最低多少位进行取反
    :param embeddedParaPath: 被嵌入有害信息的pth文件
    :return: null
    '''
    para = torch.load(paraPath)

    # change model, change the following code
    fcWeightsTensor = para["classifier.6.weight"].data

    fcWeightsTensor_intView = fcWeightsTensor.view(torch.int32)
    fcWeightsTensor_embedded_int = fcWeightsTensor_intView ^ bitReplacement
    fcWeightsTensor_embedded = fcWeightsTensor_embedded_int.view(torch.float32)

    # change model, change the following code
    para["classifier.6.weight"].data = fcWeightsTensor_embedded

    torch.save(para, embeddedParaPath)
    return


def conv2WeightLowBitXOR(paraPath, bitReplacement, embeddedParaPath):
    '''
    卷积层weight参数 最低位取反，将取反后的参数pth文件存储
    :param paraPath: 需要进行嵌入的权重参数
    :param bitReplacement: 最低多少位进行取反
    :param embeddedParaPath: 被嵌入有害信息的pth文件
    :return: null
    '''
    para = torch.load(paraPath)

    # change model, change the following code
    conv2WeightTensor = para["features.0.weight"].data

    conv2WeightTensor_intView = conv2WeightTensor.view(torch.int32)
    print(format(conv2WeightTensor_intView[30][2][1][2], '032b'))
    conv2WeightTensor_embedded_int = conv2WeightTensor_intView ^ bitReplacement
    print(format(conv2WeightTensor_embedded_int[30][2][1][2], '032b'))
    conv2WeightTensor_embedded = conv2WeightTensor_embedded_int.view(torch.float32)

    # change model, change the following code
    para["features.0.weight"].data = conv2WeightTensor_embedded

    torch.save(para, embeddedParaPath)
    return


def allParaLowBitXor(paraPath, bitReplacement, embeddedParaPath):
    para = torch.load(paraPath)

    # change model, change the following code
    conv2WeightTensor = para["conv1.weight"].data

    conv2WeightTensor_intView = conv2WeightTensor.view(torch.int32)
    conv2WeightTensor_embedded_int = conv2WeightTensor_intView ^ bitReplacement
    conv2WeightTensor_embedded = conv2WeightTensor_embedded_int.view(torch.float32)

    # change model, change the following code
    para["conv1.weight"].data = conv2WeightTensor_embedded

    torch.save(para, embeddedParaPath)
    return


def fcWeightsLowBitEmbed(paraPath, chunkSize, malware, embeddedParaPath):
    '''
    在全连接层的权重参数 低x位，嵌入malware
    注意在处理int32类型的数据，可能会存在整型溢出的问题
    :param paraPath: 需要进行嵌入的权重参数
    :param chunkSize: 分片大小
    :param malware: 嵌入的有害信息
    :param embeddedParaPath: 被嵌入有害信息的pth文件
    :return:
    '''
    para = torch.load(paraPath)

    # change model, change the following code
    fcWeightsTensor = para["fc.weight"].data  # tensor类型的变量

    dim0 = fcWeightsTensor.shape[0]
    dim1 = fcWeightsTensor.shape[1]
    malwareBit = BitArray(filename=malware)
    malwareBitLen = len(malwareBit)
    paraNum = malwareBitLen // chunkSize #  有害信息满满占用的para个数
    remain = malwareBitLen % chunkSize  # 有害信息余下的bit位

    # 倒数四个参数，存储有害信息的长度信息
    paraNumStr = BitArray(uint=paraNum, length=32).bin  # 将paraNum转换成一个uint32，此时以string形式存储
    for i in range(4):
        # 将float32转换为32位的二进制字符串
        paraNumTempStr = BitArray(int=fcWeightsTensor[dim0-1][dim1-1-i].view(torch.int32), length=32).bin
        newParaNumStr = paraNumTempStr[:24] + paraNumStr[32-((i+1)*8): 32-(i*8)]
        # 判断是否存在int32溢出的情况
        if int(newParaNumStr, 2) >= 2**31:
            newParaNumInt = torch.tensor(int(newParaNumStr, 2)-2**32, dtype=torch.int32)
            fcWeightsTensor[dim0 - 1][dim1 - 1 - i] = newParaNumInt.view(torch.float32)
        else:
            newParaNumInt = torch.tensor(int(newParaNumStr, 2), dtype=torch.int32)
            fcWeightsTensor[dim0 - 1][dim1 - 1 - i] = newParaNumInt.view(torch.float32)

    # 参数倒数第五个
    remainStr = BitArray(uint=remain, length=8).bin  # 将remain转换成一个uint32，此时以string形式存储
    # 将float32转换为32位的二进制字符串
    remainTempStr = BitArray(int=fcWeightsTensor[dim0 - 1][dim1 - 5].view(torch.int32), length=32).bin
    newRemainStr = remainTempStr[:24] + remainStr
    # 判断是否存在int32溢出的情况
    if int(newRemainStr, 2) >= 2 ** 31:
        newRemainInt = torch.tensor(int(newRemainStr, 2) - 2 ** 32, dtype=torch.int32)
        fcWeightsTensor[dim0 - 1][dim1 - 5] = newRemainInt.view(torch.float32)
    else:
        newRemainInt = torch.tensor(int(newRemainStr, 2), dtype=torch.int32)
        fcWeightsTensor[dim0 - 1][dim1 - 5] = newRemainInt.view(torch.float32)

    # chunkSize倒数第六个
    chunkSizeStr = BitArray(uint=chunkSize, length=8).bin  # 将remain转换成一个uint32，此时以string形式存储
    # 将float32转换为32位的二进制字符串
    chunkSizeTempStr = BitArray(int=fcWeightsTensor[dim0 - 1][dim1 - 6].view(torch.int32), length=32).bin
    newChunkSizeStr = chunkSizeTempStr[:24] + chunkSizeStr
    # 判断是否存在int32溢出的情况
    if int(newChunkSizeStr, 2) >= 2 ** 31:
        newChunkSizeInt = torch.tensor(int(newChunkSizeStr, 2) - 2 ** 32, dtype=torch.int32)
        fcWeightsTensor[dim0 - 1][dim1 - 6] = newChunkSizeInt.view(torch.float32)
    else:
        newChunkSizeInt = torch.tensor(int(newChunkSizeStr, 2), dtype=torch.int32)
        fcWeightsTensor[dim0 - 1][dim1 - 6] = newChunkSizeInt.view(torch.float32)

    # 循环嵌入有害信息
    for i in range(paraNum):
        dim0Temp = i // dim1
        dim1Temp = i % dim1
        # print(dim0, dim1)
        # print(i, dim0Temp, dim1Temp)
        # 将需要嵌入float32类型的参数转换为32位的二进制字符串
        paraStr = BitArray(int=fcWeightsTensor[dim0Temp][dim1Temp].view(torch.int32), length=32).bin
        newParaStr = paraStr[:32-chunkSize] + malwareBit[i * chunkSize: (i + 1) * chunkSize].bin
        # 判断是否存在int32溢出的情况
        if int(newParaStr, 2) >= 2 ** 31:
            newParaInt = torch.tensor(int(newParaStr, 2) - 2 ** 32, dtype=torch.int32)
            fcWeightsTensor[dim0Temp][dim1Temp] = newParaInt.view(torch.float32)
        else:
            newParaInt = torch.tensor(int(newParaStr, 2), dtype=torch.int32)
            fcWeightsTensor[dim0Temp][dim1Temp] = newParaInt.view(torch.float32)

    # 如果还有剩余的bit需要嵌入，此时直接在weights的最低remain位嵌入有害信息
    if remain != 0:
        paraStr = BitArray(int=fcWeightsTensor[paraNum // dim1][paraNum % dim1].view(torch.int32), length=32).bin
        newParaStr = paraStr[:32 - remain] + malwareBit[-remain:].bin
        if int(newParaStr, 2) >= 2 ** 31:
            newParaInt = torch.tensor(int(newParaStr, 2) - 2 ** 32, dtype=torch.int32)
            fcWeightsTensor[paraNum // dim1][paraNum % dim1] = newParaInt.view(torch.float32)
        else:
            newParaInt = torch.tensor(int(newParaStr, 2), dtype=torch.int32)
            fcWeightsTensor[paraNum // dim1][paraNum % dim1] = newParaInt.view(torch.float32)

    # change model, change the following code
    para["fc.weight"].data = fcWeightsTensor

    torch.save(para, embeddedParaPath)
    return


def fcWeightsLowBitExtract(paraPath, extractedParaPath):
    """
    从paraPath对应路径下的pth中提取出恶意软件，报存在extractedParaPath中
    :param paraPath: 有害pth
    :param extractedParaPath: malware路径
    :return:
    """
    para = torch.load(paraPath)

    # change model, change the following code
    fcWeightsTensor = para["fc.weight"].data  # tensor类型的变量
    dim0 = fcWeightsTensor.shape[0]
    dim1 = fcWeightsTensor.shape[1]
    malware = []


    # 提取分片数
    chunkNumStr = str()
    for i in range(4):
        chunkNumStr += format(fcWeightsTensor[dim0-1][dim1-4+i].view(torch.uint32), '032b')[24:32]
    chunkNum = int(chunkNumStr, 2)
    print(chunkNum)

    # 提取余数
    remainStr = format(fcWeightsTensor[dim0-1][dim1-5].view(torch.uint32), '032b')[24: 32]
    remain = int(remainStr, 2)
    print(remain)

    # 提取分片大小
    chunkSizeStr = format(fcWeightsTensor[dim0-1][dim1-6].view(torch.uint32), '032b')[24: 32]
    chunkSize = int(chunkSizeStr, 2)
    print(chunkSize)

    # 提取信息
    for i in range(chunkNum):
        dim0Temp = i // dim1
        dim1Temp = i % dim1
        paraStr = format(fcWeightsTensor[dim0Temp][dim1Temp].view(torch.uint32), '032b')
        chunkStr = paraStr[-chunkSize:]
        malware.append(BitArray(bin=chunkStr))
    if remain != 0:
        paraStr = format(fcWeightsTensor[chunkNum // dim1][chunkNum % dim1].view(torch.uint32), '032b')
        chunkStr = paraStr[-remain:]
        malware.append(BitArray(bin=chunkStr))

    merge_file(extractedParaPath, malware)

    return


if __name__ == "__main__":
    # parameters = torch.load(resnet18InitParaPath)
    # fcWeightsTensor = parameters["fc.weight"].data
    #
    # dim0 = fcWeightsTensor.shape[0]
    # dim1 = fcWeightsTensor.shape[1]
    # int_view = fcWeightsTensor.view(torch.int32)
    # print(int_view)
    #
    # # str_view = np.full((dim0, dim1), "", dtype=str)
    # # for i in range(dim0):
    # #     for j in range(dim1):
    # #         # str_view[i][j] = format(int_view[i][j].item(), '032b')
    # #         print(format(int_view[i][j].item(), '032b'))
    #
    #
    # # 或运算，翻转最后一个比特位
    # new_int_view = int_view ^ torch.tensor(11111111, dtype=torch.int32)
    #
    #
    #
    # new_float_view = new_int_view.view(torch.float32)
    # print(new_float_view)
    # print(type(new_float_view))
    #
    #
    # # 写入pth文件
    # parameters["classifier.6.weight"].data = new_float_view
    #
    # torch.save(parameters, "./weightsEmbedding/vgg19_embedding_8_32.pth")


    # """查看原来函数是不是可行的"""
    # parameters = torch.load(resnet18InitParaPath)
    # parameters_8 = torch.load("./weightsEmbedding/resnet18_embedding_8_32.pth")
    # parameters_16 = torch.load("./weightsEmbedding/resnet18_embedding_16_32.pth")
    # parameters_20 = torch.load("./weightsEmbedding/resnet18_embedding_20_32.pth")
    #
    # fcWeightsTensor = parameters["fc.weight"].data
    # fcWeightsTensor_8 = parameters_8["fc.weight"].data
    # fcWeightsTensor_16 = parameters_16["fc.weight"].data
    # fcWeightsTensor_20 = parameters_20["fc.weight"].data
    #
    # int_view = fcWeightsTensor.view(torch.int32)
    # int_view_8 = fcWeightsTensor_8.view(torch.int32)
    # int_view_16 = fcWeightsTensor_16.view(torch.int32)
    # int_view_20 = fcWeightsTensor_20.view(torch.int32)
    #
    # print(fcWeightsTensor)
    # print(fcWeightsTensor_8)
    # print(fcWeightsTensor_16)
    # print(fcWeightsTensor_20)
    #
    # print(format(int_view[0][0].item(), '032b'))
    # print(format(int_view_8[0][0].item(), '032b'))
    # print(format(int_view_16[0][0].item(), '032b'))
    # print(format(int_view_20[0][0].item(), '032b'))
    # print("\n\n")


    # """验证此方法的可行性"""
    # a = torch.tensor(-7.45, dtype=torch.float32)
    # a_int32 = a.view(torch.int32) # 类型也完成了转换
    # b = torch.tensor(3, dtype=torch.int32)
    # b_int32 = b.view(torch.int32) # 类型也完成了转换
    # info = torch.bitwise_xor(a_int32, b_int32)
    # print(format(a_int32.item(), '032b'))
    # print(format(b_int32.item(), '032b'))
    # print(format(info.item(), '032b'))
    #
    # print(info.view(torch.float32))


    # '''Test bit replacement data'''
    # print(format(bit_1, '032b'))
    # print(format(bit_2, '032b'))
    # print(format(bit_3, '032b'))
    # print(format(bit_4, '032b'))
    # print(format(bit_5, '032b'))
    # print(format(bit_6, '032b'))
    # print(format(bit_7, '032b'))
    # print(format(bit_8, '032b'))
    # print(format(bit_9, '032b'))
    # print(format(bit_10, '032b'))
    # print(format(bit_11, '032b'))
    # print(format(bit_12, '032b'))
    # print(format(bit_13, '032b'))
    # print(format(bit_14, '032b'))
    # print(format(bit_15, '032b'))
    # print(format(bit_16, '032b'))
    # print(format(bit_17, '032b'))
    # print(format(bit_18, '032b'))
    # print(format(bit_19, '032b'))
    # print(format(bit_20, '032b'))
    # print(format(bit_21, '032b'))
    # print(format(bit_22, '032b'))
    # print(format(bit_23, '032b'))
    # print(format(bit_24, '032b'))
    # print(format(bit_25, '032b'))
    # print(format(bit_26, '032b'))
    # print(format(bit_27, '032b'))
    # print(format(bit_28, '032b'))
    # print(format(bit_29, '032b'))
    # print(format(bit_30, '032b'))
    # print("\n\n")

    # fcWeightLowBitXOR(vgg19InitParaPath, bit_24, "./weightsEmbedding/vgg19_embedding_24_32.pth")

    conv2WeightLowBitXOR(vgg19InitParaPath, bit_24, "./convEmbedding/vgg19_embedding_24_32.pth")

    # chunkSize = 20
    # fcWeightsLowBitEmbed(resnet50InitParaPath, chunkSize, malware_path,
    #                      "./weightsEmbedding/resnet50_" + str(chunkSize) + "_test2.pth")
    # fcWeightsLowBitExtract("./weightsEmbedding/resnet50_" + str(chunkSize) + "_test2.pth", "../malware/test2_extract.jpeg")


    print("done")
