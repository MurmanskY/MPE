import os
import torch
import random
import struct
import pandas as pd
from bitstring import BitArray
from fileProcess.fileProcess import split_file, merge_file

device = torch.device("mps")

'''原始pth文件'''
resnet18InitParaPath = '../init/resnet18-f37072fd.pth'
resnet50InitParaPath = '../init/resnet50-11ad3fa6.pth'
resnet101InitParaPath = '../init/resnet101-cd907fc2.pth'
vgg11InitParaPath = '../init/vgg11-8a719046.pth'
vgg13InitParaPath = '../init/vgg13-19584684.pth'
vgg16InitParaPath = '../init/vgg16-397923af.pth'
vgg19InitParaPath = '../init/vgg19-dcbb9e9d.pth'
'''用于存储bit翻转率的结果'''
outputFileRoot = './result/'

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




def flipBit(ch):
    """
    翻转比特
    :param ch:
    :return: ch
    """
    if ch == "0":
        return "1"
    else:
        return "0"


def encode(ch):
    """
    返回编码结果
    :param ch:
    :return:
    """
    return ch + flipBit(ch) + flipBit(ch)


def decode(str):
    """
    返回译码结果
    :param str: 待判断的字符串
    :return: 0/1字符串
    """
    bit0_num = 0
    bit1_num = 0
    if str[0] == "0":
        bit0_num += 1
    else:
        bit1_num += 1
    if flipBit(str[1]) == "0":
        bit0_num += 1
    else:
        bit1_num += 1
    if flipBit(str[2]) == "0":
        bit0_num += 1
    else:
        bit1_num += 1
    if bit0_num > bit1_num:
        return "0"
    else:
        return "1"


def float32_to_bits(f):
    """
    将 float32 数字转换为二进制位字符串表示
    :param f: 输入的 float32 数字
    :return: 32 位的二进制字符串
    """
    # 使用 struct.pack 将 float32 转换为字节，再转换为整数
    [int_repr] = struct.unpack('!I', struct.pack('!f', f))

    # 将整数转换为二进制字符串并去掉 '0b' 前缀，填充为32位
    return format(int_repr, '032b')

def getPthKeys(paraPath):
    """
    返回layers
    :param paraPath: 待获得的参数pth
    :return:
    """
    return torch.load(paraPath).keys()


def getBitFlipNum(str1, str2):
    """
    对比两个字符串，返回两个字符串中不同的比特数
    :param str1:
    :param str2:
    :return: 返回两个字符串中不同的比特数
    """
    ret = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            ret += 1
    return ret


def showBitFlip(initParaPath, retrainParaPath, bitStartIdx, bitEndIdx, outputFile):
    """
    记录在[bitStartIdx,bitEndIdx)之间有多少比特发生了翻转
    记录所有层的信息在结果文件中

    弃用，这个函数是单线程的函数，现在已经优化成了多线程的函数

    :param initPara:
    :param retrainPara:
    :param bitStartIdx:
    :param bitEndIdx:
    :param layer:
    :return:
    """
    data = {}  # 用于存储结果的字典
    initPara = torch.load(initParaPath, map_location=device)
    retrainPara = torch.load(retrainParaPath, map_location=device)
    for key in initPara.keys():  # 遍历所有键值
        if len(initPara[key].data.shape) < 1:
            continue  # 只比较参数二维及以上维度可能嵌入的层
        initLayerTensor = initPara[key].data.flatten()  # 将多维向量平铺
        retrainLayerTensor = retrainPara[key].data.flatten()
        if len(initLayerTensor) != len(retrainLayerTensor):
            continue  # 如果张量的形状不同，说明已经在最后一个全连接层，可以直接跳过
        paraNum = len(initLayerTensor)
        bitFlipNum = 0
        for idx in range(paraNum):
            initLayerEleStr = BitArray(int=initLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
            retrainLayerEleStr = BitArray(int=retrainLayerTensor[idx].view(torch.int32), length=32).bin[bitStartIdx: bitEndIdx]
            # print(initLayerEleStr, retrainLayerEleStr)
            bitFlipNum += getBitFlipNum(initLayerEleStr, retrainLayerEleStr)
        data[key] = bitFlipNum / (paraNum * (bitEndIdx - bitStartIdx))
        print(key, paraNum, data[key])
    df = pd.DataFrame([data])
    df.to_csv(outputFile, index=False)
    return


def layerFracBitFLip(initParaPath, flipParaPath, bit_n, *layers):
    """
    翻转pth的layers层fa的低n bit
    :param initParaPath: 原始参数pth
    :param flipParaPath: 翻转之后的参数pth
    :param bit_n: 翻转低多少bit
    :return: void
    """
    para = torch.load(initParaPath)

    for layer in layers:  # layers数组中的所有layer
        if len(para[layer].data.shape) < 1:
            continue  # 单值除去
        # print(layer, type(layer))
        layerTensor = para[layer].data
        # print(layerTensor.shape)
        layerTensor_initView = layerTensor.view(torch.int32)
        # print(format(layerTensor_initView[0][0][0][0], '032b'), layerTensor[0][0][0][0])
        layerTensor_embedded_int = layerTensor_initView ^ bit_n
        layerTensor_embedded = layerTensor_embedded_int.view(torch.float32)
        # print(format(layerTensor_embedded_int[0][0][0][0], '032b'), layerTensor_embedded[0][0][0][0])

        para[layer].data = layerTensor_embedded

    torch.save(para, flipParaPath)
    return


def layerExpBitFlip(initParaPath, flipParaPath, bit_n, *layers):
    """
    翻转layers的指数部分的低N bit
    :param initParaPath: 原始pth
    :param flipParaPath: 处理后pth
    :param bit_n: 指数的低 N bit
    :param layers:  处理的层
    :return:
    """

    def flip_exponent_bits(tensor, bit_n):
        """
        翻转float32的低nbit
        :param tensor:
        :param bit_n:
        :return:
        """
        # View tensor as int32 to manipulate bits
        int_view = tensor.view(torch.int32)
        # Extract exponent: bits 23-30 (8 bits)
        exponent_mask = 0x7F800000  # Exponent mask for 32-bit float
        exponent_bits = (int_view & exponent_mask) >> 23
        # Flip the lower `bit_n` bits of the exponent
        exponent_bits ^= (1 << bit_n) - 1
        # Put the flipped exponent back into the int representation
        int_view = (int_view & ~exponent_mask) | (exponent_bits << 23)
        # Convert back to float32
        return int_view.view(torch.float32)

    para = torch.load(initParaPath)
    for layer in layers:  # 所有layer
        if para[layer].data.dim() < 4:
            continue  # 只在卷积层进行嵌入
        layerTensor = para[layer].data
        para[layer].data = flip_exponent_bits(layerTensor, bit_n)

    torch.save(para, flipParaPath)
    return


def layerSignBitFlip(initParaPath, flipParaPath, *layers):
    """
    将模型的某些层的参数的符号位做翻转
    :param initParaPath: 原始pth
    :param flipParaPath:
    :param layers: 需要翻转的层
    :return:
    """
    def flip_sign_bit(tensor):
        """
        翻转 IEEE 754 32 位浮点数的符号位
        :param tensor: 输入 tensor
        :return: 符号位被翻转的 tensor
        """
        # 将 tensor 视为 int32 进行操作
        int_view = tensor.view(torch.int32)
        # 符号位的掩码：符号位是第 31 位
        sign_bit_mask = 0x80000000

        # 通过异或操作翻转符号位
        int_view ^= sign_bit_mask
        # 将 int32 视为 float32 返回
        return int_view.view(torch.float32)

    para = torch.load(initParaPath, map_location=torch.device("cpu"))
    for layer in layers:
        if para[layer].data.dim() < 2:
            continue # 只嵌入大于等于2维层的参数
        layerTensor = para[layer].data
        para[layer].data = flip_sign_bit(layerTensor)
    torch.save(para, flipParaPath)
    return


def generate_file_with_bits(file_path, num_bits):
    """
    根据需要多少bit，随机生成对应大小的恶意软件
    :param file_path:
    :param num_bits:
    :return:
    """
    # 计算需要的字节数，每字节有8个bit
    num_bytes = (num_bits + 7) // 8  # 向上取整，保证比特数足够
    print("Byte Num:", num_bytes)

    # 创建一个包含随机字节的字节数组
    byte_array = bytearray(random.getrandbits(8) for _ in range(num_bytes))

    # 如果不需要最后一个字节的全部位，将多余的位清零
    if num_bits % 8 != 0:
        last_byte_bits = num_bits % 8
        # 保留最后字节所需的位数，其它位清零
        mask = (1 << last_byte_bits) - 1
        byte_array[-1] &= mask

    # 将字节数组写入文件
    with open(file_path, 'wb') as f:
        f.write(byte_array)

    print(f"File '{file_path}' generated with {num_bits} bits.")


def generateFiles(malwares_path, malwaresSize_byte):
    """
    生成多个恶意软件
    :param malwares_path:
    :param malwaresSize_byte:
    :return:
    """
    for malware_path, malwareSize in zip(malwares_path, malwaresSize_byte):
        generate_file_with_bits(malware_path, malwareSize * 8)


def showDif(file1, file2):
    """
    对比提取的恶意软件和原始恶意软件的区别
    :return:
    """
    malwareStr1 = BitArray(filename=file1).bin
    malwareStr2 = BitArray(filename=file2).bin
    for i in range(len(malwareStr1)):
        if malwareStr1[i] != malwareStr2[i]:  # 打印出所有不同的bit的位置
            print("pos:", i, "initBit:", malwareStr1[i], "extractedBit:", malwareStr2[i])
    print(malwareStr1)
    print(malwareStr2)


def getExpEmbeddSize(initParaPath, layers, interval, correct):
    """
    返回指数部分最大的嵌入容量，单位是字节Byte
    :param initParaPath:
    :param layers: list
    :param interval: 每interval个中嵌入一个
    :return: list
    """
    para = torch.load(initParaPath, map_location=torch.device("mps"))
    ret = []
    for layer in layers:
        paraTensor = para[layer].data
        paraTensor_flat = paraTensor.flatten()
        layerSize = len(paraTensor_flat) // (interval * correct * 8)
        # print(layer, len(paraTensor_flat), layerSize)
        ret.append(layerSize)
    return ret


def layerExpBitEmbedd(initParaPath, flipParaPath, layers, malwares, interval, correct):
    """
    使用编码规则100和011进行嵌入低3bit,嵌入恶意软件，将卷积层展平，使用交错的方式编码
    :param initParaPath:
    :param flipParaPath:
    :param layers: layer list
    :param malwares: malware list
    :param interval: 每interval个中嵌入一个
    :param correct: 冗余多少
    :return:
    """
    para = torch.load(initParaPath, map_location=torch.device("mps"))

    for layer, malware in zip(layers, malwares):
        # 对于每个层

        malwareStr = BitArray(filename=malware).bin
        malwareLen = len(malwareStr)
        writePos = 0  # 恶意软件读写指针
        correctionPos = 0  # 冗余指针，标识目前处于第几个冗余
        currentWriteContent = malwareStr[writePos]  # 存储当前需要写入的内容

        paraTensor_flat = para[layer].flatten()

        while writePos < malwareLen:  # 写恶意软件中每一个bit
            while correctionPos < correct:  # 交错位置冗余
                index = writePos + correctionPos * interval * malwareLen
                paraTensor_flat_str = BitArray(int = paraTensor_flat[index].view(torch.int32), length=32).bin
                newParaTensor_flat_str = paraTensor_flat_str[:6] + encode(currentWriteContent) + paraTensor_flat_str[9:32]
                # 判断是否存在溢出
                if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                    newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                    paraTensor_flat[index] = newParaInt.view(torch.float32)
                else:
                    newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                    paraTensor_flat[index] = newParaInt.view(torch.float32)

                correctionPos += 1
                if correctionPos == correct:
                    correctionPos = 0
                    break
            writePos += 1
            if writePos >= malwareLen:
                break
            else:
                currentWriteContent = malwareStr[writePos]

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)

        torch.save(para, flipParaPath)
    return


def layerExpBitExtrac(initParaPath, layers, malwares_Extract, interval, correct):
    """
    在原来参数中提取出恶意软件
    :param initParaPath:
    :param layers: 层 list
    :param malwares_Extract: 恶意软件的保存路径，list
    :param interval: 每interval个中嵌入一个
    :param correct: 冗余个数
    :return:
    """
    para = torch.load(initParaPath, map_location=torch.device("mps"))
    layersEmbeddSize = getExpEmbeddSize(initParaPath, layers, interval, correct)  # 获取每一层最大的嵌入容量Byte，list

    for layer, layerEmbeddSize, malware_Extract in zip(layers, layersEmbeddSize, malwares_Extract):
        extractPos = 0  # 提取的字节数
        correctPos = 0  # 判断的几个冗余位置
        bit0_Num = 0  # 冗余位置有几个结果是0
        bit1_Num = 0  # 冗余位置有几个结果是1
        malware = []

        paraTensor = para[layer].data
        paraTensor_flat = paraTensor.flatten()
        malwareBitLen = layerEmbeddSize * 8
        while extractPos < malwareBitLen:
            while correctPos < correct:
                index = extractPos + malwareBitLen * interval * correctPos
                paraTensor_flat_str = BitArray(int=paraTensor_flat[index].view(torch.int32), length=32).bin
                bitData = decode(paraTensor_flat_str[6:9])
                '''判断3位冗余'''
                if bitData == '0':
                    bit0_Num += 1
                else:
                    bit1_Num += 1
                '''输出提取状态'''
                print("extractPos:", extractPos, "index:", index, "embeddedData:", paraTensor_flat_str[6:9],
                      "bitData:", bitData, "correctPos:", correctPos, "bit0_Num:", bit0_Num, "bit1_Num:", bit1_Num)
                correctPos += 1
                if correctPos == correct:
                    if bit0_Num > bit1_Num:
                        malware.append(BitArray(bin="0"))
                    else:
                        malware.append(BitArray(bin="1"))
                    correctPos = 0
                    bit0_Num = 0
                    bit1_Num = 0
                    break
            extractPos += 1
        merge_file(malware_Extract, malware)


    return


def func(pth1, pth2, *layers):
    para1 = torch.load(pth1)
    para2 = torch.load(pth2)
    for layer in layers:
        if len(para1[layer].data.shape) <= 1:
            continue  # 只比较参数二维及以上维度可能嵌入的层
        print(layer, para1[layer].data.shape)
        para1Tensor = para1[layer].data.flatten()
        para2Tensor = para2[layer].data.flatten()
        print(format(para1Tensor[0].view(torch.int32), '032b')[0:32])
        print(format(para2Tensor[0].view(torch.int32), '032b')[0:32], "\n")

    return


if __name__ == "__main__":
    # layerExpBitFlip(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_convFlip.pth", 3, *getPthKeys(resnet50InitParaPath))
    # func(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_flip.pth", *getPthKeys(resnet50InitParaPath))

    # layerExpBitEmbedd(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_flip.pth", 3, *getPthKeys(resnet50InitParaPath))

    layerSignBitFlip(resnet50InitParaPath, "./resnet50/bitFlip/sign_flip.pth", *getPthKeys(resnet50InitParaPath))
    func(resnet50InitParaPath, "./resnet50/bitFlip/sign_flip.pth", *getPthKeys(resnet50InitParaPath))





    """
    完成一整个流程
    1. 确定layers
    2. 根据layers随机生成malware存储在文件夹中
    3. 将malware嵌入到参数中
    4. 从参数中提取malware
    """
    layers = ["layer4.2.conv2.weight"]
    malwares = ["./malware/l1"]
    malwares_Extract = ["./malware/l1_extrac"]
    interval = 9
    correct = 11
    savePath = "./resnet50/bitEmbedd/temp.pth"

    # sizeList = getExpEmbeddSize(resnet50InitParaPath, layers, interval, correct)
    # generateFiles(malwares, sizeList)
    # layerExpBitEmbedd(resnet50InitParaPath, savePath, layers, malwares, interval, correct)
    # layerExpBitExtrac("./resnet50/2OxfordIIITPet/temp_ep_10.pth", layers, ["./malware/l1_extrac_re_Pet"], interval, correct)
    #
    # showDif("./malware/l1", "./malware/l1_extrac_re_Pet")
    print("Done")