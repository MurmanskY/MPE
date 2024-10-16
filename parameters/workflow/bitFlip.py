import os
import torch
import random
import struct
import pandas as pd
from bitstring import BitArray
from fileProcess.fileProcess import split_file, merge_file
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math


device = torch.device("mps")

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
convnext_largeInitParaPath = '../init/convnext_large-ea097f82.pth'
googlenetInitParaPath = '../init/googlenet-1378be20.pth'
inceptionV3InitParaPath = '../init/inception_v3_google-0cc3c7bd.pth'
vitb16InitParaPath = '../init/vit_b_16-c867db91.pth'
vith14_lcswagInitParaPath = '../init/vit_h_14_lc_swag-c1eb923e.pth'
swinv2bInitParaPath = '../init/swin_v2_b-781e5279.pth'


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




def SNR(total_mal_num, total_error_num):
    return 20 * math.log(8 * total_mal_num / total_error_num, 10)


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
    return torch.load(paraPath, map_location=torch.device("mps")).keys()


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
        para[layer].data = -layerTensor
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
    对比提取的恶意软件和原始恶意软件的区别,返回出错的bit
    :return:
    """
    malwareStr1 = BitArray(filename=file1).bin
    malwareStr2 = BitArray(filename=file2).bin
    diffNum = 0
    for i in range(len(malwareStr1)):
        if malwareStr1[i] != malwareStr2[i]:  # 打印出所有不同的bit的位置
            # print("pos:", i, "initBit:", malwareStr1[i], "extractedBit:", malwareStr2[i])
            diffNum += 1
    # print(malwareStr1)
    # print(malwareStr2)
    print("different bit Num between the two files: ", diffNum)
    return diffNum


def getExpEmbeddSize(initParaPath, layers, interval, correct):
    """
    返回指数部分最大的嵌入容量，单位是字节Byte
    :param initParaPath:
    :param layers: list
    :param interval: 每interval个中嵌入一个
    :return: list
    """
    para = torch.load(initParaPath, map_location=torch.device("cpu"))
    ret = []
    for layer in layers:
        paraTensor = para[layer].data
        paraTensor_flat = paraTensor.flatten()
        # print(initParaPath, layers, paraTensor_flat.size())
        layerSize = len(paraTensor_flat) // (interval * correct * 8)
        # print(layer, len(paraTensor_flat), layerSize)
        ret.append(layerSize)
    return ret


# def layerExpBitEmbedd(initParaPath, flipParaPath, layers, malwares, interval, correct):
#     """
#     单线程
#     使用编码规则100和011进行嵌入低3bit,嵌入恶意软件，将卷积层展平，使用交错的方式编码
#     :param initParaPath:
#     :param flipParaPath:
#     :param layers: layer list
#     :param malwares: malware list
#     :param interval: 每interval个中嵌入一个
#     :param correct: 冗余多少
#     :return:
#     """
#     para = torch.load(initParaPath, map_location=torch.device("mps"))
#
#     for layer, malware in zip(layers, malwares):
#         # 对于每个层
#
#         malwareStr = BitArray(filename=malware).bin
#         malwareLen = len(malwareStr)
#         writePos = 0  # 恶意软件读写指针
#         correctionPos = 0  # 冗余指针，标识目前处于第几个冗余
#         currentWriteContent = malwareStr[writePos]  # 存储当前需要写入的内容
#
#         paraTensor_flat = para[layer].flatten()
#
#         while writePos < malwareLen:  # 写恶意软件中每一个bit
#             while correctionPos < correct:  # 交错位置冗余
#                 index = writePos + correctionPos * interval * malwareLen
#                 paraTensor_flat_str = BitArray(int = paraTensor_flat[index].view(torch.int32), length=32).bin
#                 newParaTensor_flat_str = paraTensor_flat_str[:6] + encode(currentWriteContent) + paraTensor_flat_str[9:32]
#                 # 判断是否存在溢出
#                 if int(newParaTensor_flat_str, 2) >= 2 ** 31:
#                     newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
#                     paraTensor_flat[index] = newParaInt.view(torch.float32)
#                 else:
#                     newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
#                     paraTensor_flat[index] = newParaInt.view(torch.float32)
#
#                 correctionPos += 1
#                 if correctionPos == correct:
#                     correctionPos = 0
#                     break
#             writePos += 1
#             if writePos >= malwareLen:
#                 break
#             else:
#                 currentWriteContent = malwareStr[writePos]
#
#         para[layer] = paraTensor_flat.reshape(para[layer].data.shape)
#
#         torch.save(para, flipParaPath)
#     return

def layerExpBitEmbedd(initParaPath, flipParaPath, layers, malwares, interval, correct, num_threads=8):
    para = torch.load(initParaPath, map_location=torch.device("cpu"))

    for layer, malware in zip(layers, malwares):
        malwareStr = BitArray(filename=malware).bin
        paraTensor_flat = para[layer].flatten()
        layer_size = paraTensor_flat.size(0)

        # 生成嵌入的索引
        indices = list(range(0, layer_size, interval))
        num_positions_total = len(indices)
        num_bits_total = num_positions_total // correct

        if len(malwareStr) > num_bits_total:
            raise ValueError("恶意数据太大，无法嵌入此层。")

        # 计算每个线程应处理的总位数
        total_bits = len(malwareStr)
        base_bits_per_thread = total_bits // num_threads
        extra_bits = total_bits % num_threads

        bits_per_thread = [base_bits_per_thread] * num_threads
        for i in range(extra_bits):
            bits_per_thread[i] += 1  # 将剩余的位分配给前面的线程

        # 为每个线程分配位置和恶意数据的切片
        malware_start = 0
        position_start = 0
        futures = {}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for i in range(num_threads):
                bits_this_thread = bits_per_thread[i]
                malware_end = malware_start + bits_this_thread
                malware_slice = malwareStr[malware_start:malware_end]

                positions_needed = bits_this_thread * correct
                positions = indices[position_start:position_start + positions_needed]

                futures[i] = executor.submit(process_block_embedding_thread, paraTensor_flat, malware_slice, correct, positions)

                malware_start = malware_end
                position_start += positions_needed

            for i in futures:
                futures[i].result()

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)
    torch.save(para, flipParaPath)

def process_block_embedding_thread(paraTensor_flat, malwareStr, correct, positions):
    malwareLen = len(malwareStr)
    pos_idx = 0
    for malware_idx in range(malwareLen):
        current_bit = malwareStr[malware_idx]
        for _ in range(correct):
            idx = positions[pos_idx]
            paraTensor_flat_str = BitArray(int=paraTensor_flat[idx].view(torch.int32), length=32).bin
            newParaTensor_flat_str = paraTensor_flat_str[:6] + encode(current_bit) + paraTensor_flat_str[9:]
            if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            else:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            pos_idx += 1









# def layerExpBitExtrac(initParaPath, layers, malwares_Extract, interval, correct):
#     """
#     在原来参数中提取出恶意软件
#     :param initParaPath:
#     :param layers: 层 list
#     :param malwares_Extract: 恶意软件的保存路径，list
#     :param interval: 每interval个中嵌入一个
#     :param correct: 冗余个数
#     :return:
#     """
#     para = torch.load(initParaPath, map_location=torch.device("mps"))
#     layersEmbeddSize = getExpEmbeddSize(initParaPath, layers, interval, correct)  # 获取每一层最大的嵌入容量Byte，list
#
#     for layer, layerEmbeddSize, malware_Extract in zip(layers, layersEmbeddSize, malwares_Extract):
#         extractPos = 0  # 提取的字节数
#         correctPos = 0  # 判断的几个冗余位置
#         bit0_Num = 0  # 冗余位置有几个结果是0
#         bit1_Num = 0  # 冗余位置有几个结果是1
#         malware = []
#
#         paraTensor = para[layer].data
#         paraTensor_flat = paraTensor.flatten()
#         malwareBitLen = layerEmbeddSize * 8
#         while extractPos < malwareBitLen:
#             while correctPos < correct:
#                 index = extractPos + malwareBitLen * interval * correctPos
#                 paraTensor_flat_str = BitArray(int=paraTensor_flat[index].view(torch.int32), length=32).bin
#                 bitData = decode(paraTensor_flat_str[6:9])
#                 '''判断3位冗余'''
#                 if bitData == '0':
#                     bit0_Num += 1
#                 else:
#                     bit1_Num += 1
#                 '''输出提取状态'''
#                 print("extractPos:", extractPos, "index:", index, "embeddedData:", paraTensor_flat_str[6:9],
#                       "bitData:", bitData, "correctPos:", correctPos, "bit0_Num:", bit0_Num, "bit1_Num:", bit1_Num)
#                 correctPos += 1
#                 if correctPos == correct:
#                     if bit0_Num > bit1_Num:
#                         malware.append(BitArray(bin="0"))
#                     else:
#                         malware.append(BitArray(bin="1"))
#                     correctPos = 0
#                     bit0_Num = 0
#                     bit1_Num = 0
#                     break
#             extractPos += 1
#         merge_file(malware_Extract, malware)
#
#
#     return


def layerExpBitExtrac(initParaPath, layers, malwares_Extract, interval, correct, num_threads=8):
    para = torch.load(initParaPath, map_location=torch.device("cpu"))
    layersEmbeddSize = getExpEmbeddSize(initParaPath, layers, interval, correct)

    for layer, layerEmbeddSize, malware_Extract in zip(layers, layersEmbeddSize, malwares_Extract):
        paraTensor = para[layer].data
        paraTensor_flat = paraTensor.flatten()
        layer_size = paraTensor_flat.size(0)

        indices = list(range(0, layer_size, interval))
        num_positions_total = len(indices)
        malwareBitLen = layerEmbeddSize * 8

        # 确保我们只处理嵌入了数据的位置
        indices = indices[:malwareBitLen * correct]

        # 在线程间划分位置
        positions_per_thread = np.array_split(indices, num_threads)

        all_bits_parts = [None] * num_threads

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {}
            for i in range(num_threads):
                positions = positions_per_thread[i].tolist()
                futures[i] = executor.submit(process_block_extraction_thread, paraTensor_flat, positions, correct)

            for i in sorted(futures.keys()):
                all_bits_parts[i] = futures[i].result()

        # 按顺序组合位
        all_bits = []
        for bits in all_bits_parts:
            all_bits.extend(bits)

        # 将提取的二进制数据合并到文件中
        merge_file(malware_Extract, [BitArray(bin="".join(all_bits))])

def process_block_extraction_thread(paraTensor_flat, positions, correct):
    malware_bits = []
    num_positions = len(positions)
    pos_idx = 0

    while pos_idx + correct <= num_positions:
        bit0_Num = 0
        bit1_Num = 0
        for _ in range(correct):
            idx = positions[pos_idx]
            paraTensor_flat_str = BitArray(int=paraTensor_flat[idx].view(torch.int32), length=32).bin
            bitData = decode(paraTensor_flat_str[6:9])
            if bitData == '0':
                bit0_Num += 1
            else:
                bit1_Num += 1
            pos_idx += 1
        if bit0_Num > bit1_Num:
            malware_bits.append('0')
        else:
            malware_bits.append('1')
    return malware_bits




def func(pth1, pth2, *layers):
    para1 = torch.load(pth1)
    para2 = torch.load(pth2)
    for layer in layers:
        if len(para1[layer].data.shape) <= 1:
            continue  # 只比较参数二维及以上维度可能嵌入的层
        print(layer, para1[layer].data.shape)
        para1Tensor = para1[layer].data.flatten()
        para2Tensor = para2[layer].data.flatten()
        print(para1Tensor[0], format(para1Tensor[0].view(torch.int32), '032b')[0:32])
        print(para2Tensor[0], format(para2Tensor[0].view(torch.int32), '032b')[0:32], "\n")

    return


if __name__ == "__main__":
    # layerExpBitFlip(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_convFlip.pth", 3, *getPthKeys(resnet50InitParaPath))
    # func(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_flip.pth", *getPthKeys(resnet50InitParaPath))

    # layerExpBitEmbedd(resnet50InitParaPath, "./resnet50/bitFlip/exp_3_flip.pth", 3, *getPthKeys(resnet50InitParaPath))

    # layerSignBitFlip(resnet50InitParaPath, "./resnet50/bitFlip/sign_flip.pth", *getPthKeys(resnet50InitParaPath))
    # func(resnet50InitParaPath, "./resnet50/bitFlip/sign_flip.pth", *getPthKeys(resnet50InitParaPath))

    '''翻转resnet101'''
    # layerFracBitFLip(resnet101InitParaPath, "./resnet101/bitFlip/frac_1.pth", 1, *getPthKeys(resnet101InitParaPath))
    # layerFracBitFLip(resnet101InitParaPath, "./resnet101/bitFlip/frac_8.pth", 8, *getPthKeys(resnet101InitParaPath))
    # layerFracBitFLip(resnet101InitParaPath, "./resnet101/bitFlip/frac_16.pth", 16, *getPthKeys(resnet101InitParaPath))
    # layerFracBitFLip(resnet101InitParaPath, "./resnet101/bitFlip/frac_23.pth", 23, *getPthKeys(resnet101InitParaPath))
    # layerExpBitFlip(resnet101InitParaPath, "./resnet101/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(resnet101InitParaPath))
    # layerExpBitFlip(resnet101InitParaPath, "./resnet101/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(resnet101InitParaPath))

    '''翻转vgg16'''
    # layerFracBitFLip(vgg16InitParaPath, "./vgg16/bitFlip/frac_1.pth", 1, *getPthKeys(vgg16InitParaPath))
    # layerFracBitFLip(vgg16InitParaPath, "./vgg16/bitFlip/frac_8.pth", 8, *getPthKeys(vgg16InitParaPath))
    # layerFracBitFLip(vgg16InitParaPath, "./vgg16/bitFlip/frac_16.pth", 16, *getPthKeys(vgg16InitParaPath))
    # layerFracBitFLip(vgg16InitParaPath, "./vgg16/bitFlip/frac_23.pth", 23, *getPthKeys(vgg16InitParaPath))
    # layerExpBitFlip(vgg16InitParaPath, "./vgg16/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(vgg16InitParaPath))
    # layerExpBitFlip(vgg16InitParaPath, "./vgg16/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(vgg16InitParaPath))

    '''翻转vgg19'''
    # layerFracBitFLip(vgg19InitParaPath, "./vgg19/bitFlip/frac_1.pth", 1, *getPthKeys(vgg19InitParaPath))
    # layerFracBitFLip(vgg19InitParaPath, "./vgg19/bitFlip/frac_8.pth", 8, *getPthKeys(vgg19InitParaPath))
    # layerFracBitFLip(vgg19InitParaPath, "./vgg19/bitFlip/frac_16.pth", 16, *getPthKeys(vgg19InitParaPath))
    # layerFracBitFLip(vgg19InitParaPath, "./vgg19/bitFlip/frac_23.pth", 23, *getPthKeys(vgg19InitParaPath))
    # layerExpBitFlip(vgg19InitParaPath, "./vgg19/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(vgg19InitParaPath))
    # layerExpBitFlip(vgg19InitParaPath, "./vgg19/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(vgg19InitParaPath))

    '''翻转vgg16bn'''
    # layerFracBitFLip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/frac_1.pth", 1, *getPthKeys(vgg16BNInitParaPath))
    # layerFracBitFLip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/frac_8.pth", 8, *getPthKeys(vgg16BNInitParaPath))
    # layerFracBitFLip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/frac_16.pth", 16, *getPthKeys(vgg16BNInitParaPath))
    # layerFracBitFLip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/frac_23.pth", 23, *getPthKeys(vgg16BNInitParaPath))
    # layerExpBitFlip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(vgg16BNInitParaPath))
    # layerExpBitFlip(vgg16BNInitParaPath, "./vgg16bn/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(vgg16BNInitParaPath))

    '''翻转vgg19bn'''
    # layerFracBitFLip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/frac_1.pth", 1, *getPthKeys(vgg19BNInitParaPath))
    # layerFracBitFLip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/frac_8.pth", 8, *getPthKeys(vgg19BNInitParaPath))
    # layerFracBitFLip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/frac_16.pth", 16, *getPthKeys(vgg19BNInitParaPath))
    # layerFracBitFLip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/frac_23.pth", 23, *getPthKeys(vgg19BNInitParaPath))
    # layerExpBitFlip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(vgg19BNInitParaPath))
    # layerExpBitFlip(vgg19BNInitParaPath, "./vgg19bn/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(vgg19BNInitParaPath))

    '''翻转alexnet'''
    # layerFracBitFLip(alexnetInitParaPath, "./alexnet/bitFlip/frac_1.pth", 1, *getPthKeys(alexnetInitParaPath))
    # layerFracBitFLip(alexnetInitParaPath, "./alexnet/bitFlip/frac_8.pth", 8, *getPthKeys(alexnetInitParaPath))
    # layerFracBitFLip(alexnetInitParaPath, "./alexnet/bitFlip/frac_16.pth", 16, *getPthKeys(alexnetInitParaPath))
    # layerFracBitFLip(alexnetInitParaPath, "./alexnet/bitFlip/frac_23.pth", 23, *getPthKeys(alexnetInitParaPath))
    # layerExpBitFlip(alexnetInitParaPath, "./alexnet/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(alexnetInitParaPath))
    # layerExpBitFlip(alexnetInitParaPath, "./alexnet/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(alexnetInitParaPath))

    '''翻转convnext'''
    # layerFracBitFLip(convnextInitParaPath, "./convnext/bitFlip/frac_1.pth", 1, *getPthKeys(convnextInitParaPath))
    # layerFracBitFLip(convnextInitParaPath, "./convnext/bitFlip/frac_8.pth", 8, *getPthKeys(convnextInitParaPath))
    # layerFracBitFLip(convnextInitParaPath, "./convnext/bitFlip/frac_16.pth", 16, *getPthKeys(convnextInitParaPath))
    # layerFracBitFLip(convnextInitParaPath, "./convnext/bitFlip/frac_23.pth", 23, *getPthKeys(convnextInitParaPath))
    # layerExpBitFlip(convnextInitParaPath, "./convnext/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(convnextInitParaPath))
    # layerExpBitFlip(convnextInitParaPath, "./convnext/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(convnextInitParaPath))

    '''翻转googlenet'''
    # layerFracBitFLip(googlenetInitParaPath, "./googlenet/bitFlip/frac_1.pth", 1, *getPthKeys(googlenetInitParaPath))
    # layerFracBitFLip(googlenetInitParaPath, "./googlenet/bitFlip/frac_8.pth", 8, *getPthKeys(googlenetInitParaPath))
    # layerFracBitFLip(googlenetInitParaPath, "./googlenet/bitFlip/frac_16.pth", 16, *getPthKeys(googlenetInitParaPath))
    # layerFracBitFLip(googlenetInitParaPath, "./googlenet/bitFlip/frac_23.pth", 23, *getPthKeys(googlenetInitParaPath))
    # layerExpBitFlip(googlenetInitParaPath, "./googlenet/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(googlenetInitParaPath))
    # layerExpBitFlip(googlenetInitParaPath, "./googlenet/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(googlenetInitParaPath))

    '''翻转inceptionV3'''
    # layerFracBitFLip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/frac_1.pth", 1, *getPthKeys(inceptionV3InitParaPath))
    # layerFracBitFLip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/frac_8.pth", 8, *getPthKeys(inceptionV3InitParaPath))
    # layerFracBitFLip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/frac_16.pth", 16, *getPthKeys(inceptionV3InitParaPath))
    # layerFracBitFLip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/frac_23.pth", 23, *getPthKeys(inceptionV3InitParaPath))
    # layerExpBitFlip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(inceptionV3InitParaPath))
    # layerExpBitFlip(inceptionV3InitParaPath, "./inceptionV3/bitFlip/exp_3_convFlip.pth", 3,*getPthKeys(inceptionV3InitParaPath))

    '''翻转vitb16'''
    # layerFracBitFLip(vitb16InitParaPath, "./vitb16/bitFlip/frac_1.pth", 1, *getPthKeys(vitb16InitParaPath))
    # layerFracBitFLip(vitb16InitParaPath, "./vitb16/bitFlip/frac_8.pth", 8, *getPthKeys(vitb16InitParaPath))
    # layerFracBitFLip(vitb16InitParaPath, "./vitb16/bitFlip/frac_16.pth", 16, *getPthKeys(vitb16InitParaPath))
    # layerFracBitFLip(vitb16InitParaPath, "./vitb16/bitFlip/frac_23.pth", 23, *getPthKeys(vitb16InitParaPath))
    # layerExpBitFlip(vitb16InitParaPath, "./vitb16/bitFlip/exp_3_allFlip.pth", 3, *getPthKeys(vitb16InitParaPath))
    # layerExpBitFlip(vitb16InitParaPath, "./vitb16/bitFlip/exp_3_convFlip.pth", 3, *getPthKeys(vitb16InitParaPath))



    # """
    # 完成一整个流程
    # 1. 确定layers
    # 2. 根据layers随机生成malware存储在文件夹中
    # 3. 将malware嵌入到参数中
    # 4. 从参数中提取malware
    # """
    # layers = ["layer4.2.conv2.weight"]
    # malwares = ["./malware/l1"]
    # malwares_Extract = ["./malware/l1_extrac"]
    # interval = 9
    # correct = 11
    # savePath = "./resnet50/bitEmbedd/temp.pth"
    #
    # # sizeList = getExpEmbeddSize(resnet50InitParaPath, layers, interval, correct)
    # # generateFiles(malwares, sizeList)
    # # layerExpBitEmbedd(resnet50InitParaPath, savePath, layers, malwares, interval, correct)
    # # layerExpBitExtrac("./resnet50/2OxfordIIITPet/temp_ep_10.pth", layers, ["./malware/l1_extrac_re_Pet"], interval, correct)
    # #
    # # showDif("./malware/l1", "./malware/l1_extrac_re_Pet")
    # print("Done")


    """
    20240914 流程对比实验
    resnet50
    layer4.0.conv2.weight
    layer4.1.conv2.weight
    使用模型resnet的时候需要换成单线程的嵌入和提取函数
    """
    # layers = ["layer4.0.conv2.weight",
    #           "layer4.1.conv2.weight"]
    # malwares = ["./malware/resnet50_l1",
    #            "./malware/resnet50_l2"]
    # malwares_extract = ["./malware/resnet50_l1_extract",
    #                    "./malware/resnet50_l2_extract"]
    # interval = 9
    # correct = 11
    # savePath = "./resnet50/bitEmbedd/resnet50_2layers_9inter_11corr.pth"
    #
    #
    # sizeList = getExpEmbeddSize(resnet50InitParaPath, layers, interval, correct)
    # # # generateFiles(malwares, sizeList)
    # # # layerExpBitEmbedd(resnet50InitParaPath, savePath, layers, malwares, interval, correct, num_threads=1)
    #
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_2_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_4_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_6_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_8_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_10_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_12_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_14_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_16_ep_20.pth", layers, malwares_extract, interval, correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_18_ep_10.pth", layers, malwares_extract, interval,
    #                   correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./resnet50/2PCAM/resnet50_2layers_9inter_11corr_ep_20_ep_10.pth", layers, malwares_extract, interval,
    #                   correct, num_threads=1)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))





















    """
    20240916 流程对比实验
    convnext_base
    features.7.0.block.0.weight
    features.7.1.block.0.weight
    features.7.2.block.0.weight
    """
    # layers = ["features.5.18.block.3.weight",
    #           "features.5.18.block.5.weight",
    #           "features.5.19.block.3.weight",
    #           "features.5.19.block.5.weight",
    #           "features.5.20.block.3.weight",
    #           "features.5.20.block.5.weight",
    #           "features.5.21.block.3.weight",
    #           "features.5.21.block.5.weight",
    #           "features.5.22.block.3.weight",
    #           "features.5.22.block.5.weight",
    #           "features.5.23.block.3.weight",
    #           "features.5.23.block.5.weight",
    #           "features.5.24.block.3.weight",
    #           "features.5.24.block.5.weight",
    #           "features.5.25.block.3.weight",
    #           "features.5.25.block.5.weight",
    #           "features.5.26.block.3.weight",
    #           "features.5.26.block.5.weight",
    #           "features.7.0.block.3.weight",
    #           "features.7.0.block.5.weight",
    #           "features.7.1.block.3.weight",
    #           "features.7.1.block.5.weight",
    #           "features.7.2.block.3.weight",
    #           "features.7.2.block.5.weight"]
    # malwares = ["./malware/convnext_base_l1",
    #             "./malware/convnext_base_l2",
    #             "./malware/convnext_base_l3",
    #             "./malware/convnext_base_l4",
    #             "./malware/convnext_base_l5",
    #             "./malware/convnext_base_l6",
    #             "./malware/convnext_base_l7",
    #             "./malware/convnext_base_l8",
    #             "./malware/convnext_base_l9",
    #             "./malware/convnext_base_l10",
    #             "./malware/convnext_base_l11",
    #             "./malware/convnext_base_l12",
    #             "./malware/convnext_base_l13",
    #             "./malware/convnext_base_l14",
    #             "./malware/convnext_base_l15",
    #             "./malware/convnext_base_l16",
    #             "./malware/convnext_base_l17",
    #             "./malware/convnext_base_l18",
    #             "./malware/convnext_base_l19",
    #             "./malware/convnext_base_l20",
    #             "./malware/convnext_base_l21",
    #             "./malware/convnext_base_l22",
    #             "./malware/convnext_base_l23",
    #             "./malware/convnext_base_l24"]
    # malwares_extract = ["./malware/convnext_base_l1_extract",
    #                     "./malware/convnext_base_l2_extract",
    #                     "./malware/convnext_base_l3_extract",
    #                     "./malware/convnext_base_l4_extract",
    #                     "./malware/convnext_base_l5_extract",
    #                     "./malware/convnext_base_l6_extract",
    #                     "./malware/convnext_base_l7_extract",
    #                     "./malware/convnext_base_l8_extract",
    #                     "./malware/convnext_base_l9_extract",
    #                     "./malware/convnext_base_l10_extract",
    #                     "./malware/convnext_base_l11_extract",
    #                     "./malware/convnext_base_l12_extract",
    #                     "./malware/convnext_base_l13_extract",
    #                     "./malware/convnext_base_l14_extract",
    #                     "./malware/convnext_base_l15_extract",
    #                     "./malware/convnext_base_l16_extract",
    #                     "./malware/convnext_base_l17_extract",
    #                     "./malware/convnext_base_l18_extract",
    #                     "./malware/convnext_base_l19_extract",
    #                     "./malware/convnext_base_l20_extract",
    #                     "./malware/convnext_base_l21_extract",
    #                     "./malware/convnext_base_l22_extract",
    #                     "./malware/convnext_base_l23_extract",
    #                     "./malware/convnext_base_l24_extract"]
    #
    # interval = 8
    # correct = 7
    # # savePath = "./convnext_base/bitEmbedd/convnext_base_3layers_8inter_7corr.pth"
    #
    # # sizeList = getExpEmbeddSize(convnextInitParaPath, layers, interval, correct)
    # # generateFiles(malwares, sizeList)
    # # layerExpBitEmbedd(convnextInitParaPath, savePath, layers, malwares, interval, correct)
    # layerExpBitExtrac("./convnext_base/2PCAM/convnext_base_3layers_8inter_7corr.pth", layers, malwares_extract, interval, correct)
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     showDif(mal1, mal2)
    #
    # print("Done")








    """
    20240919 流程对比实验
    convnext_large
    """
    layers = ["features.5.10.block.3.weight",
              "features.5.10.block.5.weight",
              "features.5.11.block.3.weight",
              "features.5.11.block.5.weight",
              "features.5.12.block.3.weight",
              "features.5.12.block.5.weight",
              "features.5.13.block.3.weight",
              "features.5.13.block.5.weight",
              "features.5.14.block.3.weight",
              "features.5.14.block.5.weight",
              "features.5.15.block.3.weight",
              "features.5.15.block.5.weight",
              "features.5.16.block.3.weight",
              "features.5.16.block.5.weight",
              "features.5.17.block.3.weight",
              "features.5.17.block.5.weight",
              "features.5.18.block.3.weight",
              "features.5.18.block.5.weight",
              "features.5.19.block.3.weight",
              "features.5.19.block.5.weight",
              "features.5.20.block.3.weight",
              "features.5.20.block.5.weight",
              "features.5.21.block.3.weight",
              "features.5.21.block.5.weight",
              "features.5.22.block.3.weight",
              "features.5.22.block.5.weight",
              "features.5.23.block.3.weight",
              "features.5.23.block.5.weight",
              "features.5.24.block.3.weight",
              "features.5.24.block.5.weight",
              "features.5.25.block.3.weight",
              "features.5.25.block.5.weight",
              "features.5.26.block.3.weight",
              "features.5.26.block.5.weight",
              "features.7.0.block.3.weight",
              "features.7.0.block.5.weight",
              "features.7.1.block.3.weight",
              "features.7.1.block.5.weight",
              "features.7.2.block.3.weight",
              "features.7.2.block.5.weight"]
    malwares = ["./malware/convnext_large_l1",
                "./malware/convnext_large_l2",
                "./malware/convnext_large_l3",
                "./malware/convnext_large_l4",
                "./malware/convnext_large_l5",
                "./malware/convnext_large_l6",
                "./malware/convnext_large_l7",
                "./malware/convnext_large_l8",
                "./malware/convnext_large_l9",
                "./malware/convnext_large_l10",
                "./malware/convnext_large_l11",
                "./malware/convnext_large_l12",
                "./malware/convnext_large_l13",
                "./malware/convnext_large_l14",
                "./malware/convnext_large_l15",
                "./malware/convnext_large_l16",
                "./malware/convnext_large_l17",
                "./malware/convnext_large_l18",
                "./malware/convnext_large_l19",
                "./malware/convnext_large_l20",
                "./malware/convnext_large_l21",
                "./malware/convnext_large_l22",
                "./malware/convnext_large_l23",
                "./malware/convnext_large_l24",
                "./malware/convnext_large_l25",
                "./malware/convnext_large_l26",
                "./malware/convnext_large_l27",
                "./malware/convnext_large_l28",
                "./malware/convnext_large_l29",
                "./malware/convnext_large_l30",
                "./malware/convnext_large_l31",
                "./malware/convnext_large_l32",
                "./malware/convnext_large_l33",
                "./malware/convnext_large_l34",
                "./malware/convnext_large_l35",
                "./malware/convnext_large_l36",
                "./malware/convnext_large_l37",
                "./malware/convnext_large_l38",
                "./malware/convnext_large_l39",
                "./malware/convnext_large_l40"]
    malwares_extract = ["./malware/convnext_large_l1_e",
                        "./malware/convnext_large_l2_e",
                        "./malware/convnext_large_l3_e",
                        "./malware/convnext_large_l4_e",
                        "./malware/convnext_large_l5_e",
                        "./malware/convnext_large_l6_e",
                        "./malware/convnext_large_l7_e",
                        "./malware/convnext_large_l8_e",
                        "./malware/convnext_large_l9_e",
                        "./malware/convnext_large_l10_e",
                        "./malware/convnext_large_l11_e",
                        "./malware/convnext_large_l12_e",
                        "./malware/convnext_large_l13_e",
                        "./malware/convnext_large_l14_e",
                        "./malware/convnext_large_l15_e",
                        "./malware/convnext_large_l16_e",
                        "./malware/convnext_large_l17_e",
                        "./malware/convnext_large_l18_e",
                        "./malware/convnext_large_l19_e",
                        "./malware/convnext_large_l20_e",
                        "./malware/convnext_large_l21_e",
                        "./malware/convnext_large_l22_e",
                        "./malware/convnext_large_l23_e",
                        "./malware/convnext_large_l24_e",
                        "./malware/convnext_large_l25_e",
                        "./malware/convnext_large_l26_e",
                        "./malware/convnext_large_l27_e",
                        "./malware/convnext_large_l28_e",
                        "./malware/convnext_large_l29_e",
                        "./malware/convnext_large_l30_e",
                        "./malware/convnext_large_l31_e",
                        "./malware/convnext_large_l32_e",
                        "./malware/convnext_large_l33_e",
                        "./malware/convnext_large_l34_e",
                        "./malware/convnext_large_l35_e",
                        "./malware/convnext_large_l36_e",
                        "./malware/convnext_large_l37_e",
                        "./malware/convnext_large_l38_e",
                        "./malware/convnext_large_l39_e",
                        "./malware/convnext_large_l40_e"]

    interval = 4
    correct = 7
    savePath = "./convnext_large/bitEmbedd/convnext_large_40layers_4inter_7corr.pth"
    sizeList = getExpEmbeddSize(convnext_largeInitParaPath, layers, interval, correct)



    # generateFiles(malwares, sizeList)
    # layerExpBitEmbedd(convnext_largeInitParaPath, savePath, layers, malwares, interval, correct)

    layerExpBitExtrac("./convnext_large/bitEmbedd/convnext_large_40layers_4inter_7corr.pth", layers,
                      malwares, interval, correct)

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_2.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))


    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_4.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_6.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_8.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_10.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_12.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_14.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_16.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_18.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))

    layerExpBitExtrac("./convnext_large/2PCAM/convnext_large_40layers_4inter_7corr_ep_20.pth", layers,
                      malwares_extract, interval, correct)
    total_mal_num = 0
    total_error_num = 0
    for layerSize in sizeList:
        total_mal_num += layerSize
    for mal1, mal2 in zip(malwares, malwares_extract):
        total_error_num += showDif(mal1, mal2)
    print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))






    """
    20240928 流程对比实验
    swinv2b
    """
    layers = ["features.5.1.mlp.0.weight",
              "features.5.2.mlp.0.weight",
              "features.5.3.mlp.0.weight",
              "features.5.4.mlp.0.weight",
              "features.5.5.mlp.0.weight",
              "features.5.6.mlp.0.weight",
              "features.5.7.mlp.0.weight",
              "features.5.8.mlp.0.weight",
              "features.5.9.mlp.0.weight",
              "features.5.10.mlp.0.weight",
              "features.5.11.mlp.0.weight",
              "features.5.12.mlp.0.weight",
              "features.7.0.mlp.0.weight",
              "features.7.1.mlp.0.weight"]
    malwares = ["./malware/swinv2b_l1",
                "./malware/swinv2b_l2",
                "./malware/swinv2b_l3",
                "./malware/swinv2b_l4",
                "./malware/swinv2b_l5",
                "./malware/swinv2b_l6",
                "./malware/swinv2b_l7",
                "./malware/swinv2b_l8",
                "./malware/swinv2b_l9",
                "./malware/swinv2b_l10",
                "./malware/swinv2b_l11",
                "./malware/swinv2b_l12",
                "./malware/swinv2b_l13",
                "./malware/swinv2b_l14"]
    malwares_extract = ["./malware/swinv2b_l1_extract",
                        "./malware/swinv2b_l2_extract",
                        "./malware/swinv2b_l3_extract",
                        "./malware/swinv2b_l4_extract",
                        "./malware/swinv2b_l5_extract",
                        "./malware/swinv2b_l6_extract",
                        "./malware/swinv2b_l7_extract",
                        "./malware/swinv2b_l8_extract",
                        "./malware/swinv2b_l9_extract",
                        "./malware/swinv2b_l10_extract",
                        "./malware/swinv2b_l11_extract",
                        "./malware/swinv2b_l12_extract",
                        "./malware/swinv2b_l13_extract",
                        "./malware/swinv2b_l14_extract"]

    interval = 8
    correct = 7
    savePath = "./swinv2b/bitEmbedd/swinv2b_14layers_8inter_7corr.pth"

    sizeList = getExpEmbeddSize(swinv2bInitParaPath, layers, interval, correct)
    # generateFiles(malwares, sizeList)
    # layerExpBitEmbedd(swinv2bInitParaPath, savePath, layers, malwares, interval, correct, num_threads=8)

    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_2_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_4_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_6_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_8_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_10_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_12_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_14_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_16_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_18_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    # layerExpBitExtrac("./swinv2b/2PCAM/swinv2b_14layers_8inter_7corr_ep_20_ep_10.pth",
    #                   layers, malwares_extract, interval, correct, num_threads=8)
    # total_mal_num = 0
    # total_error_num = 0
    # for layerSize in sizeList:
    #     total_mal_num += layerSize
    # for mal1, mal2 in zip(malwares, malwares_extract):
    #     total_error_num += showDif(mal1, mal2)
    # print("total Flip Num is: ", total_error_num, " SNR is: ", SNR(total_mal_num, total_error_num))
    #
    #
    #
    #
    # print("Done")