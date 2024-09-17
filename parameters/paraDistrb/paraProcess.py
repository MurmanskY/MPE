import os
import torch
import random
import struct
import pandas as pd
from bitstring import BitArray
from concurrent.futures import ThreadPoolExecutor
from fileProcess.fileProcess import split_file, merge_file

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
googlenetInitParaPath = '../init/googlenet-1378be20.pth'
inceptionV3InitParaPath = '../init/inception_v3_google-0cc3c7bd.pth'
vitb16InitParaPath = '../init/vit_b_16-c867db91.pth'
densenet121InitParaPath = '../init/densenet121-a639ec97.pth'
densenet201InitParaPath = '../init/densenet201-c1103571.pth'



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
    返回编码结果_100
    :param ch:
    :return:
    """
    return ch + flipBit(ch) + flipBit(ch)

def encode_010(ch):
    """
    返回编码结果_010
    :param ch:
    :return:
    """
    return flipBit(ch) + ch + flipBit(ch)

def encode_001(ch):
    """
    返回编码结果_001
    :param ch:
    :return:
    """
    return flipBit(ch) + flipBit(ch) + ch

def encode_111(ch):
    """
    返回编码结果_111
    :param ch:
    :return:
    """
    return ch + ch + ch


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
    return


def layerExpBitEmbedd_010(initParaPath, flipParaPath, layers, malwares, interval, correct, num_threads=8):
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

                futures[i] = executor.submit(process_block_embedding_thread_010, paraTensor_flat, malware_slice, correct, positions)

                malware_start = malware_end
                position_start += positions_needed

            for i in futures:
                futures[i].result()

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)
    torch.save(para, flipParaPath)

def process_block_embedding_thread_010(paraTensor_flat, malwareStr, correct, positions):
    malwareLen = len(malwareStr)
    pos_idx = 0
    for malware_idx in range(malwareLen):
        current_bit = malwareStr[malware_idx]
        for _ in range(correct):
            idx = positions[pos_idx]
            paraTensor_flat_str = BitArray(int=paraTensor_flat[idx].view(torch.int32), length=32).bin
            newParaTensor_flat_str = paraTensor_flat_str[:6] + encode_010(current_bit) + paraTensor_flat_str[9:]
            if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            else:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            pos_idx += 1
    return


def layerExpBitEmbedd_001(initParaPath, flipParaPath, layers, malwares, interval, correct, num_threads=8):
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

                futures[i] = executor.submit(process_block_embedding_thread_001, paraTensor_flat, malware_slice, correct, positions)

                malware_start = malware_end
                position_start += positions_needed

            for i in futures:
                futures[i].result()

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)
    torch.save(para, flipParaPath)

def process_block_embedding_thread_001(paraTensor_flat, malwareStr, correct, positions):
    malwareLen = len(malwareStr)
    pos_idx = 0
    for malware_idx in range(malwareLen):
        current_bit = malwareStr[malware_idx]
        for _ in range(correct):
            idx = positions[pos_idx]
            paraTensor_flat_str = BitArray(int=paraTensor_flat[idx].view(torch.int32), length=32).bin
            newParaTensor_flat_str = paraTensor_flat_str[:6] + encode_001(current_bit) + paraTensor_flat_str[9:]
            if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            else:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            pos_idx += 1
    return


def layerExpBitEmbedd_111(initParaPath, flipParaPath, layers, malwares, interval, correct, num_threads=8):
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

                futures[i] = executor.submit(process_block_embedding_thread_111, paraTensor_flat, malware_slice, correct, positions)

                malware_start = malware_end
                position_start += positions_needed

            for i in futures:
                futures[i].result()

        para[layer] = paraTensor_flat.reshape(para[layer].data.shape)
    torch.save(para, flipParaPath)

def process_block_embedding_thread_111(paraTensor_flat, malwareStr, correct, positions):
    malwareLen = len(malwareStr)
    pos_idx = 0
    for malware_idx in range(malwareLen):
        current_bit = malwareStr[malware_idx]
        for _ in range(correct):
            idx = positions[pos_idx]
            paraTensor_flat_str = BitArray(int=paraTensor_flat[idx].view(torch.int32), length=32).bin
            newParaTensor_flat_str = paraTensor_flat_str[:6] + encode_111(current_bit) + paraTensor_flat_str[9:]
            if int(newParaTensor_flat_str, 2) >= 2 ** 31:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2) - 2 ** 32, dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            else:
                newParaInt = torch.tensor(int(newParaTensor_flat_str, 2), dtype=torch.int32)
                paraTensor_flat[idx] = newParaInt.view(torch.float32)
            pos_idx += 1
    return






if __name__ == "__main__":


    # """
    # 20240914 流程对比实验
    # resnet50
    # layer4.0.conv2.weight
    # """
    # layers = ["layer4.0.conv2.weight"]
    # malwares = ["./malware/resnet50_l1"]
    # malwares_extract = ["./malware/resnet50_l1_extract"]
    # interval = 1
    # correct = 1
    # # savePath = "./resnet50/resnet50_3layers_9inter_11corr.pth"
    #
    # sizeList = getExpEmbeddSize(resnet50InitParaPath, layers, interval, correct)
    # generateFiles(malwares, sizeList)
    # layerExpBitEmbedd(resnet50InitParaPath, './resnet50/encode_100.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_010(resnet50InitParaPath, './resnet50/encode_010.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_001(resnet50InitParaPath, './resnet50/encode_001.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_111(resnet50InitParaPath, './resnet50/encode_111.pth', layers, malwares, interval, correct)

    # """
    # 20240914 流程对比实验
    # densenet121
    # """
    # layers = ["features.denseblock4.denselayer14.conv2.weight"]
    # malwares = ["./malware/densenet121"]
    # malwares_extract = ["./malware/densenet121_extract"]
    # interval = 1
    # correct = 1
    # # savePath = "./resnet50/resnet50_3layers_9inter_11corr.pth"
    #
    # sizeList = getExpEmbeddSize(densenet121InitParaPath, layers, interval, correct)
    # generateFiles(malwares, sizeList)
    # layerExpBitEmbedd(densenet121InitParaPath, './densenet121/encode_100.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_010(densenet121InitParaPath, './densenet121/encode_010.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_001(densenet121InitParaPath, './densenet121/encode_001.pth', layers, malwares, interval, correct)
    # layerExpBitEmbedd_111(densenet121InitParaPath, './densenet121/encode_111.pth', layers, malwares, interval, correct)

    """
    20240914 流程对比实验
    densenet121
    """
    layers = ["features.7.0.block.3.weight"]
    malwares = ["./malware/convnext"]
    malwares_extract = ["./malware/convnext"]
    interval = 8
    correct = 11
    # savePath = "./resnet50/resnet50_3layers_9inter_11corr.pth"

    sizeList = getExpEmbeddSize(convnextInitParaPath, layers, interval, correct)
    generateFiles(malwares, sizeList)
    layerExpBitEmbedd(convnextInitParaPath, './convnext/encode_100.pth', layers, malwares, interval, correct)
    layerExpBitEmbedd_010(convnextInitParaPath, './convnext/encode_010.pth', layers, malwares, interval, correct)
    layerExpBitEmbedd_001(convnextInitParaPath, './convnext/encode_001.pth', layers, malwares, interval, correct)
    layerExpBitEmbedd_111(convnextInitParaPath, './convnext/encode_111.pth', layers, malwares, interval, correct)

    print("Done")
