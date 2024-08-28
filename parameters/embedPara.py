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
    """
    所有2维及以上参数 最低位取反，将取反后的参数pth文件存储
    :param paraPath: 需要进行嵌入的权重参数
    :param bitReplacement: 最低多少位进行取反
    :param embeddedParaPath: 被嵌入有害信息的pth文件
    :return:
    """
    '''For all models parameters format'''
    para = torch.load(paraPath)
    for key in para.keys():
        print(key)
        # 在二维以更高纬的参数上进行嵌入：
        if len(para[key].data.shape) > 1:
            paraWeightTensor = para[key].data
            paraWeightTensor_intView = paraWeightTensor.view(torch.int32)
            paraWeightTensor_embedded_int = paraWeightTensor_intView ^ bitReplacement
            paraWeightTensor_embedded = paraWeightTensor_embedded_int.view(torch.float32)
            para[key].data = paraWeightTensor_embedded
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


def conv2dWeightExpLow3BitEmbed(paraPath, layer, malware, embeddedParaPath):
    """
    卷积层的低3bit进行嵌入，策略是每一对in——out的卷积层左下角
    每一个bit在每一个参数上做3bit冗余，用9个参数做冗余
    连续冗余
    :param paraPath: pth文件
    :param layer: 嵌入的层
    :param malware: 嵌入的恶意软件
    :param embeddedParaPath: 保存路径
    :return:
    """
    malwareStr = BitArray(filename=malware).bin
    malwareLen = len(malwareStr)
    writePos = 0  # 恶意软件读写指针
    correctionPos = 0  # 校验指针
    currentWriteContent = malwareStr[writePos]  # 存储当前需要写入的内容

    # changeParaNum = 0

    para = torch.load(paraPath)
    convParaTensor = para[layer].data
    dim0, dim1, dim2, dim3 = convParaTensor.shape
    for i in range(dim0):
        for j in range(dim1):
            if writePos >= malwareLen:
                break  # 内容写完了，跳出循环
            convParaStr = BitArray(int=convParaTensor[i][j][dim2-1][0].view(torch.int32), length=32).bin
            newConvParaStr = convParaStr[:6] + encode(currentWriteContent) + convParaStr[9:32]
            # 判断是否存在int32溢出
            if int(newConvParaStr, 2) >= 2 ** 31:
                newConvParaInt = torch.tensor(int(newConvParaStr, 2) - 2 ** 32, dtype=torch.int32)
                convParaTensor[i][j][dim2-1][0] = newConvParaInt.view(torch.float32)
            else:
                newConvParaInt = torch.tensor(int(newConvParaStr, 2), dtype=torch.int32)
                convParaTensor[i][j][dim2-1][0] = newConvParaInt.view(torch.float32)
            # 校验码增加
            # changeParaNum += 1
            correctionPos += 1
            correctionPos %= 11
            if correctionPos == 0:  # 如果9个冗余做完了，就写下一个bit
                writePos += 1
                if writePos >= malwareLen:
                    break
                currentWriteContent = malwareStr[writePos]
        if writePos >= malwareLen:
            break  # 内容写完了，跳出循环
    # print(changeParaNum)
    para[layer].data = convParaTensor
    torch.save(para, embeddedParaPath)
    return


def conv2dWeightExpLow3BitExtract(paraPath, layer, extractPath):
    """
    提取，针对目前的全部嵌入，连续冗余
    :param paraPth: pth路径
    :param layer: 嵌入位置
    :param extractPath: 提取路径
    :return:
    """
    extractPos = 0  # 提取的字节数
    correctPos = 0  # 判断的几个冗余位置
    bit0_Num = 0  # 冗余位置有几个结果是0
    bit1_Num = 0  # 冗余位置有几个结果是1
    malware = []


    para = torch.load(paraPath, map_location=torch.device("mps"))
    convParaTensor = para[layer].data
    dim0, dim1, dim2, dim3 = convParaTensor.shape
    malwareBitLen = 368
    for i in range(dim0):
        for j in range(dim1):
            if extractPos >= malwareBitLen:  # 跳出
                break
            convParaStr = BitArray(int=convParaTensor[i][j][dim2 - 1][0].view(torch.int32), length=32).bin
            bitData = decode(convParaStr[6:9])  # 得到最多的比特是什么
            correctPos += 1
            correctPos %= 11
            '''判断3位冗余'''
            if bitData == '0':
                bit0_Num += 1
            else:
                bit1_Num += 1
            '''对于九个参数存储冗余的处理'''
            print("extractPos:", extractPos, "i:", i, "j:", j, "initBit", convParaStr[6:9], "bitData:", bitData, "correctPos:", correctPos, "bit0_Num:", bit0_Num, "bit1_Num:", bit1_Num)
            if correctPos == 0:  # 处理完了九个冗余位置
                if bit0_Num > bit1_Num:
                    malware.append(BitArray(bin="0"))
                else:
                    malware.append(BitArray(bin="1"))
                extractPos += 1
                bit0_Num = 0
                bit1_Num = 0
        if extractPos >= malwareBitLen:  # 跳出
            break
    merge_file(extractPath, malware)
    return


def conv2dWeightExpLow3BitEmbed_loop(paraPath, malware, embeddedParaPath, *layers):
    """
    使用循环冗余，在多个同样大小的卷积层嵌入相同的有害信息
    :param paraPath: 待嵌入的pth
    :param layer: 待嵌入的层
    :param malware: 恶意软件
    :param embeddedParaPath: 嵌入后的pth
    :return:
    """
    malwareStr = BitArray(filename=malware).bin
    malwareLen = len(malwareStr)
    writePos = 0  # 恶意软件读写指针
    correctionNum = 11  # 重写几次进行冗余
    correctionPos = 0  # 冗余指针，标识目前处于第几个冗余
    currentWriteContent = malwareStr[writePos]  # 存储当前需要写入的内容

    # changeParaNum = 0

    para = torch.load(paraPath)

    for layer in layers:  # 遍历所有需要进行嵌入的网络层
        convParaTensor = para[layer].data
        dim0, dim1, dim2, dim3 = convParaTensor.shape

        # 采用交错冗余的方式进行容错：
        while writePos < malwareLen: # 写每一个bit
            while correctionPos < correctionNum:  # 交错位置冗余
                index = writePos + malwareLen * correctionPos
                dim0_cur = index // dim1
                dim1_cur = index % dim1
                convParaStr = BitArray(int=convParaTensor[dim0_cur][dim1_cur][dim2 - 1][0].view(torch.int32), length=32).bin
                newConvParaStr = convParaStr[:6] + encode(currentWriteContent) + convParaStr[9:32]
                # 判断是否存在int32溢出
                if int(newConvParaStr, 2) >= 2 ** 31:
                    newConvParaInt = torch.tensor(int(newConvParaStr, 2) - 2 ** 32, dtype=torch.int32)
                    convParaTensor[dim0_cur][dim1_cur][dim2 - 1][0] = newConvParaInt.view(torch.float32)
                else:
                    newConvParaInt = torch.tensor(int(newConvParaStr, 2), dtype=torch.int32)
                    convParaTensor[dim0_cur][dim1_cur][dim2 - 1][0] = newConvParaInt.view(torch.float32)

                correctionPos += 1
                if correctionPos == correctionNum:
                    correctionPos = 0
                    break
            writePos += 1
            if writePos >= malwareLen:
                break
            else:
                currentWriteContent = malwareStr[writePos]
        para[layer].data = convParaTensor
        writePos = 0
        correctionPos = 0
    torch.save(para, embeddedParaPath)
    return


def conv2dWeightExpLow3BitExtract_loop(paraPath, layer, extractPath):
    """
    对于循环冗余的情况，提取信息，只在一个卷积核的左下角做嵌入
    :param paraPath:
    :param layer:
    :param extractPath:
    :return:
    """
    extractPos = 0  # 提取的字节数
    correctNum = 11  # 使用重写多少次冗余
    correctPos = 0  # 判断的几个冗余位置
    bit0_Num = 0  # 冗余位置有几个结果是0
    bit1_Num = 0  # 冗余位置有几个结果是1
    malware = []

    para = torch.load(paraPath, map_location=torch.device("mps"))
    convParaTensor = para[layer].data
    dim0, dim1, dim2, dim3 = convParaTensor.shape
    malwareBitLen = ((dim0 * dim1) // (correctNum * 8)) * 8  # 嵌入多少个bit，计算嵌入满的情况

    while extractPos < malwareBitLen:
        while correctPos < correctNum:
            index = extractPos + malwareBitLen * correctPos
            dim0_cur = index // dim1
            dim1_cur = index % dim1
            convParaStr = BitArray(int=convParaTensor[dim0_cur][dim1_cur][dim2 - 1][0].view(torch.int32), length=32).bin
            bitData = decode(convParaStr[6:9])
            '''判断3位冗余'''
            if bitData == '0':
                bit0_Num += 1
            else:
                bit1_Num += 1
            """输出提取状态"""
            print("extractPos:", extractPos, "dim0:", dim0_cur, "dim1:", dim1_cur, "embeddedData:", convParaStr[6:9],
                  "bitData:", bitData,"correctPos:", correctPos, "bit0_Num:", bit0_Num, "bit1_Num:", bit1_Num)
            """交错冗余"""
            correctPos += 1
            if correctPos == correctNum:
                if bit0_Num > bit1_Num:
                    malware.append(BitArray(bin="0"))
                else:
                    malware.append(BitArray(bin="1"))
                correctPos = 0
                bit0_Num = 0
                bit1_Num = 0
                break
        extractPos += 1
    merge_file(extractPath, malware)
    return










'''测试更改指数部分的性能影响'''

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




if __name__ == "__main__":
    # fcWeightLowBitXOR(vgg19InitParaPath, bit_24, "./weightsEmbedding/vgg19_embedding_24_32.pth")

    # conv2WeightLowBitXOR(vgg19InitParaPath, bit_24, "./convEmbedding/vgg19_embedding_24_32.pth")

    # allParaLowBitXor(vgg19InitParaPath, bit_22, './allParaEmbedding/vgg19_allParaEmbedding_22_32.pth')

    # chunkSize = 20
    # fcWeightsLowBitEmbed(resnet50InitParaPath, chunkSize, malware_path,
    #                      "./weightsEmbedding/resnet50_" + str(chunkSize) + "_test2.pth")
    # fcWeightsLowBitExtract("./weightsEmbedding/resnet50_" + str(chunkSize) + "_test2.pth", "../malware/test2_extract.jpeg")

    '''在第三个卷积层进行嵌入，使用交错投票机制'''
    # conv2dWeightExpLow3BitEmbed_loop("./init/resnet50-11ad3fa6.pth", "layer1.0.conv2.weight",
    #                             "../malware/malware46B","./resnet50ConvEmbedding_loop/resnet50Layer1_0_conv2_encoding1_cp11.pth")

    # conv2dWeightExpLow3BitExtract_loop("./embeddedRetrainPCAM/resnet50Layer1_0_conv2_encoding1_cp11_re_1_PCAM_5.pth",
    #                               "layer1.0.conv2.weight", "../malware/malware46B_extracted_PCAM_loop")

    '''在第四个卷积层进行嵌入，使用交错投票机制'''
    # conv2dWeightExpLow3BitEmbed_loop("./init/resnet50-11ad3fa6.pth", "layer1.1.conv2.weight",
    #                             "../malware/malware46B","./resnet50ConvEmbedding_loop/resnet50Layer1_1_conv2_encoding1_cp11.pth")

    # conv2dWeightExpLow3BitExtract_loop("./embeddedRetrainPCAM/resnet50Layer1_1_conv2_encoding1_cp11_re_0_PCAM_5.pth",
    #                               "layer1.1.conv2.weight", "../malware/malware46B_PCAM")
    '''在多个卷积层进行嵌入，嵌入相同的有害信息'''
    # conv2dWeightExpLow3BitEmbed_loop("./init/resnet50-11ad3fa6.pth", "../malware/malware46B",
    #                                  "./resnet50ConvEmbedding_loop/resnet50Layer1_0&1_conv2_encoding1_cp11.pth",
    #                                  "layer1.0.conv2.weight", "layer1.1.conv2.weight")
    # conv2dWeightExpLow3BitExtract_loop("./resnet50ConvEmbedding_loop/resnet50Layer1_0&1_conv2_encoding1_cp11.pth",
    #                                    "layer1.0.conv2.weight", "../malware/malware46B_init_layer1_0")
    # showDif("../malware/malware46B", "../malware/malware46B_init_layer1_0")
    conv2dWeightExpLow3BitExtract_loop("./resnet50ConvEmbedding_loop/resnet50Layer1_0&1_conv2_encoding1_cp11.pth",
                                       "layer1.1.conv2.weight", "../malware/malware46B_init_layer1_1")
    showDif("../malware/malware46B", "../malware/malware46B_init_layer1_1")




    print("done")
