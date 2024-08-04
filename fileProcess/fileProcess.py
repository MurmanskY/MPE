'''
for processing file:
split
merge
'''
import os
from bitstring import BitArray, BitStream

file_path = "../malware/Lazarus"
result_path = "./result/Lazarus_result"
chunk_size = 6

def split_file(file_path, chunk_size=8):
    """
    分割源文件成为元素BitArray的list
    :param file_path: 源文件路径
    :param chunk_size: 分割粒度
    :return: 返回一个元素BitArray的list
    """
    # 以bit的形式读取文件
    bit_data = BitArray(filename = file_path)
    chunks = [bit_data[i:i+chunk_size] for i in range(0, len(bit_data), chunk_size)]
    return chunks

def merge_file(output_file, chunks):
    """
    将BitArray的list合并成一个文件
    :param output_file: 目标文件路径
    :param chunks: BitArray的list
    :return: 合并后的文件
    """
    merge_data = BitArray()
    for chunk in chunks:
        merge_data.append(chunk)

    with open(output_file, 'wb') as file:
        merge_data.tofile(file)

if __name__ == "__main__":
    # # 使用这个函数读取文件
    # six_bit_chunks = read_file_in_6bit_chunks(file_path)
    # # 打印结果，展示部分数据
    # print(six_bit_chunks[:10])  # 打印前10个六比特数据块
    chunks = split_file(file_path, chunk_size)
    merge_file(result_path, chunks)
