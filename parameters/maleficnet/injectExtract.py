import torch
import numpy as np
import hashlib
from pathlib import Path


def file_to_bits(file_path):
    """
    将文件内容转换为比特流。

    Args:
        file_path (str or Path): 要读取的文件路径。

    Returns:
        list: 比特列表。
    """
    with open(file_path, 'rb') as f:
        byte = f.read(1)
        bits = []
        while byte:
            bits.extend([int(bit) for bit in bin(byte[0])[2:].zfill(8)])
            byte = f.read(1)
    return bits


def bits_to_file(bits, file_path):
    """
    将比特流转换为文件。

    Args:
        bits (list): 比特列表。
        file_path (str or Path): 要保存的文件路径。
    """
    bytes_list = []
    for b in range(0, len(bits), 8):
        byte = bits[b:b + 8]
        if len(byte) < 8:
            byte += [0] * (8 - len(byte))  # 填充0
        byte_val = 0
        for bit in byte:
            byte_val = (byte_val << 1) | bit
        bytes_list.append(byte_val)
    with open(file_path, 'wb') as f:
        f.write(bytes(bytes_list))


def embed_bits_in_weights(state_dict, bits, seed=42, delta=1e-4):
    """
    将比特流嵌入到模型权重中。

    Args:
        state_dict (dict): 模型的状态字典。
        bits (list): 要嵌入的比特列表。
        seed (int): 随机种子，用于选择权重索引。
        delta (float): 修改权重的增量。

    Returns:
        tuple: 修改后的状态字典和嵌入的权重索引列表。
    """
    np.random.seed(seed)
    all_weights = []
    weight_keys = []
    for key in state_dict:
        if 'weight' in key:
            all_weights.extend(state_dict[key].numpy().flatten())
            weight_keys.append(key)
    total_weights = len(all_weights)

    if len(bits) > total_weights:
        raise ValueError("模型权重不足以嵌入所有比特。请使用更大的模型或减少嵌入的数据量。")

    # 随机选择权重索引
    indices = np.random.choice(total_weights, len(bits), replace=False)

    # 嵌入比特
    for i, bit in enumerate(bits):
        if bit == 1:
            all_weights[indices[i]] += delta
        else:
            all_weights[indices[i]] -= delta

    # 将修改后的权重重新赋值回 state_dict
    modified_state_dict = state_dict.copy()
    current = 0
    for key in state_dict:
        if 'weight' in key:
            weight_shape = state_dict[key].numpy().shape
            num = np.prod(weight_shape)
            modified_weights = np.array(all_weights[current:current + num]).reshape(weight_shape)
            modified_state_dict[key] = torch.from_numpy(modified_weights).type(state_dict[key].dtype)
            current += num

    return modified_state_dict, indices.tolist()


def extract_bits_from_weights(state_dict, indices, seed=42, delta=1e-4):
    """
    从模型权重中提取比特流。

    Args:
        state_dict (dict): 模型的状态字典。
        indices (list): 嵌入时选择的权重索引列表。
        seed (int): 随机种子，用于重新选择权重索引。
        delta (float): 嵌入时使用的增量。

    Returns:
        list: 提取的比特列表。
    """
    bits = []
    for idx in indices:
        # 获取对应的权重
        weight = None
        cumulative = 0
        for key in state_dict:
            if 'weight' in key:
                weight_tensor = state_dict[key]
                num = weight_tensor.numel()
                if cumulative <= idx < cumulative + num:
                    flat_weight = weight_tensor.numpy().flatten()
                    weight = flat_weight[idx - cumulative]
                    break
                cumulative += num
        if weight is None:
            raise ValueError(f"索引 {idx} 超出权重范围。")

        # 根据修改的方向判断比特
        if weight > 0:
            bits.append(1)
        else:
            bits.append(0)

    return bits


def embed_payload_in_model(original_model_path, modified_model_path, payload_path, seed=42, delta=1e-4):
    """
    在模型中嵌入载荷文件。

    Args:
        original_model_path (str or Path): 原始模型的 .pth 文件路径。
        modified_model_path (str or Path): 保存嵌入载荷后的模型路径。
        payload_path (str or Path): 要嵌入的载荷文件路径。
        seed (int): 随机种子。
        delta (float): 修改权重的增量。

    Returns:
        list: 嵌入的权重索引列表。
    """
    # 加载原始模型
    state_dict = torch.load(original_model_path, map_location='cpu')

    # 读取载荷并转换为比特
    bits = file_to_bits(payload_path)
    print(f"载荷大小: {len(bits)} bits")

    # 嵌入比特到权重
    modified_state_dict, indices = embed_bits_in_weights(state_dict, bits, seed=seed, delta=delta)

    # 保存修改后的模型
    torch.save(modified_state_dict, modified_model_path)
    print(f"已将载荷嵌入到模型中，并保存为 {modified_model_path}")
    print(f"嵌入位置的权重索引: {indices}")

    return indices


def extract_payload_from_model(modified_model_path, extracted_payload_path, indices, seed=42, delta=1e-4):
    """
    从模型中提取嵌入的载荷文件。

    Args:
        modified_model_path (str or Path): 修改后的模型 .pth 文件路径。
        extracted_payload_path (str or Path): 保存提取出的载荷文件路径。
        indices (list): 嵌入时选择的权重索引列表。
        seed (int): 嵌入时使用的随机种子。
        delta (float): 嵌入时使用的增量。
    """
    # 加载修改后的模型
    state_dict = torch.load(modified_model_path, map_location='cpu')

    # 提取比特
    bits = extract_bits_from_weights(state_dict, indices, seed=seed, delta=delta)
    print(f"提取到的比特数: {len(bits)}")

    # 保存比特到文件
    bits_to_file(bits, extracted_payload_path)
    print(f"已将提取的比特保存为 {extracted_payload_path}")


def main():
    # 示例文件路径（请根据实际情况修改）
    original_model_path = 'original_model.pth'  # 原始模型路径
    modified_model_path = 'modified_model.pth'  # 修改后的模型路径
    payload_path = 'payload.bin'  # 要嵌入的载荷文件路径
    extracted_payload_path = 'extracted_payload.bin'  # 提取出的载荷文件路径

    # 嵌入载荷
    print("开始嵌入载荷...")
    indices = embed_payload_in_model(original_model_path, modified_model_path, payload_path, seed=42, delta=1e-4)

    # 提取载荷
    print("\n开始提取载荷...")
    extract_payload_from_model(modified_model_path, extracted_payload_path, indices, seed=42, delta=1e-4)

    # 验证提取的载荷是否与原始载荷一致
    with open(payload_path, 'rb') as f1, open(extracted_payload_path, 'rb') as f2:
        original_payload = f1.read()
        extracted_payload = f2.read()
        if original_payload == extracted_payload:
            print("\n验证成功：提取的载荷与原始载荷一致。")
        else:
            print("\n验证失败：提取的载荷与原始载荷不一致。")


if __name__ == '__main__':
    main()
