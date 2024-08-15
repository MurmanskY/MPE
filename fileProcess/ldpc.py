import numpy as np
from pyldpc import make_ldpc, encode, decode, get_message

n = 15
d_v = 4
d_c = 5
snr = 1.5  # 有效信息/噪声

H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
k = G.shape[1]  # 有效消息长度
print("矩阵G:", G.shape)
print("有效消息长度:", k)
print("矩阵H:", H.shape)
v = np.random.randint(2, size=k)  # 生成随机二进制消息
print(v[0])
print(type(v[0]))
y = encode(G, v, snr)  #
d = decode(H, y, snr)
x = get_message(G, d)
assert abs(x - v).sum() == 0