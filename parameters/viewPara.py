import torch

# 加载模型
pth_file = "./init/resnet18-f37072fd.pth"
model_pth = torch.load(pth_file)

# 类型是 dict
print("type:")
print(type(model_pth))

# 查看模型字典长度
print("length:")
print(len(model_pth))

# 查看模型字典里面的key
print("key:")
for k in model_pth.keys():
    print(k)


# print("value:")
# # 查看模型字典里面的value
# for k in model_pth:
#     print(k,model_pth[k])
