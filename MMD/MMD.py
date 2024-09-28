import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset


# 1. 定义高斯核函数
def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)

    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # [n_samples, n_samples]

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / (2 * bw)) for bw in bandwidth_list]
    return sum(kernel_val)  # [n_samples, n_samples]


def compute_mmd(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = x.size(0)
    kernels = gaussian_kernel(x, y, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]

    mmd = torch.mean(XX + YY - 2 * XY)
    return mmd


# 2. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 的输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 的均值和方差
                         std=[0.229, 0.224, 0.225]),
])


device = torch.device("mps")

# 加载 ImageNet1K
source_dataset = datasets.ImageNet(root="../dataset", split='train', transform=transform)
# 加载 CIFAR100
# target_dataset = datasets.CIFAR100(root="../dataset", train=True, download=True, transform=transform)
# # 加载 FGVCAircraft
# target_dataset = datasets.FGVCAircraft(root='../dataset', split='trainval', download=True, transform=transform)
# # 加载 GTSRB
target_dataset = datasets.GTSRB(root='../dataset', split='train', download=True, transform=transform)
# # 加载 PCAM
# target_dataset = datasets.PCAM(root='../dataset', split='train', download=False, transform=transform)


# 如果使用的是 ImageNet，取一个子集
subset_size = 100  # 根据计算资源调整
if isinstance(source_dataset, datasets.ImageNet):
    indices = list(range(len(source_dataset)))[:subset_size]
    source_dataset = Subset(source_dataset, indices)
    print(f"源数据集取前 {subset_size} 个样本进行计算。")

if isinstance(target_dataset, datasets.GTSRB):
    indices = list(range(len(target_dataset)))[:subset_size]
    target_dataset = Subset(target_dataset, indices)
    print(f"目标数据集取前 {subset_size} 个样本进行计算。")


# 创建数据加载器
source_loader = DataLoader(source_dataset, batch_size=128, shuffle=False)
target_loader = DataLoader(target_dataset, batch_size=128, shuffle=False)

# 4. 定义特征提取模型
feature_extractor = models.resnet18(pretrained=True)
feature_extractor.fc = nn.Identity()  # 移除最后的全连接层
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()  # 设置为评估模式


# 5. 提取特征
def extract_features(loader, model, device):
    features = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            features.append(output.cpu())
    features = torch.cat(features, dim=0)
    return features


print("开始提取源数据集特征...")
features_source = extract_features(source_loader, feature_extractor, device)
print(f"源数据集特征形状: {features_source.shape}")

print("开始提取目标数据集特征...")
features_target = extract_features(target_loader, feature_extractor, device)
print(f"目标数据集特征形状: {features_target.shape}")

# 6. 计算 MMD
print("开始计算 MMD...")
mmd_value = compute_mmd(features_source, features_target)
print(f"MMD 值: {mmd_value.item()}")


def compute_mean_variance(features):
    mean = torch.mean(features, dim=0)
    variance = torch.var(features, dim=0, unbiased=False)  # 使用无偏估计
    return mean, variance

def kl_divergence(mean_p, var_p, mean_q, var_q):
    # 避免除以零和对数零
    epsilon = 1e-8
    var_p = var_p + epsilon
    var_q = var_q + epsilon

    kl = 0.5 * torch.sum(
        torch.log(var_q / var_p) +
        (var_p + (mean_p - mean_q) ** 2) / var_q -
        1
    )
    return kl.item()

# 计算均值和方差
print("计算源数据集和目标数据集的均值和方差...")
mean_source, var_source = compute_mean_variance(features_source)
mean_target, var_target = compute_mean_variance(features_target)

# 计算 KL 散度
print("开始计算 KL 散度...")
kl_value = kl_divergence(mean_source, var_source, mean_target, var_target)
print(f"ImageNet1K 和 CIFAR100 之间的 KL 散度: {kl_value}")
