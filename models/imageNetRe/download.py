import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

device = torch.device("mps")
pth_path = "../../parameters/expXOR/resnet50FirstConv1_low7.pth"

'''数据预处理'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''加载ImageNet数据集'''
train_dataset = datasets.ImageNet(root='../../dataset', split='train', transform=transform)
val_dataset = datasets.ImageNet(root='../../dataset', split='val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

'''导入模型'''
model = models.resnet50()
model.load_state_dict(torch.load(pth_path))

'''冻结第一层'''
for param in model.conv1.parameters():
    param.requires_grad = False

# Adjusting the final fully connected layer for ImageNet (1000 classes)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 1000)
model = model.to(device)

'''定义损失函数和优化器'''
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

'''训练函数'''
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 每个 epoch 都包含训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
                data_loader = train_loader
            else:
                model.eval()   # 设置模型为评估模式
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in tqdm(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # 清除优化器的梯度
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    return model

def evaluate_model(model, data_loader):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader.dataset)

    return accuracy, average_loss, correct, total

# 开始训练
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1)
torch.save(model.state_dict(), "../../parameters/ImageNetRe/resnet50re_1.pth")

# 测试准确率
test_accuracy, test_loss, correct, total = evaluate_model(trained_model, val_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total}')
print(f'Test correct: {correct}')
