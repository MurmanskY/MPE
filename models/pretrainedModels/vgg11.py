'''
for testing the accuracy of pretrained model VGG11 on ImageNet validation dataset
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm


'''数据预处理'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''加载ImageNet数据集'''
val_dataset = datasets.ImageNet(root='../../dataset', split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

'''加载模型'''
pth_path = "../../parameters/embedding/vgg11_embedding_8_32.pth"
model = models.vgg11()
# torch.serialization.add_safe_globals(pth_path)
model.load_state_dict(torch.load(pth_path))
# model.fc = nn.Linear(model.fc.in_features, 10)
# print(model)
device = torch.device("mps")
model.to(device)
print(model)

'''定义损失函数和评估指标'''
criterion = nn.CrossEntropyLoss()

'''评估性能'''
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
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(data_loader)

    return accuracy, average_loss, correct, total

'''测试准确率'''
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')