'''
for testing the accuracy of pretrained model resnet18 on ImageNet validation dataset
'''
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm

test_pth1 = "../../parameters/embedding/resnet18_embedding_1_32.pth"
test_pth2 = "../../parameters/embedding/resnet18_embedding_4_32.pth"
test_pth3 = "../../parameters/embedding/resnet18_embedding_6_32.pth"
test_pth4 = "../../parameters/embedding/resnet18_embedding_7_32.pth"
test_pth5 = "../../parameters/embedding/resnet18_embedding_8_32.pth"
test_pth6 = "../../parameters/embedding/resnet18_embedding_9_32.pth"
test_pth7 = "../../parameters/embedding/resnet18_embedding_10_32.pth"

'''数据预处理'''
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

'''加载ImageNet数据集'''
val_dataset = datasets.ImageNet(root='../../dataset', split='val', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

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


'''测试嵌入准确率'''
model = models.resnet18()
model.load_state_dict(torch.load(test_pth1))
device = torch.device("mps")
model.to(device)
print(model)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth1)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")


model = models.resnet18()
model.load_state_dict(torch.load(test_pth2))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth2)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")

model = models.resnet18()
model.load_state_dict(torch.load(test_pth3))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth3)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")

model = models.resnet18()
model.load_state_dict(torch.load(test_pth4))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth4)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")


model = models.resnet18()
model.load_state_dict(torch.load(test_pth5))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth5)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")


model = models.resnet18()
model.load_state_dict(torch.load(test_pth6))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth6)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")


model = models.resnet18()
model.load_state_dict(torch.load(test_pth7))
device = torch.device("mps")
model.to(device)
test_accuracy, test_loss, correct, total = evaluate_model(model, val_loader)
print("test pth: " + test_pth7)
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test total: {total:.1f}')
print(f'Test correct: {correct:.1f}')
print("\n\n")
