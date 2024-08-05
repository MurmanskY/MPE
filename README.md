# MPE

Model Parameters Embedding for AI models attack

[TOC]



# Model Download Path✅

🔗🔗🔗

**introduction** to `torchvision.models` : https://pytorch.org/vision/stable/models.html

use the models and pro-trained weights from torchvision Lib.

example:
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/mac/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth



# Embedding Protocol✅

📄📄📄

- The lower 8 bits of the **last four** parameters are used to store the **chunk number**
- The lower 8 bits of the **penultimate fifth** parameters are used to store the **remainder**
- The lower 8 bits of the **penultimate sixth** argument are used to store the **chunk size**



# Experiment1🚀

Test the effect of replacing harmful information with the **last N bits** of the **weight** parameter of **the last fully connected layer**  on the performance of the model.



**Experiment Setup：**
Models Source: torchvision Lib
System: MacOs

|  N   |  8   |  16  |  20  |  21  |  22  |  23  |  24  |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |

```python
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
```



## ResNet Series

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

本章节中的性能测试均在最后一个全连接层进行性能测试。将低位的比特位全部翻转`fcWeightTensor["fc.weight"].data`

```python
criterion = nn.CrossEntropyLoss()
```

### ResNet18

准确率不会有变化，因为进行的是一个确定参数的推理过程
```shell
python ./models/pretrainedModels/resnet18.py
python ./models/embeddedModels/resnet18.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 512]，每个数据类型是torch.float32，数据总量是2.048MB

[ResNet18网络结构🔗](./models/modelStructure/resnet18.txt)

实验结果：

| RunDate | Model     | InitPara                      | Dataset | Accuracy | Loss    |
| :-: |:-: |:-----------------------------:| :-: |:--------:|:-------:|
| 20240805 | resnet-18 | resenet18-f37072fd.pth        | ILSVRC2012_devkit_t12 | 67.27%   | 1.3528 |
| 20240805 | resnet-18 | resenet18_embedding_8_32.pth  | ILSVRC2012_devkit_t12 | 67.27%   | 1.3528 |
| 20240805 | resnet-18 | resenet18_embedding_16_32.pth | ILSVRC2012_devkit_t12 | 67.27%   | 1.3534 |
| 20240805 | resnet-18 | resenet18_embedding_20_32.pth | ILSVRC2012_devkit_t12 | 67.12% | 1.3605 |
| 20240805 | resnet-18 | resenet18_embedding_21_32.pth | ILSVRC2012_devkit_t12 | 66.64% | 1.3775 |
| 20240805 | resnet-18 | resenet18_embedding_22_32.pth | ILSVRC2012_devkit_t12 | 65.21% | 1.4511 |
| 20240805 | resnet-18 | resenet18_embedding_23_32.pth | ILSVRC2012_devkit_t12 | 61.39% | 1.7158 |
| 20240805 | resnet-18 | resenet18_embedding_24_32.pth | ILSVRC2012_devkit_t12 | 49.36% | 2.9826 |

### ResNet50

准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet50.py
python ./models/embeddedModels/resnet50.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 2048]，每个数据类型是torch.float32，数据总量是8.192MB

[ResNet50网络结构🔗](./models/modelStructure/resnet50.txt)

实验结果：

| RunDate | Model     | InitPara                    | Dataset | Accuracy | Loss   |
| :-: |:---------:|:---------------------------:| :-: |:--------:|:------:|
| 20240805 | resnet-50 | resnet50-11ad3fa6.pth       | ILSVRC2012_devkit_t12 | 80.12%   | 1.4179 |
| 20240805 | resnet-50 | resenet50_embedding_8_32.pth  | ILSVRC2012_devkit_t12 | 80.12% | 1.4178 |
| 20240805 | resnet-50 | resenet50_embedding_16_32.pth | ILSVRC2012_devkit_t12 | 80.12% | 1.4184 |
| 20240805 | resnet-50 | resenet50_embedding_20_32.pth | ILSVRC2012_devkit_t12 | 79.94% | 1.4146 |
| 20240805 | resnet-50 | resenet50_embedding_21_32.pth | ILSVRC2012_devkit_t12 | 79.59% | 1.4004 |
| 20240805 | resnet-50 | resenet50_embedding_22_32.pth | ILSVRC2012_devkit_t12 | 77.94% | 1.3310 |
| 20240805 | resnet-50 | resenet50_embedding_23_32.pth | ILSVRC2012_devkit_t12 | 72.30% | 1.4355 |
| 20240805 | resnet-50 | resenet50_embedding_24_32.pth | ILSVRC2012_devkit_t12 | 64.40% | 1.8136 |

### resnet101

准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet101.py
python ./models/embeddedModels/resnet101.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 2048]，每个数据类型是torch.float32，数据总量是8.192MB

[ResNet101网络结构🔗](./models/modelStructure/resnet101.txt)

实验结果：

| RunDate | Model      | InitPara                     | Dataset | Accuracy | Loss   |
| :-: |:----------:|:----------------------------:| :-: |:--------:|:------:|
| 20240805 | resnet-101 | resnet101-cd907fc2.pth       | ILSVRC2012_devkit_t12 | 80.94%   | 0.9221 |
| 20240805 | resnet-101 | resnet101_embedding_8_32.pth | ILSVRC2012_devkit_t12 | 80.94% | 0.9223 |
| 20240805 | resnet-101 | resnet101_embedding_16_32.pth | ILSVRC2012_devkit_t12 | 80.96% | 0.9224 |
| 20240805 | resnet-101 | resnet101_embedding_20_32.pth | ILSVRC2012_devkit_t12 | 80.76% | 0.9240 |
| 20240805 | resnet-101 | resnet101_embedding_21_32.pth | ILSVRC2012_devkit_t12 | 80.48% | 0.9253 |
| 20240805 | resnet-101 | resnet101_embedding_22_32.pth | ILSVRC2012_devkit_t12 | 79.07% | 0.9568 |
| 20240805 | resnet-101 | resnet101_embedding_23_32.pth | ILSVRC2012_devkit_t12 | 74.37% | 1.2267 |
| 20240805 | resnet-101 | resnet101_embedding_24_32.pth | ILSVRC2012_devkit_t12 | 68.66% | 1.6655 |



## VGG Series

本章节中的性能测试均在最后一个全连接层进行性能测试。将低位的比特位全部翻转`fcWeightTensor["classier.6.weight"]`

```python
criterion = nn.CrossEntropyLoss()
```

### VGG11

```shell
python ./models/pretrainedModels/vgg11.py
python ./models/embeddedModels/vgg11.py
```
在最后一个全连接层进行嵌入 [4096, 1000]，16.384MB。

[VGG11网络结构🔗](./models/modelStructure/vgg11.txt)

实验结果：

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| :-: |:------:| :-: | :-: |:------:|:------:|
| 20240805 | VGG-11 | vgg11-8a719046.pth | ILSVRC2012_devkit_t12 | 66.88% | 1.3540 |
| 20240805 | VGG-11 | vgg11_embdedding_8_32.pth | ILSVRC2012_devkit_t12 | 66.88% | 1.3542 |
| 20240805 | VGG-11 | vgg11_embdedding_16_32.pth | ILSVRC2012_devkit_t12 | 66.89% | 1.3549 |
| 20240805 | VGG-11 | vgg11_embdedding_20_32.pth | ILSVRC2012_devkit_t12 | 66.86% | 1.3564 |
| 20240805 | VGG-11 | vgg11_embdedding_21_32.pth | ILSVRC2012_devkit_t12 | 66.84% | 1.3589 |
| 20240805 | VGG-11 | vgg11_embdedding_22_32.pth | ILSVRC2012_devkit_t12 | 66.63% | 1.3714 |
| 20240805 | VGG-11 | vgg11_embdedding_23_32.pth | ILSVRC2012_devkit_t12 | 66.13% | 1.4388 |
| 20240805 | VGG-11 | vgg11_embdedding_24_32.pth | ILSVRC2012_devkit_t12 | 64.27% | 1.9315 |

### VGG13

```shell
python ./models/pretrainedModels/vgg13.py
python ./models/embeddedModels/vgg13.py
```
在最后一个全连接层进行嵌入 [4096, 1000]，16.384MB

[VGG13网络结构🔗](./models/modelStructure/vgg13.txt)

实验结果如下：

| RunDate | Model  | InitPara                  | Dataset | Accuracy | Loss   |
| :-: |:------:|:-------------------------:| :-: |:--------:|:------:|
| 20240805 | VGG-13 | vgg13-19584684.pth        | ILSVRC2012_devkit_t12 | 68.14%   | 1.3052 |
| 20240805 | VGG-13 | vgg13_embdedding_8_32.pth | ILSVRC2012_devkit_t12 | 68.14% | 1.3035 |
| 20240805 | VGG-13 | vgg13_embdedding_16_32.pth | ILSVRC2012_devkit_t12 | 68.14% | 1.3041 |
| 20240805 | VGG-13 | vgg13_embdedding_20_32.pth | ILSVRC2012_devkit_t12 | 68.15% | 1.3048 |
| 20240805 | VGG-13 | vgg13_embdedding_21_32.pth | ILSVRC2012_devkit_t12 | 67.96% | 1.3087 |
| 20240805 | VGG-13 | vgg13_embdedding_22_32.pth | ILSVRC2012_devkit_t12 | 67.84% | 1.3209 |
| 20240805 | VGG-13 | vgg13_embdedding_23_32.pth | ILSVRC2012_devkit_t12 | 67.25% | 1.3859 |
| 20240805 | VGG-13 | vgg13_embdedding_24_32.pth | ILSVRC2012_devkit_t12 | 65.42% | 1.8825 |

### VGG16

```shell
python ./models/pretrainedModels/vgg16.py
python ./models/embeddedModels/vgg16.py
```
在最后一个全连接层进行嵌入 [4096, 1000]，16.384MB

[VGG16网络结构🔗](./models/modelStructure/vgg16.txt)

实验结果：

| RunDate | Model  | InitPara              | Dataset | Accuracy | Loss   |
| --- |:------:|:---------------------:| :-: |:--------:|:------:|
| 20240805 | VGG-16 | vgg16-397923af.pth | ILSVRC2012_devkit_t12 | 70.02%   | 1.2210 |
| 20240805 | VGG-16 | vgg16_embdedding_8_32.pth | ILSVRC2012_devkit_t12 | 70.02% | 1.2230 |
| 20240805 | VGG-16 | vgg16_embdedding_16_32.pth | ILSVRC2012_devkit_t12 | 70.03% | 1.2218 |
| 20240805 | VGG-16 | vgg16_embdedding_20_32.pth | ILSVRC2012_devkit_t12 | 69.98% | 1.2225 |
| 20240805 | VGG-16 | vgg16_embdedding_21_32.pth | ILSVRC2012_devkit_t12 | 69.93% | 1.2267 |
| 20240805 | VGG-16 | vgg16_embdedding_22_32.pth | ILSVRC2012_devkit_t12 | 69.72% | 1.2424 |
| 20240805 | VGG-16 | vgg16_embdedding_23_32.pth | ILSVRC2012_devkit_t12 | 69.35% | 1.3138 |
| 20240805 | VGG-16 | vgg16_embdedding_24_32.pth | ILSVRC2012_devkit_t12 | 67.52% | 1.8072 |

### VGG19

```shell
python ./models/pretrainedModels/vgg19.py
python ./models/embeddedModels/vgg19.py
```
在最后一个全连接层进行嵌入 [4096, 1000]，16.384MB

[VGG19网络结构🔗](./models/modelStructure/vgg19.txt)

实验结果：


| RunDate | Model  | InitPara                 | Dataset | Accuracy | Loss   |
| --- |:------:|:------------------------:| :-: |:--------:|:------:|
| 20240805 | VGG-19 | vgg19-dcbb9e9d.pth       | ILSVRC2012_devkit_t12 | 70.68%   | 1.1922 |
| 20240805 | VGG-19 | vgg19_embedding_8_32.pth | ILSVRC2012_devkit_t12 | 70.68% | 1.1923 |
| 20240805 | VGG-19 | vgg19_embedding_16_32.pth | ILSVRC2012_devkit_t120 | 70.68% | 1.1922 |
| 20240805 | VGG-19 | vgg19_embedding_20_32.pth | ILSVRC2012_devkit_t12  | 70.68% | 1.1946 |
| 20240805 | VGG-19 | vgg19_embedding_21_32.pth | ILSVRC2012_devkit_t12  | 70.69% | 1.1967 |
| 20240805 | VGG-19 | vgg19_embedding_22_32.pth | ILSVRC2012_devkit_t12  | 70.44% | 1.2112 |
| 20240805 | VGG-19 | vgg19_embedding_23_32.pth | ILSVRC2012_devkit_t12  |  70.09%  | 1.2840 |
| 20240805 | VGG-19 | vgg19_embedding_24_32.pth | ILSVRC2012_devkit_t12  | 68.19% | 1.7680 |



# Experiment2🚀

For **Functional Test**

Test the correctness of the functionality of the embedded and extracted functions

