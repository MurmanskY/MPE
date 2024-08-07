# MPE

Model Parameters Embedding for AI models attack

[TOC]



# Model Download Path✅

🔗🔗🔗

**introduction** to `torchvision.models` : https://pytorch.org/vision/stable/models.html

use the models and pro-trained weights from torchvision Lib.

example:
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/mac/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth



# Malware

- Lazarus 20.9MB
- test1.jpeg 864KB
- test1.jpeg 3.1MB





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

Testing the performance of embedding harmful information on the **parameters of the first convolutional** layer

using the **ImageNet Validation** dataset

## ResNet Series

### ResNet18

[ResNet18网络结构🔗](./models/modelStructure/resnet18.txt)

| RunDate  |   Model   |    Para     | Accuracy |  Loss  |
| :------: | :-------: | :---------: | :------: | :----: |
| 20240806 | ResNet-18 | InitialPara |  67.27%  | 1.3528 |
| 20240806 | ResNet-18 |    16_32    |  67.30%  | 1.3527 |
| 20240806 | ResNet-18 |    19_32    |  67.22%  | 1.3563 |
| 20240806 | ResNet-18 |    20_32    |  67.12%  | 1.3591 |
| 20240806 | ResNet-18 |    21_32    |  66.22%  | 1.4028 |
| 20240806 | ResNet-18 |    22_32    |  57.47%  | 1.8530 |
| 20240806 | ResNet-18 |    23_32    |  35.39%  | 3.3473 |
| 20240806 | ResNet-18 |    24_32    |  8.98%   | 7.2026 |

### ResNet50

[ResNet50网络结构🔗](./models/modelStructure/resnet50.txt)

| RunDate  |   Model   |    Para     | Accuracy |  Loss   |
| :------: | :-------: | :---------: | :------: | :-----: |
| 20240806 | ResNet-50 | InitialPara |  80.12%  | 1.4179  |
| 20240806 | ResNet-50 |    16_32    |  80.13%  | 1.4182  |
| 20240806 | ResNet-50 |    19_32    |  79.87%  | 1.4331  |
| 20240806 | ResNet-50 |    20_32    |  78.82%  | 1.4833  |
| 20240806 | ResNet-50 |    21_32    |  76.27%  | 1.5902  |
| 20240806 | ResNet-50 |    22_32    |  65.63%  | 2.0357  |
| 20240806 | ResNet-50 |    23_32    |  57.54%  | 2.4152  |
| 20240806 | ResNet-50 |    24_32    |  9.07%   | 10.0144 |

### ResNet101

[ResNet101网络结构🔗](./models/modelStructure/resnet101.txt)

| RunDate  |   Model    |    Para     | Accuracy |  Loss  |
| :------: | :--------: | :---------: | :------: | :----: |
| 20240806 | ResNet-101 | InitialPara |  80.94%  | 0.9221 |
| 20240806 | ResNet-101 |    16_32    |  80.95%  | 0.9221 |
| 20240806 | ResNet-101 |    19_32    |  80.87%  | 0.9239 |
| 20240806 | ResNet-101 |    20_32    |  80.75%  | 0.9309 |
| 20240806 | ResNet-101 |    21_32    |  79.41%  | 0.9939 |
| 20240806 | ResNet-101 |    22_32    |  73.24%  | 1.2881 |
| 20240806 | ResNet-101 |    23_32    |  62.81%  | 1.8137 |
| 20240806 | ResNet-101 |    24_32    |  37.95%  | 3.3804 |

## VGG Series

### VGG11

[VGG11网络结构🔗](./models/modelStructure/vgg11.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-11 | InitialPara |  66.88%  | 1.3540 |
| 20240806 | VGG-11 |    16_32    |  66.91%  | 1.3545 |
| 20240806 | VGG-11 |    19_32    |  66.63%  | 1.3663 |
| 20240806 | VGG-11 |    20_32    |  64.29%  | 1.4808 |
| 20240806 | VGG-11 |    21_32    |  60.45%  | 1.6614 |
| 20240806 | VGG-11 |    22_32    |  44.65%  | 2.5821 |
| 20240806 | VGG-11 |    23_32    |  33.32%  | 3.3020 |
| 20240806 | VGG-11 |    24_32    |  3.93%   | 8.6345 |

### VGG13

[VGG13网络结构🔗](./models/modelStructure/vgg13.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-13 | InitialPara |  68.14%  | 1.3052 |
| 20240806 | VGG-13 |    16_32    |  68.15%  | 1.3049 |
| 20240806 | VGG-13 |    19_32    |  68.11%  | 1.3051 |
| 20240806 | VGG-13 |    20_32    |  67.65%  | 1.3266 |
| 20240806 | VGG-13 |    21_32    |  67.07%  | 1.3522 |
| 20240806 | VGG-13 |    22_32    |  58.91%  | 1.7601 |
| 20240806 | VGG-13 |    23_32    |  54.99%  | 1.9757 |
| 20240806 | VGG-13 |    24_32    |  37.88%  | 3.2278 |

### VGG16

[VGG16网络结构🔗](./models/modelStructure/vgg16.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-16 | InitialPara |  70.02%  | 1.2210 |
| 20240806 | VGG-16 |    16_32    |  70.01%  | 1.2216 |
| 20240806 | VGG-16 |    19_32    |  69.91%  | 1.2270 |
| 20240806 | VGG-16 |    20_32    |  69.77%  | 1.2306 |
| 20240806 | VGG-16 |    21_32    |  69.13%  | 1.2609 |
| 20240806 | VGG-16 |    22_32    |  61.51%  | 1.6538 |
| 20240806 | VGG-16 |    23_32    |  56.89%  | 1.8700 |
| 20240806 | VGG-16 |    24_32    |  24.46%  | 4.8595 |

### VGG19

[VGG19网络结构🔗](./models/modelStructure/vgg19.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-19 | InitialPara |  70.68%  | 1.1922 |
| 20240806 | VGG-19 |    16_32    |  70.68%  | 1.1920 |
| 20240806 | VGG-19 |    19_32    |  70.63%  | 1.1955 |
| 20240806 | VGG-19 |    20_32    |  70.44%  | 1.2059 |
| 20240806 | VGG-19 |    21_32    |  69.03%  | 1.2730 |
| 20240806 | VGG-19 |    22_32    |  65.51%  | 1.4517 |
| 20240806 | VGG-19 |    23_32    |  55.29%  | 2.0323 |
| 20240806 | VGG-19 |    24_32    |  40.71%  | 3.0027 |









# Experiment3🚀

对参数中所有高维（维度大于2）的参数的最低比特进行嵌入测试：



## ResNet Series

### ResNet18

[ResNet18网络结构🔗](./models/modelStructure/resnet18.txt)

| RunDate  |   Model   |    Para     | Accuracy |  Loss  |
| :------: | :-------: | :---------: | :------: | :----: |
| 20240806 | ResNet-18 | InitialPara |  67.27%  | 1.3528 |
| 20240806 | ResNet-18 |    16_32    |  67.30%  | 1.3528 |
| 20240806 | ResNet-18 |    17_32    |  67.28%  | 1.3539 |
| 20240806 | ResNet-18 |    18_32    |  67.20%  | 1.3565 |
| 20240806 | ResNet-18 |    19_32    |  67.13%  | 1.3631 |
| 20240806 | ResNet-18 |    20_32    |  66.26%  | 1.3953 |
| 20240806 | ResNet-18 |    21_32    |  61.99%  | 1.5967 |
| 20240806 | ResNet-18 |    22_32    |  37.74%  | 3.1640 |

### ResNet50

[ResNet50网络结构🔗](./models/modelStructure/resnet50.txt)

| RunDate  |   Model   |    Para     | Accuracy |  Loss  |
| :------: | :-------: | :---------: | :------: | :----: |
| 20240806 | ResNet-50 | InitialPara |  80.12%  | 1.4179 |
| 20240806 | ResNet-50 |    16_32    |  80.12%  | 1.4190 |
| 20240806 | ResNet-50 |    17_32    |  80.16%  | 1.4128 |
| 20240806 | ResNet-50 |    18_32    |  80.14%  | 1.4116 |
| 20240806 | ResNet-50 |    19_32    |  79.16%  | 1.4404 |
| 20240806 | ResNet-50 |    20_32    |  75.14%  | 1.5994 |
| 20240806 | ResNet-50 |    21_32    |  55.08%  | 2.5866 |
| 20240806 | ResNet-50 |    22_32    |  4.92%   | 6.5583 |

### ResNet101

[ResNet101网络结构🔗](./models/modelStructure/resnet101.txt)

| RunDate  |   Model    |    Para     | Accuracy |  Loss  |
| :------: | :--------: | :---------: | :------: | :----: |
| 20240806 | ResNet-101 | InitialPara |  80.94%  | 0.9221 |
| 20240806 | ResNet-101 |    16_32    |  80.92%  | 0.9227 |
| 20240806 | ResNet-101 |    17_32    |  80.92%  | 0.9248 |
| 20240806 | ResNet-101 |    18_32    |  80.78%  | 0.9305 |
| 20240806 | ResNet-101 |    19_32    |  80.65%  | 0.9432 |
| 20240806 | ResNet-101 |    20_32    |  79.86%  | 0.9712 |
| 20240806 | ResNet-101 |    21_32    |  72.17%  | 1.3286 |
| 20240806 | ResNet-101 |    22_32    |  19.51%  | 5.3451 |

## VGG Series

### VGG11

[VGG11网络结构🔗](./models/modelStructure/vgg11.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-11 | InitialPara |  66.88%  | 1.3540 |
| 20240806 | VGG-11 |    16_32    |  66.93%  | 1.3546 |
| 20240806 | VGG-11 |    17_32    |  66.88%  | 1.3546 |
| 20240806 | VGG-11 |    18_32    |  66.79%  | 1.3593 |
| 20240806 | VGG-11 |    19_32    |  66.65%  | 1.3688 |
| 20240806 | VGG-11 |    20_32    |  64.04%  | 1.4937 |
| 20240806 | VGG-11 |    21_32    |  59.72%  | 1.7133 |
| 20240806 | VGG-11 |    22_32    |  42.89%  | 3.0529 |

### VGG13

[VGG13网络结构🔗](./models/modelStructure/vgg13.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-13 | InitialPara |  68.14%  | 1.3052 |
| 20240806 | VGG-13 |    16_32    |  68.13%  | 1.3044 |
| 20240806 | VGG-13 |    17_32    |  68.14%  | 1.3047 |
| 20240806 | VGG-13 |    18_32    |  68.10%  | 1.3060 |
| 20240806 | VGG-13 |    19_32    |  67.97%  | 1.3081 |
| 20240806 | VGG-13 |    20_32    |  67.52%  | 1.3351 |
| 20240806 | VGG-13 |    21_32    |  66.20%  | 1.4033 |
| 20240806 | VGG-13 |    22_32    |  54.69%  | 2.3744 |

### VGG16

[VGG16网络结构🔗](./models/modelStructure/vgg16.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-16 | InitialPara |  70.02%  | 1.2210 |
| 20240806 | VGG-16 |    16_32    |  69.98%  | 1.2226 |
| 20240806 | VGG-16 |    17_32    |  69.96%  | 1.2221 |
| 20240806 | VGG-16 |    18_32    |  69.98%  | 1.2240 |
| 20240806 | VGG-16 |    19_32    |  69.82%  | 1.2306 |
| 20240806 | VGG-16 |    20_32    |  69.57%  | 1.2415 |
| 20240806 | VGG-16 |    21_32    |  68.36%  | 1.3269 |
| 20240806 | VGG-16 |    22_32    |  56.11%  | 2.4337 |

### VGG19

[VGG19网络结构🔗](./models/modelStructure/vgg19.txt)

| RunDate  | Model  |    Para     | Accuracy |  Loss  |
| :------: | :----: | :---------: | :------: | :----: |
| 20240806 | VGG-19 | InitialPara |  70.68%  | 1.1922 |
| 20240806 | VGG-19 |    16_32    |  70.67%  | 1.1925 |
| 20240806 | VGG-19 |    17_32    |  70.66%  | 1.1936 |
| 20240806 | VGG-19 |    18_32    |  70.70%  | 1.1930 |
| 20240806 | VGG-19 |    19_32    |  70.65%  | 1.1992 |
| 20240806 | VGG-19 |    20_32    |  70.40%  | 1.2225 |
| 20240806 | VGG-19 |    21_32    |  68.26%  | 1.3580 |
| 20240806 | VGG-19 |    22_32    |  62.23%  | 1.9975 |





# 重训练

对深度深度神经网络进行重训练（微调）时，训练模型哪些层距取决于几个因素：数据集的相似性、数据规模、计算资源以及具体任务需求：

- **重训练内容：**

  1. **全连接层：**重训练全连接层是最常见的做法，因为直接与分类任务相关。通常用新的全连接层替换预训练模型的最后一层，以适应新任务的输出类别数量。
  2. **高层卷积层：**如果新任务的数据与预测数据存在一定差异，通常重训练最后几层卷积层。这些层学习到的特征与特定任务相关，通过重训练可以使模型更好适应新的数据特征。
  3. **所有层：**如果新任务数据与原数据差异极大，且数据集足够大，可能需要重训练所有层。耗费计算资源，但是提供更大灵活性。

- **重训练低层卷积层需要考虑因素**

  > 低层卷积层通常学习到的是通用的图像特征，例如边缘、纹理、颜色变化等。这些特征在许多视觉任务中是共享的。因此是否重训练这些层需要根据具体情况进行判断。

  1. **数据集相似性：**相似数据集：如果新任务的数据集与预训练的数据集非常相似，一般不需要重训练底层卷积层。这些层能够有效捕获许多通用特征；不同数据集：如果新数据集与预训练数据集差异较大，例如从自然图像转到医学图像，或从真实图像转到手绘插图，重训练底层卷积层会更有利于模型适应新的数据特征。
  2. **数据集规模：**小数据集：对于小数据集，通常选择冻结低层卷积层，以减少过拟合的风险并利用预训练模型已有的特征；大数据集：对于较大规模的数据集，尤其是新任务与预训练任务差异较大时，可以考虑重训练低层卷积层。
  3. **计算资源：**有限的计算资源：冻结低层卷积层可以减少计算开销和训练时间，因为这些层通常是计算密集型的；充足的计算资源：可以选择重训练所有层，包括低层卷积层，以充分利用计算资源提高模型性能。



- **汉明距离**

- **任务迁移：**迁移学习应用广泛，设计从大型、广泛的数据集迁移到特定、较小的数据集，或在相似任务之间进行知识迁移。
  从ImageNet迁移到特定领域的图像数据集，通常用于预训练深度学习模型。这些模型随后可以迁移到具体的小型数据集上，如医疗图像识别、卫星图像分析等。

  目标数据集示例：医疗影像数据集（如肺部X光图像用于检测肺炎）、车辆识别、动物识别。

- **从ImageNet到医疗影像**

  从ImageNet到医疗影像数据集进行迁移学习时，参数变化的程度取决于多个因素，包括任务的不同、数据特性的差异，以及迁移学习策略的选择。以下是一些主要的考虑点：

  1. **数据特性的差异**。**图像内容和上下文**：ImageNet包含多样的自然图像，覆盖从动物到日常物品的广泛类别，而医疗影像（如CT、MRI或X光图）主要是人体内部结构的图像，具有更高的结构和内容专业性；**图像格式和处理**：医疗图像可能是灰度图，分辨率和对比度也与常规彩色照片不同，这需要在预处理阶段进行调整。
  2. **模型架构的调整**。**特征提取层**：在ImageNet上预训练的模型，如VGG、ResNet等，其卷积层通常不需要大幅修改，因为它们负责从图像中提取通用特征；**分类或决策层**：最后的全连接层或分类层往往需要根据医疗图像的具体任务进行重大调整或完全替换，以适应不同的输出需求（如疾病分类、异常检测等）。
  3. **参数微调和再训练**。**微调程度**：尽管模型的初级和中级特征提取器可能保持相对稳定，但针对特定医疗任务，通常需要对更高层或特定层的参数进行微调；**学习策略**：可以选择冻结初级卷积层的参数，仅对高层进行微调；或者对整个网络进行细致的调整，特别是在医疗图像数据上。
  4. **训练数据量和标注**。**数据量**：医疗图像数据集通常比ImageNet小得多，这可能限制了对深层网络大规模重新训练的可能性；**数据标注**：医疗数据的标注需要专业知识，且标注成本高，这可能影响训练数据的质量和数量。
  5. **总结**：迁移学习中，从ImageNet到医疗影像的参数变化可能从较小（如仅微调最后几层）到较大（如完全重新设计分类层和深层结构）。因此，具体变化程度取决于目标任务的复杂性和所需的精确度。通常，迁移学习策略旨在通过利用在ImageNet上学到的通用视觉特征，减少在医疗领域所需的训练数据量，同时优化模型在特定医疗任务上的性能。



# Experiment4🚀

测试从ImageNet1000迁移到CIFAR-100数据集，各个模型的参数会如何变化。

实验选取三种情况：欠拟合，拟合，过拟合来探究参数的变化；同时探究参数的每一层是如何变化的

## ResNet Series

### ResNet18

[ResNet18网络结构🔗](./models/modelStructure/resnet18.txt)







### ResNet50

[ResNet50网络结构🔗](./models/modelStructure/resnet50.txt)



### ResNet101

[ResNet101网络结构🔗](./models/modelStructure/resnet101.txt)



## VGG Series

### VGG11

[VGG11网络结构🔗](./models/modelStructure/vgg11.txt)



### VGG13

[VGG13网络结构🔗](./models/modelStructure/vgg13.txt)



### VGG16

[VGG16网络结构🔗](./models/modelStructure/vgg16.txt)



### VGG19

[VGG19网络结构🔗](./models/modelStructure/vgg19.txt)
