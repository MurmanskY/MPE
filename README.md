# MPE
Model Parameters Embedding for AI models attack


# model download path
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /Users/mac/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

# resnet 系列
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
## resnet18
准确率不会有变化，因为进行的是一个确定参数的推理过程
```shell
python ./models/pretrainedModels/resnet18.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 512]，每个数据类型是torch.float32，数据总量是2.048MB

| RunDate | Model     | InitPara                      | Dataset | Accuracy | Loss   |
| - |- |-------------------------------| - |----------|--------|
| 20240802 | resnet-18 | resenet18-f37072fd.pth        | ILSVRC2012_devkit_t12 | 67.27%   | 1.3545 |
| 20240802 | resnet-18 | resenet18_embedding_1_32.pth  | ILSVRC2012_devkit_t12 | 67.27%   | 1.3545 |
| 20240802 | resnet-18 | resenet18_embedding_4_32.pth  | ILSVRC2012_devkit_t12 | 67.27%   | 1.3532 |
| 20240802 | resnet-18 | resenet18_embedding_6_32.pth  | ILSVRC2012_devkit_t12 | 67.30%   | 1.3533 |
| 20240802 | resnet-18 | resenet18_embedding_7_32.pth  | ILSVRC2012_devkit_t12 | 66.87%   | 1.3700 |
| 20240802 | resnet-18 | resenet18_embedding_8_32.pth  | ILSVRC2012_devkit_t12 | 52.65%   | 2.4685 |
| 20240802 | resnet-18 | resenet18_embedding_8_32.pth  | ILSVRC2012_devkit_t12 | 0.04%     | 11.336 |
| 20240802 | resnet-18 | resenet18_embedding_10_32.pth | ILSVRC2012_devkit_t12 | 0%       | nan    |



## resnet50
准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet50.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 2048]，每个数据类型是torch.float32，数据总量是8.192MB

| RunDate | Model     | InitPara                    | Dataset | Accuracy | Loss   |
| - |-----------|-----------------------------| - |----------|--------|
| 20240802 | resnet-50 | resnet50-11ad3fa6.pth       | ILSVRC2012_devkit_t12 | 80.12%   | 1.4183 |
| 20240802 | resnet-50 | resnet50_embedding_4_32.pth | ILSVRC2012_devkit_t12 | 80.12%   | 1.4179 |
| 20240802 | resnet-50 | resnet50_embedding_6_32.pth | ILSVRC2012_devkit_t12 | 80.14%   | 1.4185 |
| 20240802 | resnet-50 | resnet50_embedding_6_32.pth | ILSVRC2012_devkit_t12 | 79.72%   | 1.4054 |
| 20240802 | resnet-50 | resnet50_embedding_8_32.pth | ILSVRC2012_devkit_t12 | 66.91%   | 1.7324 |



## resnet101
准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet101.py
```
原始weights结构：只有一个全连接层，在最后一层，weights的大小是[1000, 2048]，每个数据类型是torch.float32，数据总量是8.192MB

| RunDate | Model      | InitPara                     | Dataset | Accuracy | Loss   |
| - |------------|------------------------------| - |----------|--------|
| 20240802 | resnet-101 | resnet101-cd907fc2.pth       | ILSVRC2012_devkit_t12 | 80.94%   | 0.9227 |
| 20240802 | resnet-101 | resnet101_embedding_4_32.pth | ILSVRC2012_devkit_t12 | 80.94%   | 0.9231 |
| 20240802 | resnet-101 | resnet101_embedding_6_32.pth | ILSVRC2012_devkit_t12 | 80.92%   | 0.9223 |
| 20240802 | resnet-101 | resnet101_embedding_7_32.pth | ILSVRC2012_devkit_t12 | 80.52%   | 0.9281 |
| 20240802 | resnet-101 | resnet101_embedding_8_32.pth | ILSVRC2012_devkit_t12 | 69.80%   | 1.6811 |



# VGG系列
## VGG11
```shell
python ./models/pretrainedModels/vgg11.py
```
原始效果

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |--------|--------|
| 20240802 | VGG-11 | vgg11-8a719046.pth | ILSVRC2012_devkit_t12 | 66.88% | 1.3540 |

在最后一个全连接层进行嵌入 [4096, 1000]，16.384MB

| RunDate | Model  | InitPara                  | Dataset | Accuracy | Loss   |
| - |--------|---------------------------| - |----------|--------|
| 20240802 | VGG-11 | vgg11_embdedding_4_32.pth | ILSVRC2012_devkit_t12 | 66.88%   | 1.3540 |
| 20240802 | VGG-11 | vgg11_embdedding_6_32.pth | ILSVRC2012_devkit_t12 | 66.91%   | 1.3553 |
| 20240802 | VGG-11 | vgg11_embdedding_7_32.pth | ILSVRC2012_devkit_t12 | 66.85%   | 1.3566 |
| 20240802 | VGG-11 | vgg11_embdedding_8_32.pth | ILSVRC2012_devkit_t12 | 64.62%   | 1.6994 |


## VGG13
```shell
python ./models/pretrainedModels/vgg13.py
```

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |----------|--------|
| 20240802 | VGG-13 | vgg13-19584684.pth | ILSVRC2012_devkit_t12 | 68.14%   | 1.3035 |

## VGG16
```shell
python ./models/pretrainedModels/vgg16.py
```

| RunDate | Model  | InitPara                  | Dataset | Accuracy | Loss   |
| - |--------|---------------------------| - |----------|--------|
| 20240802 | VGG-16 | vgg11_embdedding_4_32.pth | ILSVRC2012_devkit_t12 | 70.02%   | 1.2218 |
| 20240802 | VGG-16 | vgg11_embdedding_6_32.pth | ILSVRC2012_devkit_t12 | 70.02%   | 1.2213 |
| 20240802 | VGG-16 | vgg11_embdedding_7_32.pth | ILSVRC2012_devkit_t12 | 69.97%   | 1.2251 |
| 20240802 | VGG-16 | vgg11_embdedding_8_32.pth | ILSVRC2012_devkit_t12 | 67.91%   | 1.5585 |

## VGG19
```shell
python ./models/pretrainedModels/vgg19.py
```

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |--------|--------|
| 20240802 | VGG-19 | vgg19-dcbb9e9d.pth | ILSVRC2012_devkit_t12 | 70.68% | 1.1921 |


# 测试在全连接层的weights中嵌入有害信息
