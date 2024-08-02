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

| RunDate | Model     | InitPara | Dataset | Accuracy | Loss |
| - |- | - | - |--------| - |
| 20240802 | resnet-18 | resenet18-f37072fd.pth | ILSVRC2012_devkit_t12 | 67.27% | 1.3545 |


## resnet50
准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet50.py
```

| RunDate | Model     | InitPara | Dataset | Accuracy | Loss   |
| - |-----------| - | - |----------|--------|
| 20240802 | resnet-50 | resnet50-11ad3fa6.pth | ILSVRC2012_devkit_t12 | 80.12%   | 1.4183 |


## resnet101
准确率不会有变化，因为进行的是一个确定参数的推理过
```shell
python ./models/pretrainedModels/resnet101.py
```

| RunDate | Model      | InitPara | Dataset | Accuracy | Loss   |
| - |------------| - | - |----------|--------|
| 20240802 | resnet-101 | resnet101-cd907fc2.pth | ILSVRC2012_devkit_t12 | 80.94%   | 0.9227 |


# VGG系列
## VGG11
```shell
python ./models/pretrainedModels/vgg11.py
```

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |--------|--------|
| 20240802 | VGG-11 | vgg11-8a719046.pth | ILSVRC2012_devkit_t12 | 66.88% | 1.3540 |

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

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |--------|--------|
| 20240802 | VGG-16 | vgg16-397923af.pth | ILSVRC2012_devkit_t12 | 70.02% | 1.2218 |

## VGG19
```shell
python ./models/pretrainedModels/vgg19.py
```

| RunDate | Model  | InitPara | Dataset | Accuracy | Loss   |
| - |--------| - | - |--------|--------|
| 20240802 | VGG-19 | vgg19-dcbb9e9d.pth | ILSVRC2012_devkit_t12 | 70.68% | 1.1921 |