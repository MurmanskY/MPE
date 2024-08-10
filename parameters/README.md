保存各种本项目中可能用到的参数

- allParaEmbedding

早期用于测试在不同的网络结构中的所有层中，的所有高于等于2维的参数，的低Nbit进行翻转

在原始数据集ImageNet上测试分类性能。

用于测试 Lowest N bit 嵌入的对模型性能的影响。





- convEmbedding

在第一个卷积层的weight参数中，low N bit进行翻转

在原始数据集ImageNet上测试分类性能。

用于测试 Lowest N bit 嵌入的对模型性能的影响。





- expReplacement

将由原始参数迁移到其他数据集中重训练得到的结果，在卷积层互换：

resnet50PCAM2Init1.pth：用PCAM重训练得到的参数替换原始参数的第一个卷积层

2.pth：替换第12层

3.pth：替换第123层

4.pth：替换第1234层

5.pth：替换234层





- expXOR

对原始参数的第一个卷积层的指数部分，的低Nbit进行翻转得到的参数





- ImageNetRe

翻转指数之后，在ImageNet1k数据集上进行重训练得到的参数





- init

原始参数





- retrained_ImageNet2CIFAR100

各种模型在CIFAR-100数据集上进行epoch=N的重训练得到的参数





- retrained_ImageNet2FGVCAircraft

各种模型在FGVCAircraft数据集上进行epoch=N的重训练得到的参数





- retrained_ImageNet2GTSRB

各种模型在GTSRB数据集上进行epoch=N的重训练得到的参数





- retrained_ImageNet2OxfordIIITPet

各种模型在OxfordIIITPet数据集上进行epoch=N的重训练得到的参数





- retrained_ImageNet2PCAM

各种模型在PCAM数据集上进行epoch=N的重训练得到的参数





- weightsEmbedding

在各种模型的最后一个全连接层的weight参数的low N bit进行嵌入实验