# 完整工作流
1. 根据前置对比实验的翻转结果，确定网络的嵌入层
2. 在该层进行嵌入
3. 使用ImageNet1K进行重训练
4. 在2CIFAR100数据集上进行迁移训练
5. 使用Last N bit替换策略，对比指数低3bit嵌入，对比bit翻转率

这部分的实验设置是：

1. LSB和指数替换选择相同的嵌入层【满嵌入】，唯一的区别是一个使用尾数lsb，一个使用指数替换
2. 实验的目的是控制其他条件，只对比修改尾数和修改指数之前的区别。







## ResNet50

同一个bottleneck中，conv2中的参数大小大于conv1和conv3，为了尽可能大得获得嵌入容量，我们在每一个bottleneck中选择中间的一个卷积层conv2

`layer4.2.conv2.weight`
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

`layer4.1.conv2.weight`
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

`layer4.0.conv2.weight`
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)





### 嵌入后性能：

67.07%，使用编码加冗余

重训来呢
