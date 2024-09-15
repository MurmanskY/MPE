# 完整工作流
1. 根据前置对比实验的翻转结果，确定网络的嵌入层
2. 在该层进行嵌入
3. 使用ImageNet1K进行重训练
4. 在2CIFAR100数据集上进行迁移训练
5. 使用Last N bit替换策略，对比指数低3bit嵌入，对比bit翻转率

这部分的实验设置是：

1. LSB和指数替换选择相同的嵌入层【满嵌入】，唯一的区别是一个使用尾数lsb，一个使用指数替换
2. 实验的目的是控制其他条件，只对比修改尾数和修改指数之前的区别。



[TOC]





## ResNet50

同一个bottleneck中，conv2中的参数大小大于conv1和conv3，为了尽可能大得获得嵌入容量，我们在每一个bottleneck中选择中间的一个卷积层conv2

`layer4.2.conv2.weight`
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

`layer4.1.conv2.weight`
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

`layer4.0.conv2.weight`
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)





### 嵌入后性能：

原始性能：80.12%

67.07%，使用编码加冗余



### 重训练恢复结果：

78.84%

```
Epoch 1/1
----------
train Loss: 1.1628 Acc: 0.7782
val Loss: 0.8137 Acc: 0.7884
Test Accuracy: 78.84%
Test Loss: 0.8137
Test total: 50000
Test correct: 39420
```



### 翻转结果：

- CIFAR100：
  6.28% 6.53% 4.98% （没有纠错）
  使用纠错后总的bit翻转率为：1,  总bit数为：71472，翻转率为0.001399%，SNR为：97.08

  ```
  Files already downloaded and verified
  Files already downloaded and verified
  Epoch 1/5
  ----------
  train Loss: 2.0955 Acc: 0.5142
  val Loss: 0.8804 Acc: 0.7412
  Epoch 2/5
  ----------
  train Loss: 0.7468 Acc: 0.7792
  val Loss: 0.6629 Acc: 0.8010
  Epoch 3/5
  ----------
  train Loss: 0.5181 Acc: 0.8424
  val Loss: 0.6066 Acc: 0.8147
  Epoch 4/5
  ----------
  train Loss: 0.3779 Acc: 0.8842
  val Loss: 0.5759 Acc: 0.8288
  Epoch 5/5
  ----------
  train Loss: 0.2835 Acc: 0.9149
  val Loss: 0.5709 Acc: 0.8320
  Test Accuracy: 83.20%
  Test Loss: 0.5714
  Test total: 10000
  Test correct: 8320
  ```

  

- PCAM：
  7.51% 7.29% 5.54%
  使用纠错后总bit翻转率为：2，总bit数为：71472，翻转率为0.0028%，SNR为：91.06

  ```
  Epoch 1/5
  ----------
  train Loss: 0.2036 Acc: 0.9246
  val Loss: 0.3605 Acc: 0.8680
  Epoch 2/5
  ----------
  train Loss: 0.1030 Acc: 0.9635
  val Loss: 0.3935 Acc: 0.8717
  Epoch 3/5
  ----------
  train Loss: 0.0691 Acc: 0.9757
  val Loss: 0.4481 Acc: 0.8705
  Epoch 4/5
  ----------
  train Loss: 0.0443 Acc: 0.9847
  val Loss: 0.5830 Acc: 0.8553
  Epoch 5/5
  ----------
  train Loss: 0.0288 Acc: 0.9903
  val Loss: 0.5524 Acc: 0.8760
  Test Accuracy: 87.60%
  Test Loss: 0.5524
  Test total: 32768
  Test correct: 28706
  ```

  



### DenseNet



