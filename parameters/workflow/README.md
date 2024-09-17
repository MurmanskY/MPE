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
  **6.28% 6.53% 4.98% （没有纠错）**
  **使用纠错后总的bit翻转率为：1,  总bit数为：71472，翻转率为0.001399%，SNR为：97.08**

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
  **7.51% 7.29% 5.54%**
  **使用纠错后总bit翻转率为：2，总bit数为：71472，翻转率为0.0028%，SNR为：91.06**

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

  



## ConvNeXt_6层

使用的是convnext_base

```
layers = ["features.7.0.block.3.weight",
          "features.7.0.block.5.weight",
          "features.7.1.block.3.weight",
          "features.7.1.block.5.weight",
          "features.7.2.block.3.weight",
          "features.7.2.block.5.weight"]
Linear(in_features=1024, out_features=4096, bias=True)
          (4): GELU(approximate='none')
          (5): Linear(in_features=4096, out_features=1024, bias=True)
```



### 嵌入后性能：

原始性能：83.72%

嵌入后性能：83.63%





### 翻转结果：

- CIFAR100
  
  单层大小：5957B，即47656bit
  六层容量大小：35742B，即285936bit
  **0% 0% 0% 0% 0% 0% 【6 * 6KB】**
  **SNR = 无穷**
  
  ```
  Files already downloaded and verified
  Files already downloaded and verified
  Epoch 1/5
  ----------
  train Loss: 2.7225 Acc: 0.4575
  val Loss: 1.1614 Acc: 0.7477
  Epoch 2/5
  ----------
  train Loss: 0.9714 Acc: 0.7717
  val Loss: 0.6835 Acc: 0.8135
  Epoch 3/5
  ----------
  train Loss: 0.6705 Acc: 0.8182
  val Loss: 0.5448 Acc: 0.8377
  Epoch 4/5
  ----------
  train Loss: 0.5526 Acc: 0.8397
  val Loss: 0.4929 Acc: 0.8488
  Epoch 5/5
  ----------
  train Loss: 0.4767 Acc: 0.8581
  val Loss: 0.4657 Acc: 0.8551
  Test Accuracy: 85.51%
  Test Loss: 0.4663
  Test total: 10000
  Test correct: 8551
  ```
  
  



- PCAM
  
  单层大小：5957B，即47656bit
  六层容量大小：35742B，即285936bit
  **0% 0% 0% 0% 0% 0%【36KB】**
  **SNR=无穷**
  
  ```
  Epoch 1/5
  ----------
  train Loss: 0.1913 Acc: 0.9257
  val Loss: 0.3246 Acc: 0.8874
  Epoch 2/5
  ----------
  train Loss: 0.1255 Acc: 0.9550
  val Loss: 0.2803 Acc: 0.8985
  Epoch 3/5
  ----------
  train Loss: 0.1047 Acc: 0.9633
  val Loss: 0.3983 Acc: 0.8843
  Epoch 4/5
  ----------
  train Loss: 0.0923 Acc: 0.9680
  val Loss: 0.3724 Acc: 0.8961
  Epoch 5/5
  ----------
  train Loss: 0.0826 Acc: 0.9721
  val Loss: 0.4042 Acc: 0.8956
  Test Accuracy: 89.56%
  Test Loss: 0.4042
  Test total: 32768
  Test correct: 29348
  ```
  
  





## ConvNeXt_24层

使用的是convnext_base

```
layers = ["features.5.18.block.3.weight",
              "features.5.18.block.5.weight",
              "features.5.19.block.3.weight",
              "features.5.19.block.5.weight",
              "features.5.20.block.3.weight",
              "features.5.20.block.5.weight",
              "features.5.21.block.3.weight",
              "features.5.21.block.5.weight",
              "features.5.22.block.3.weight",
              "features.5.22.block.5.weight",
              "features.5.23.block.3.weight",
              "features.5.23.block.5.weight",
              "features.5.24.block.3.weight",
              "features.5.24.block.5.weight",
              "features.5.25.block.3.weight",
              "features.5.25.block.5.weight",
              "features.5.26.block.3.weight",
              "features.5.26.block.5.weight",
              "features.7.0.block.3.weight",
              "features.7.0.block.5.weight",
              "features.7.1.block.3.weight",
              "features.7.1.block.5.weight",
              "features.7.2.block.3.weight",
              "features.7.2.block.5.weight"]
          
          (3): Linear(in_features=512, out_features=2048, bias=True)
          (5): Linear(in_features=4096, out_features=1024, bias=True)

```

interval： 8

correct：7

总容量是：98292B，92KB

```
Byte Num: 2340
File './malware/convnext_base_l1' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l2' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l3' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l4' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l5' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l6' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l7' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l8' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l9' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l10' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l11' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l12' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l13' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l14' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l15' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l16' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l17' generated with 18720 bits.
Byte Num: 2340
File './malware/convnext_base_l18' generated with 18720 bits.
Byte Num: 9362
File './malware/convnext_base_l19' generated with 74896 bits.
Byte Num: 9362
File './malware/convnext_base_l20' generated with 74896 bits.
Byte Num: 9362
File './malware/convnext_base_l21' generated with 74896 bits.
Byte Num: 9362
File './malware/convnext_base_l22' generated with 74896 bits.
Byte Num: 9362
File './malware/convnext_base_l23' generated with 74896 bits.
Byte Num: 9362
File './malware/convnext_base_l24' generated with 74896 bits.
```



### 嵌入后性能：

原始性能：83.72%

嵌入后的性能：Test Accuracy: 81.52%



### 翻转结果：

- CIFAR100

  pos: 17121 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  pos: 7115 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  pos: 2069 initBit: 0 extractedBit: 1
  pos: 4442 initBit: 0 extractedBit: 1
  different bit Num between the two files:  2
  different bit Num between the two files:  0
  pos: 3677 initBit: 0 extractedBit: 1
  pos: 5765 initBit: 0 extractedBit: 1
  different bit Num between the two files:  2
  different bit Num between the two files:  0
  pos: 6264 initBit: 0 extractedBit: 1
  pos: 11118 initBit: 0 extractedBit: 1
  pos: 16938 initBit: 0 extractedBit: 1
  different bit Num between the two files:  3
  different bit Num between the two files:  0
  pos: 5180 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  pos: 1680 initBit: 0 extractedBit: 1
  pos: 17531 initBit: 0 extractedBit: 1
  different bit Num between the two files:  2
  different bit Num between the two files:  0
  pos: 993 initBit: 0 extractedBit: 1
  pos: 995 initBit: 0 extractedBit: 1
  different bit Num between the two files:  2
  different bit Num between the two files:  0
  pos: 6797 initBit: 0 extractedBit: 1
  pos: 9578 initBit: 0 extractedBit: 1
  pos: 11231 initBit: 0 extractedBit: 1
  pos: 11235 initBit: 0 extractedBit: 1
  pos: 15324 initBit: 0 extractedBit: 1
  different bit Num between the two files:  5
  pos: 9994 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  pos: 46825 initBit: 0 extractedBit: 1
  pos: 61990 initBit: 0 extractedBit: 1
  pos: 74637 initBit: 0 extractedBit: 1
  different bit Num between the two files:  3
  different bit Num between the two files:  0
  pos: 39962 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  pos: 43896 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  pos: 9961 initBit: 0 extractedBit: 1
  pos: 17337 initBit: 0 extractedBit: 1
  pos: 22430 initBit: 0 extractedBit: 1
  pos: 54710 initBit: 1 extractedBit: 0
  different bit Num between the two files:  4
  different bit Num between the two files:  0

  **一共发生29bit错误，总量786336bit**

  **bit翻转率为：0.00369%**
  **SNR为：88.66**

  ```
  iles already downloaded and verified
  Epoch 1/5
  ----------
  train Loss: 2.7776 Acc: 0.4405
  val Loss: 1.2000 Acc: 0.7384
  Epoch 2/5
  ----------
  train Loss: 1.0232 Acc: 0.7609
  val Loss: 0.6917 Acc: 0.8080
  Epoch 3/5
  ----------
  train Loss: 0.7013 Acc: 0.8129
  val Loss: 0.5523 Acc: 0.8361
  Epoch 4/5
  ----------
  train Loss: 0.5717 Acc: 0.8356
  val Loss: 0.5190 Acc: 0.8438
  Epoch 5/5
  ----------
  train Loss: 0.4948 Acc: 0.8542
  val Loss: 0.4603 Acc: 0.8608
  Test Accuracy: 86.08%
  Test Loss: 0.4615
  Test total: 10000
  Test correct: 8608
  ```

  

- PCAM

  different bit Num between the two files:  0
  different bit Num between the two files:  0
  pos: 15018 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  pos: 5570 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  pos: 17982 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  pos: 11118 initBit: 0 extractedBit: 1
  pos: 13591 initBit: 0 extractedBit: 1
  pos: 13874 initBit: 0 extractedBit: 1
  pos: 15791 initBit: 0 extractedBit: 1
  different bit Num between the two files:  4
  different bit Num between the two files:  0
  pos: 1030 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  pos: 5976 initBit: 0 extractedBit: 1
  pos: 12941 initBit: 0 extractedBit: 1
  different bit Num between the two files:  2
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  different bit Num between the two files:  0
  pos: 1807 initBit: 0 extractedBit: 1
  different bit Num between the two files:  1
  different bit Num between the two files:  0

  **一共发生19bit错误，总量786336bit**

  **bit翻转率为：0.00242%**
  **SNR为：92.34**

  ```
  Epoch 1/5
  ----------
  train Loss: 0.1992 Acc: 0.9216
  val Loss: 0.2795 Acc: 0.9017
  Epoch 2/5
  ----------
  train Loss: 0.1279 Acc: 0.9538
  val Loss: 0.2761 Acc: 0.9043
  Epoch 3/5
  ----------
  train Loss: 0.1060 Acc: 0.9627
  val Loss: 0.3138 Acc: 0.8867
  Epoch 4/5
  ----------
  train Loss: 0.0944 Acc: 0.9674
  val Loss: 0.3607 Acc: 0.8961
  Epoch 5/5
  ----------
  train Loss: 0.0840 Acc: 0.9717
  val Loss: 0.3036 Acc: 0.9076
  Test Accuracy: 90.76%
  Test Loss: 0.3036
  Test total: 32768
  Test correct: 29741
  ```

  







## vit_h_14

使用的是lc_swag，不需要调整图片尺寸的参数

```
```



### 嵌入后

原始性能：Test Accuracy: 85.48%



### 重训练恢复结果



### 翻转结果：



