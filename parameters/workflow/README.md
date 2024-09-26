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

70.24%，使用编码加冗余

9inter 11 corr



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
  train ss: 0.5759 Acc: 0.8288
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








## ResNet50

`layer4.2.conv2.weight`
Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

`layer4.1.conv2.weight`
(conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

9inter 11corr

### 重训练结果：

原始性能：80.12%

嵌入后性能：76.58%

使用ImageNet重训练后的性能：















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



### 不需要进行重训练



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



### 不需要重训练：



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

  





## ConvNeXt_large 嵌入36层

使用convnext_large

嵌入总容量为：284370B，

```
layers = ["features.5.12.block.3.weight",
              "features.5.12.block.5.weight",
              "features.5.13.block.3.weight",
              "features.5.13.block.5.weight",
              "features.5.14.block.3.weight",
              "features.5.14.block.5.weight",
              "features.5.15.block.3.weight",
              "features.5.15.block.5.weight",
              "features.5.16.block.3.weight",
              "features.5.16.block.5.weight",
              "features.5.17.block.3.weight",
              "features.5.17.block.5.weight",
              "features.5.18.block.3.weight",
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
```

```
Byte Num: 5266
File './malware/convnext_large_l1' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l2' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l3' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l4' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l5' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l6' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l7' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l8' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l9' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l10' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l11' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l12' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l13' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l14' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l15' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l16' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l17' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l18' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l19' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l20' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l21' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l22' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l23' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l24' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l25' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l26' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l27' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l28' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l29' generated with 42128 bits.
Byte Num: 5266
File './malware/convnext_large_l30' generated with 42128 bits.
Byte Num: 21065
File './malware/convnext_large_l31' generated with 168520 bits.
Byte Num: 21065
File './malware/convnext_large_l32' generated with 168520 bits.
Byte Num: 21065
File './malware/convnext_large_l33' generated with 168520 bits.
Byte Num: 21065
File './malware/convnext_large_l34' generated with 168520 bits.
Byte Num: 21065
File './malware/convnext_large_l35' generated with 168520 bits.
Byte Num: 21065
File './malware/convnext_large_l36' generated with 168520 bits.
```

### 嵌入后的性能：

原始性能：84.17%

嵌入后性能：82.29%



### 不需要进行重训练

### 翻转结果：

- FGCVAircraft：2274960Bit
  ```
  epoch = 5: total Flip Num is: 3, SNR = 117.60
  epoch = 10: total Flip Num is: 17, SNR = 102.53
  epoch = 15: total Flip Num is: 33, SNR = 96.77
  epoch = 20: total Flip Num is: 35, SNR = 96.26
  epoch = 25: total Flip Num is: 52, SNR = 92.82
  epoch = 30: total Flip Num is: 54, SNR = 92.49
  epoch = 35: total Flip Num is: 53, SNR = 92.66
  epoch = 40: total Flip Num is: 58, SNR = 91.87
  ```

  





## ConvNeXt_large 嵌入40层

6inter, 7 correct

400KB

### 嵌入后性能：

原始性能：84.17%

嵌入后性能：80.79%







## ConvNeXt_large嵌入40层

4inter，7correct

42130*6 + 10532 * 34 = 589804 Byte = 4718432 bit = 590KB

### 嵌入后性能：

原始性能：84.17%

嵌入后性能：79.18%

### 重训练结果：

在ImageNet上重训练一个epoch之后的性能：82.88%

 

### 翻转结果：

- CIFAR100

```
epoch=5, total Flip Num is:  8  SNR is:  115.414154271471804
epoch=10, total Flip Num is:  21  SNR is:  107.031568116632291
epoch=15, total Flip Num is:  34  SNR is:  102.846375670465574
epoch=20, total Flip Num is:  65  SNR is:  97.217686878453565
epoch=25, total Flip Num is:  59  SNR is:  98.058913778467792
epoch=30, total Flip Num is:  75  SNR is:  95.974728743476675
epoch=35, total Flip Num is:  72  SNR is:  96.329304082685307
epoch=40, total Flip Num is:  74  SNR is:  96.0913196166911527
```

- FGCVAircraft

```
epoch=5, total Flip Num is:  4  SNR is:  121.73954742206094
epoch=10, total Flip Num is:  14  SNR is:  110.85818653505542
epoch=15, total Flip Num is:  32  SNR is:  103.67774768222206
epoch=20, total Flip Num is:  39  SNR is:  101.9594551080902
epoch=25, total Flip Num is:  48  SNR is:  100.15592250110844
epoch=30, total Flip Num is:  61  SNR is:  98.07415054840484
epoch=35, total Flip Num is:  73  SNR is:  96.51429004621107
epoch=40, total Flip Num is:  77  SNR is:  96.05093274517054
```

- GTSRB

```
epoch=5, total Flip Num is:  29  SNR is:  104.53278729064107
epoch=10, total Flip Num is:  40  SNR is:  101.73954742206092
epoch=15, total Flip Num is:  41  SNR is:  101.52507011422546
epoch=20, total Flip Num is:  43  SNR is:  101.11137813702845
epoch=25, total Flip Num is:  42  SNR is:  101.31576144066217
epoch=30, total Flip Num is:  47  SNR is:  100.33879008990583
epoch=35, total Flip Num is:  46  SNR is:  100.5255906149887
epoch=40, total Flip Num is:  46  SNR is:  100.5255906149887
```

- PCAM

```
epoch=2, total Flip Num is:  19  SNR is:  108.20567522956361
epoch=4, total Flip Num is:  15  SNR is:  110.25892206750656
epoch=6, total Flip Num is:  30  SNR is:  104.23832215422694
epoch=8, total Flip Num is:  48  SNR is:  100.15592250110844
epoch=10, total Flip Num is:  61  SNR is:  98.07415054840484
epoch=12, total Flip Num is:  56  SNR is:  98.81698670849617
epoch=14, total Flip Num is:  76  SNR is:  96.16447540300436
epoch=16, total Flip Num is:  74  SNR is:  96.39611285400065
epoch=18, total Flip Num is:  89  SNR is:  94.79294711572194
epoch=20, total Flip Num is:  99  SNR is:  93.86804335666918
```













## vit_h_14

使用的是lc_swag，不需要调整图片尺寸的参数

```
```



### 嵌入后

原始性能：Test Accuracy: 85.48%





### 重训练恢复结果



### 翻转结果：



