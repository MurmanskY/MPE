- imageNetRe

对原始参数进行修改后，在ImageNet1K数据集上进行重训练的模型代码



- pretrainedModels

项目中需要使用到的预训练模型源码



- modelStructure

模型结构



- embeddedModels
  对修改的参数进行性能测试
  - allParaEmbeddingWithLowBitFlip20240806
    所有参数低N比特翻转得倒的性能结果
  - embeddingWithLowBitFlip20240805
    错误，忽略
  - embeddingWithXOR20240804
    在最后一个全连接层进行low N bit翻转的实验结果
  - firstConvEmbeddingWithLowBitFlip20240806
    第一额卷积层进行low N bit 翻转的实验结果