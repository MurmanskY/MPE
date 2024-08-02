为了实现模型的触发器，在torch库中的函数更改部分，如果需要迁移，则要重新复制代码

![image-20240803011543373](/Users/mac/Library/Application Support/typora-user-images/image-20240803011543373.png)

```shell
/opt/anaconda3/envs/MPE/lib/python3.12/site-packages/torchvision/models/resnet.py
```





# 修改resnet的\_forward\_impl部分

![image-20240803011543373](/Users/mac/Library/Application Support/typora-user-images/image-20240803011543373.png)

Q：pytorch中的resnet在最后一层经过全连接层后并没有使用softmax操作，此时logits还存在吗，是什么

A：在 PyTorch 中，ResNet 模型的实现通常不会在最后一层全连接层（fully connected layer）之后直接应用 softmax 操作。这是因为 softmax 操作通常会在计算损失时与损失函数（如 `CrossEntropyLoss`）一起应用，而不是在模型的前向传播过程中应用。因此，经过全连接层后的输出即为 logits。

![image-20240803011957616](/Users/mac/Library/Application Support/typora-user-images/image-20240803011957616.png)

对于resnet网络结构中，经过最后一个全连接层后输出的值就是logits值





# 修改VGG的forward部分

![image-20240803012338296](/Users/mac/Library/Application Support/typora-user-images/image-20240803012338296.png)

也是最后一个全连接层的输出值为logits值

![image-20240803012636495](/Users/mac/Library/Application Support/typora-user-images/image-20240803012636495.png)

