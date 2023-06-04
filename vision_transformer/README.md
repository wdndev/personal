# ViT 实现

# 1.Description

Vision Transformer算法pytorch实现

代码来源：https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

Transformer架构解析可参考如下文章：

[Transformer架构解析](https://wdndev.github.io/2023/05/25/Transformer/Transformer%E6%9E%B6%E6%9E%84%E8%A7%A3%E6%9E%90)

# 2.Environment

- Python 3
- Pytorch

# 3.Directory

```shell
  ├── data: 分类数据集存放文件夹
  ├── model: 存放模型
  ├───── pre_model: 预训练模型存放文件夹
  ├───── weights: 训练生成模型存放文件夹
  ├── flops.py: 
  ├── my_dataset.py: 数据类封装
  ├── predict.py: 预测
  ├── train.py: 训练
  ├── utils.py: 共有文件
  └── vit_model.py：visvision transformer模型实现
```
# 4.Running

数据集及预训练权重下载：链接：https://pan.baidu.com/s/1td_WlB9ow3JVHvsvjEAZ7Q  提取码：o4r3

## 4.1 需要修改源码地方

1. 在train.py脚本中将--data-path设置成解压后的flower_photos文件夹绝对路径
2. 下载预训练权重，在vit_model.py文件中每个模型都有提供预训练权重的下载地址，根据自己使用的模型下载对应预训练权重,如果速度过慢，使用上述百度网盘链接
3. 在train.py脚本中将--weights参数设成下载好的预训练权重路径
4. 设置好数据集的路径--data-path以及预训练权重的路径--weights就能使用train.py脚本开始训练了(训练过程中会自动生成class_indices.json文件)
5. 在predict.py脚本中导入和训练脚本中同样的模型，并将model_weight_path设置成训练好的模型权重路径(默认保存在weights文件夹下)
6. 在predict.py脚本中将img_path设置成你自己需要预测的图片绝对路径
7. 设置好权重路径model_weight_path和预测的图片路径img_path就能使用predict.py脚本进行预测了
8. 如果要使用自己的数据集，请按照花分类数据集的文件结构进行摆放(即一个类别对应一个文件夹)，并且将训练以及预测脚本中的num_classes设置成你自己数据的类别数


