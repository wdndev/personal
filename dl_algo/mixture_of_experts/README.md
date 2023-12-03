# MoE 实现

# 1.Description

MoE 算法pytorch实现

代码来源：[mixture-of-experts](https://github.com/davidmrau/mixture-of-experts)

MoE 架构解析可参考如下文章：

- [Mixture of Experts-Introduction](https://abdulkaderhelwan.medium.com/mixture-of-experts-introduction-39f244a4ff05)
- [Understanding the Mixture-of-Experts Model in Deep Learning](https://medium.com/@jain.sm/understanding-the-mixture-of-experts-model-in-deep-learning-71d2e20650ac)
- 

# 2.Environment

- Python 3
- Pytorch

# 3.Directory

```shell
  ├── data: 数据集存放文件夹
  ├── model: 存放模型
  ├───── weights: 训练生成模型存放文件夹
  ├── cifar10_moe.py: cifar10_moe demo文件
  ├── example.py: moe简单测试
  └── moe.py：MoE模型实现
```

# 4.Running

## 4.1 数据集

cifar10数据集下载，运行程序自动下载

## 4.2 cifar10_moe

训练10个epoch，准确率达到 40%

```shell
Files already downloaded and verified
Files already downloaded and verified
[1,  3100] loss: 2.215
[2,  3100] loss: 2.150
[3,  3100] loss: 2.107
[4,  3100] loss: 2.092
[5,  3100] loss: 2.074
[6,  3100] loss: 2.048
[7,  3100] loss: 2.052
[8,  3100] loss: 2.048
[9,  3100] loss: 2.046
[10,  3100] loss: 2.037
Finished Training
Accuracy of the network on the 10000 test images: 40 %
```

训练50个epoch，准确率达到 43%

``` shell
[10,  3100] loss: 2.037
[20,  3100] loss: 2.006
[30,  3100] loss: 1.986
[40,  3100] loss: 1.966
[50,  3100] loss: 1.944
Finished Training
Accuracy of the network on the 10000 test images: 43 %
```
