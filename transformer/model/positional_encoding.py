# -*- coding: utf-8 -*-
#  @file        - positional_encoding.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 位置编码
#  @version     - 0.0
#  @date        - 2023.06.06
#  @copyright   - Copyright (c) 2023

import torch
from torch import nn 
import math
import numpy as np
from torch.autograd import Variable

class Embeddings(nn.Module):
    """ 实现文本嵌入层
        无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转表为向量表示，
        希望在这样的高维空间捕捉词汇间的关系。
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        """ 初始化
            - d_model：词嵌入的维度
            - vocab：词表的大小
        """

        # 获得词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, X):
        """ 前向传播
        """
        return self.lut(X) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """ 位置编码类
        Transformer结构中，没有针对词汇位置信息的处理，因此需要在Embeding层后加入位置编码，
        将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失。
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        """ 初始化
            - d_model: 词嵌入维度
            - dropout: 置0比例
            - max_len: 每个句子的最大长度
        """

        # 实例化nn中预定的Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，大小为max_len*d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，在这里，词汇的绝对位置就是用它的索引取表示。
        # 首先使用arange方法获得一个连续自然数向量，然后再使用unsqeeze方法拓展向量维度
        # 参数为1，代表矩阵拓展的位置，会使向量变成一个max_len * 1的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 考虑如何将这些位置信息加入到位置编码矩阵中
        # 最简单的思路就是先将max_lemn*1的绝对位置矩阵，
        # 变换成max_len*d_model形状，然后覆盖原来的初始位置编码即可。
        # 变换矩阵 1*d_model -> max_len*d_model，
        # 还希望将自然数的绝对位置编码缩放成足够小的数，有助于后续梯度下降中的收敛。
        # torch.arange(0, d_model, 2), 初始化了一半，可以看作使初始化了两次，
        # 每次初始化做不同的处理，填充再位置编码矩阵的偶数和奇数位置上
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             - (math.log(10000.0) / d_model))
        # 将前面定义变换矩阵div_term，对奇数和偶数不同赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe为二维矩阵，而embedding的输出为三维，扩展矩阵
        pe = pe.unsqueeze(0)

        # 把pe位置编码矩阵注册成模型的buffer
        # buffer，对模型有效果，但不是超参，不需要随优化器调整，注册之后可以随模型加载和保存
        self.register_buffer("pe", pe)

    def forward(self, X):
        """ 前向传播
            -X: 文本序列的词嵌入表示
        """
        # self.pe[:, :X.size(1) 截取部分pe，与输入匹配
        # 明确pe的编码太长了，将第二个维度，
        # 也就是max_len对应的那个维度缩小成x的句子长度同等的长度
        # X = X + Variable(self.pe[:, :X.size(1)], requires_grad=False)
        X = X +  self.pe[:, : X.size(1)].requires_grad_(False)

        return self.dropout(X)


import matplotlib.pyplot as plt
def show_positional():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    # 向pe传入被Vaiiable封装的tensor，这样pe会执行forward函数
    # 且这个tensor里的数值都是0，被处理后相当于位置编码张量
    y = pe.forward(torch.zeros(1, 100, 20))

    # 横坐标到100的长度，纵坐标是某一个词汇中的某维特征在不同长度下对应的值
    # 因为总共有20维之多，我们这里只查看4，5，6，7维的值
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # 画布上填写维度提示信息
    plt.legend(["dim %d" % p for p in [4,5,6,7]])

    # 输出效果分析:
    ## 每条颜色的曲线代表某一个词汇中的特征在不同位置的含义.
    ## 保证同一词汇随着所在位置不同它对应位置嵌入向量会发生变化.
    ## 正弦波和余弦波的值域范围都是1到-1这又很好的控制了嵌入数值的大小,有助于梯度的快速计算.

    plt.show()


def main():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)
    print("embr: ", embr)
    print(embr.shape)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(embr)
    print("pe_res: ", pe_res)
    print(pe_res.shape)

    show_positional()
    
    print("111")

if __name__ == '__main__' :
    main()






