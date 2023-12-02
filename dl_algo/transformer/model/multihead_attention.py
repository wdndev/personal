# -*- coding: utf-8 -*-
#  @file        - multihead_attention.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 多头注意力
#  @version     - 0.0
#  @date        - 2023.06.06
#  @copyright   - Copyright (c) 2023

import math
import pandas as pd
import torch
from torch import nn 
from torch.autograd import Variable
import copy


from .utils import clones


def attention(query, key, value, mask=None, dropout=None):
    """注意力机制的实现, 
        输入分别是query, key, value, mask: 掩码张量, 
       dropout是nn.Dropout层的实例化对象, 默认为None
    """

    # 将query的最后一个维度提取出来，代表的是词嵌入的维度
    d_k = query.size(-1)

    # 按照注意力公式, 将query与key的转置相乘, 这里面key是将最后两个维度进行转置, 
    # 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算.
    # 得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 接着判断是否使用掩码张量
    if mask is not None:
        # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 
        # 如果掩码张量处为0，则对应的scores张量用-1e9这个值来替换, 如下演示
        scores = scores.masked_fill(mask == 0, -1e9)

    # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
    # 这样获得最终的注意力张量
    p_attn = scores.softmax(dim=-1)

    # 之后判断是否使用dropout进行随机置0
    if dropout is not None:
        # 将p_attn传入dropout对象中进行'丢弃'处理
        p_attn = dropout(p_attn)

    # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    """
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        """ 多头注意力机制
            - head代表头数
            - embedding_dim代表词嵌入的维度， 
           - dropout代表进行dropout操作时置0比率，默认是0.1
        """
        
        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        # 传入头数head
        self.head = head

         # 然后获得线性层对象，通过nn的Linear实例化，
         # 它的内部变换矩阵是embedding_dim x embedding_dim，
         # 然后使用clones函数克隆四个，为什么是四个呢，
         # 这是因为在多头注意力中，Q，K，V各需要一个，
         # 最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ 输入参数有四个，前三个就是注意力机制需要的Q, K, V，
            最后一个是注意力机制中可能需要的mask掩码张量，默认是None.
        """
        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(1)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，
        # 将输入QKV分别传到线性层中，做完线性变换后，
        # 开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，
        # 这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.
        # 这样我们就得到了每个头的输入.
        query, key, value = [
            model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))
        ]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，
        # 我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，
        # 先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        
        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)




from .positional_encoding import Embeddings, PositionalEncoding

def test_attention():
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

    query = key = value = pe_res
    attn, p_attn = attention(query, key, value)

    print("attn:", attn)
    print("attn shape:", attn.shape)
    print("p_attn:", p_attn)
    print("p_attn shape:", p_attn.shape)

def test_mult_head():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(embr)

    head = 8
    embedding_dim = 512
    dropout = 0.2

    query = key = value = pe_res

    mask = Variable(torch.zeros(8, 4, 4))

    mha = MultiHeadAttention(head, embedding_dim, dropout)
    mha_res = mha(query, key, value, mask)

    print(mha_res)
    print(mha_res.shape)

if __name__ == '__main__' :
    
    test_mult_head()



