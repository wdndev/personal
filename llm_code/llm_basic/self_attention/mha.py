# -*- coding: utf-8 -*-
#  @file        - hma.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 多头自注意力
#  @version     - 0.0
#  @date        - 2024.03.03
#  @copyright   - Copyright (c) 2024

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """ 多头注意力机制
        - head 代表头数
        - embedding_dim 代表词嵌入的维度， 
        - dropout 代表进行dropout操作时置0比率，默认是0.1
    """
    def __init__(self, head, embedding_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        self.wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.att_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print("mha in: ", x.shape)
        batch_size, seq_len, hidden_dim = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # (b, ql, hd) -> (b, ql, nh, dk)
        q = q.view(batch_size, seq_len, self.head, self.d_k)
        k = k.view(batch_size, seq_len, self.head, self.d_k)
        v = v.view(batch_size, seq_len, self.head, self.d_k)

        # (b, ql, nh, dk) -> (b, nh, ql, dk)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # (b, nh, ql, dk) @ (b, nh, dk, ql) => (b, nh, ql, ql)
        att = torch.matmul(q, k.transpose(2, 3))
        att /= math.sqrt(self.d_k)
        score = F.softmax(att.float(), dim=-1)
        score = self.att_dropout(score)
        
        # (b, nh, ql, ql) @ (b, nh, ql, dk) => (b, nh, ql, dk)
        attv = torch.matmul(score, v)
        # (b, nh, ql, dk) -> (b, ql, nh*dk) = (b, ql, hd)
        attv = attv.view(batch_size, seq_len, -1)
        # print("mha out: ", attv.shape)
        return score, attv



