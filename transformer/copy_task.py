import numpy as np
import pandas as pd
import torch

from transformer_utils import Batch


def data_generator(V, batch_size, num_batch):
    """ 该函数用于随机生成copy任务的数据
        - V: 随机生成数字的最大值+1
        - batch: 每次输送给模型更新一次参数的数据量
        - num_batch: 一共输送num_batch次完成一轮
    """
    for i in range(num_batch):
        # 在循环中使用randint方法随机生成[1, V)的整数, 
        # 分布在(batch, 10)形状的矩阵中, 
        data = torch.randint(1, V, size=(batch_size, 10))
        # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列, 
        # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
        data[:, 0] = 1
        # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
        # 因此requires_grad设置为False
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(src, tgt, 0)


