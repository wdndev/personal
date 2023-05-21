# -*- coding: utf-8 -*-
#  @file        - se_block.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - ECA 通道注意力机制
#  @version     - 0.0
#  @date        - 2022.07.06
#  @copyright   - Copyright (c) 2021 

import math
import torch
import torch.nn as nn

from torchstat import stat  # 查看网络参数

class ECABlock(nn.Module):
    """ SE注意力机制类
    """
    def __init__(self, in_channel, gamma=2, b=1):
        """ 初始化
            - channel: 输入特征图的通道数
            - gamma: 公式中的两个系数
            - b: 公式中的两个系数
        """
        super(ECABlock, self).__init__()
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                              padding=(kernel_size-1)//2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """ 前向传播
        """
        # v = self.avg_pooling(X)
        # v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # # 归一化处理
        # v = self.sigmoid(v)

        # 获得输入图像的shape
        b, c, h, w = X.shape
        
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        v = self.avg_pooling(X)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        v = v.view([b,1,c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        v = self.conv(v)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        v = v.view([b,c,1,1])
         # 权值归一化
        v = self.sigmoid(v)

        return X * v

def test_eca_net():
    # 构造输入层 [b,c,h,w]==[4,32,16,16]
    inputs = torch.rand([4,32,16,16])
    # 获取输入图像的通道数
    in_channel = inputs.shape[1]
    # 模型实例化
    model = ECABlock(in_channel=in_channel)
    # 前向传播
    outputs = model(inputs)
    
    print(outputs.shape)  # 查看输出结果
    print(model)    # 查看网络结构
    stat(model, input_size=[32,16,16])  # 查看网络参数


if __name__ == "__main__":
    test_eca_net()
