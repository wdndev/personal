# -*- coding: utf-8 -*-
#  @file        - se_block.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - SE 通道注意力机制
#  @version     - 0.0
#  @date        - 2022.07.06
#  @copyright   - Copyright (c) 2021 

import torch
import torch.nn as nn

from torchstat import stat  # 查看网络参数

class SEBlock(nn.Module):
    """ SE注意力机制类
    """
    def __init__(self, in_channel, ratio=4, mode="avg"):
        """ 初始化
            - model：池化方法
            - channel：输入特征图的通道数
            - ratio：全连接层下降通道倍数
        """
        super(SEBlock, self).__init__()

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_polling = nn.AdaptiveMaxPool2d(1)

        if mode == "max":
            self.global_pooling = self.max_polling
        elif mode == "avg":
            self.global_pooling = self.avg_pooling

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """ 前向传播
        """
        b, c, h, w = X.shape
        # 全局池化 + 维度调整：[b,c,h,w]==>[b,c,1,1] ==> [b,c]
        v = self.global_pooling(X).view(b, c)
        # MLP + 维度调整
        v = self.fc_layers(v).view(b, c, 1, 1)
        # 归一化处理
        v = self.sigmoid(v)

        return X * v

def test_se_net():
    # 构造输入层shape==[4,32,16,16]
    inputs = torch.rand(4,32,16,16)
    # 获取输入通道数
    in_channel = inputs.shape[1]
    # 模型实例化
    model = SEBlock(channels=in_channel)
    
    # 前向传播查看输出结果
    outputs = model(inputs)
    print(outputs.shape)  # [4,32,16,16])
    
    print(model) # 查看模型结构
    stat(model, input_size=[32,16,16])  # 查看参数，不需要指定batch维度

if __name__ == "__main__":
    test_se_net()
