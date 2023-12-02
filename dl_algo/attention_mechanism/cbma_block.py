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

class ChannelAttentionFC(nn.Module):
    """ 通道注意力机制——全连接层版本
    """
    def __init__(self, in_channel, ratio):
        """ 初始化
            - channel：输入特征图的通道数
            - ratio：全连接层下降通道倍数
        """
        super(ChannelAttentionFC, self).__init__()
        # 全局最大池化 [b,c,h,w]==>[b,c,1,1]
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = in_channel, out_features = in_channel // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = in_channel // ratio, out_features = in_channel, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """ 前向传播
        """
        b, c, h, w = X.shape
        # 输入图像做全局最大池化 [b,c,h,w]==>[b,c,1,1]==>[b,c]
        avg_x = self.avg_pooling(X).view(b, c)
        # 输入图像的全局平均池化 [b,c,h,w]==>[b,c,1,1]==>[b,c]
        max_x = self.max_pooling(X).view(b, c)
        # [b,c]==>[b,c//4]==>[b,c]
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        # sigmoid函数权值归一化, # 调整维度 [b,c]==>[b,c,1,1]
        v = self.sigmoid(v).view(b, c, 1, 1)
        # 输入特征图和通道权重相乘 [b,c,h,w]
        return X * v

class ChannelAttentionConv(nn.Module):
    """ 通道注意力机制——一维卷积版本
    """
    def __init__(self, in_channel, gamma = 2, b = 1):
        """ 初始化
            - channel: 输入特征图的通道数
            - gamma: 公式中的两个系数
            - b: 公式中的两个系数
        """
        super(ChannelAttentionConv, self).__init__()
        # 根据输入通道数自适应调整卷积核大小
        kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
        # 如果卷积核大小是奇数
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # 池化
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # 一维卷积
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, 
                              padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """ 前向传播
        """
        # 全局池化 [b,c,h,w]==>[b,c,1,1]
        avg_x = self.avg_pooling(X)
        max_x = self.max_pooling(X)
        # [b,c,1,1]==>[b,1,c] =1D卷积=> [b,1,c]==>[b,c,1,1]
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 权值归一化
        v = self.sigmoid(avg_out + max_out)
        # 输入特征图和通道权重相乘 [b,c,h,w]
        return X * v

class SpatialAttention(nn.Module):
    """ 空间注意力机制
    """
    def __init__(self, k: int = 7):
        """初始化
            - k: 卷积核大小,只能为3, 5, 7
        """
        super(SpatialAttention, self).__init__()
        # 池化
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), 
                              padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """ 前向传播
        """
        # compress the C channel to 1 and keep the dimensions
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        avg_x = self.avg_pooling(X, dim = 1, keepdim = True)
        # 在通道维度上平均池化 [b,1,h,w]
        max_x, _ = self.max_pooling(X, dim = 1, keepdim = True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        # 空间权重归一化
        v = self.sigmoid(v)
        # 输入特征图和空间权重相乘
        return X * v

class CBAMBlock(nn.Module):
    """ CBAM注意力机制
    """
    def __init__(self, channel_attention_mode: str = "Conv",
                 spatial_attention_kernel: int = 7, 
                 in_channel: int = None, 
                 ratio: int = 4,
                 gamma: int = 2, 
                 b: int = 1):
        """ 初始化
            - channel_attention_mode: 通道注意力机制模式，可选["FC", "Conv"]
            - spatial_attention_kernel: 空间注意力机制的卷积核大小
            - in_channel: 通道注意力机制的输入通道数
            - ratio: 通道注意力机制的全连接下降的通道数
            - gamma: 公式中的两个系数
            - b: 公式中的两个系数
        """
        super(CBAMBlock, self).__init__()
        if channel_attention_mode == "FC":
            assert in_channel != None \
                and ratio != None   \
                and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = ChannelAttentionFC(in_channel = in_channel, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert in_channel != None \
                and gamma != None \
                and b != None \
                and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = ChannelAttentionConv(in_channel = in_channel, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = SpatialAttention(k = spatial_attention_kernel)

    def forward(self, X):
        """ 前向传播
        """
        # 先将输入图像经过通道注意力机制
        X = self.channel_attention_block(X)
        # 然后经过空间注意力机制
        X = self.spatial_attention_block(X)
        return X

if __name__ == "__main__":
    # feature_maps = torch.randn((8, 54, 32, 32))
    # model = CBAMBlock("FC", 5, in_channel = 54, ratio = 9)
    # model(feature_maps)
    # model = CBAMBlock("Conv", 5, in_channel = 54, gamma = 2, b = 1)
    # model(feature_maps)

    # 构造输入层 [b,c,h,w]==[4,32,16,16]
    inputs = torch.rand([4,32,16,16])
    # 获取输入图像的通道数
    in_channel = inputs.shape[1]
    # 模型实例化
    model = CBAMBlock(in_channel=in_channel)
    # 前向传播
    outputs = model(inputs)
    
    print(outputs.shape)  # 查看输出结果
    print(model)    # 查看网络结构
    # stat(model, input_size=[32,16,16])  # 查看网络参数


