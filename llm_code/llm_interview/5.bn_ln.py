import torch
from torch import nn
import numpy as np

class BN:
    def __init__(self, momentum=0.01, eps=1e-5, feat_dim=2):
        """
        初始化参数值
        :param momentum: 动量，用于计算每个batch均值和方差的滑动均值
        :param eps: 防止分母为0
        :param feat_dim: 特征维度
        """
        # 均值和方差的滑动均值
        self._running_mean = np.zeros(shape=(feat_dim, ))
        self._running_var = np.ones(shape=(feat_dim, ))
        # 更新self._running_xxx时的动量
        self._momentum = momentum
        # 防止分母计算为0
        self._eps = eps
        # 对应Batch Norm中需要更新的beta和gamma，采用pytorch文档中的初始化值
        self._beta = np.zeros(shape=(feat_dim, ))
        self._gamma = np.ones(shape=(feat_dim, ))

    def batch_norm(self, x):
        if self.training:
            x_mean = x.mean(axis=0)
            x_var = x.var(axis=0)
            # 对应running_mean的更新公式
            self._running_mean = (1-self._momentum)*x_mean + self._momentum*self._running_mean
            self._running_var = (1-self._momentum)*x_var + self._momentum*self._running_var
            # 对应论文中计算BN的公式
            x_hat = (x-x_mean)/np.sqrt(x_var+self._eps)
        else:
            x_hat = (x-self._running_mean)/np.sqrt(self._running_var+self._eps)
        return self._gamma*x_hat + self._beta

class LN(nn.Module):
    # 初始化
    def __init__(self, normalized_shape,  # 在哪个维度上做LN
                 eps:float = 1e-5, # 防止分母为0
                 elementwise_affine:bool = True):  # 是否使用可学习的缩放因子和偏移因子
        super(LN, self).__init__()
        # 需要对哪个维度的特征做LN, torch.size查看维度
        self.normalized_shape = normalized_shape  # [c,w*h]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 构造可训练的缩放因子和偏置
        if self.elementwise_affine:  
            self.gain = nn.Parameter(torch.ones(normalized_shape))  # [c,w*h]
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # [c,w*h]
 

    def forward(self, x: torch.Tensor): # [b,c,w*h]
        # 需要做LN的维度和输入特征图对应维度的shape相同
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]  # [-2:]
        # 需要做LN的维度索引
        dims = [-(i+1) for i in range(len(self.normalized_shape))]  # [b,c,w*h]维度上取[-1,-2]维度，即[c,w*h]
        # 计算特征图对应维度的均值和方差
        mean = x.mean(dim=dims, keepdims=True)  # [b,1,1]
        mean_x2 = (x**2).mean(dim=dims, keepdims=True)  # [b,1,1]
        var = mean_x2 - mean**2  # [b,c,1,1]
        x_norm = (x-mean) / torch.sqrt(var+self.eps)  # [b,c,w*h]
        # 线性变换
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias  # [b,c,w*h]
        return x_norm
 
# ------------------------------- #
# 验证
# ------------------------------- #
 
if __name__ == '__main__':
 
    x = torch.linspace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2,3,2*2])  # [b,c,w*h]
    # 实例化
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln(x)
    print(x.shape)
    


