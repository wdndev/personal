# -*- coding: utf-8 -*-
#  @file        - copy_task.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - COPY任务，用于测试模型
#  @version     - 0.0
#  @date        - 2023.06.08
#  @copyright   - Copyright (c) 2023

import numpy as np
import pandas as pd
import torch
import os

from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

from model.transformer import make_model
from model.utils import Batch, LabelSmoothing, SimpleLossCompute, greedy_decode, run_epoch, rate, DummyScheduler, DummyOptimizer


def data_generator(V, batch_size, num_batch, device="cpu"):
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
        src = data.requires_grad_(False).clone().detach().to(device)
        tgt = data.requires_grad_(False).clone().detach().to(device)
        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(src, tgt, 0)


def test_data_gen():
    V = 11
    batch_size = 20
    num_batch = 30
    res = data_generator(V, batch_size, num_batch)
    print(res)

def test_label_smoothing():
    # 使用LabelSmoothing实例化一个crit对象.
    # 第一个参数size代表目标数据的词汇总数, 也是模型最后一层得到张量的最后一维大小
    # 这里是5说明目标词汇总数是5个. 第二个参数padding_idx表示要将那些tensor中的数字
    # 替换成0, 一般padding_idx=0表示不进行替换. 第三个参数smoothing, 表示标签的平滑程度
    # 如原来标签的表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].
    crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

    # 假定一个任意的模型最后输出预测结果和真实结果
    predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                                [0, 0.2, 0.7, 0.1, 0], 
                                [0, 0.2, 0.7, 0.1, 0]]))

    # 标签的表示值是0，1，2
    target = Variable(torch.LongTensor([2, 1, 0]))

    # 将predict, target传入到对象中
    crit(predict, target)

    # 绘制标签平滑图像
    plt.imshow(crit.true_dist)
    plt.waitforbuttonpress()

def test_get_model_loss():
    V = 11
    batch_size = 20
    num_batch = 30
    model = make_model(V, V, N=2)
    # 获得模型优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion)


def copy_task_train(epochs, device="cpu"):
    """
    """
    V = 11
    batch_size = 128
    model = make_model(V, V, N=2, d_model=256, d_ff=512, h=4, dropout=0.2).to(device)
    # 获得模型优化器
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0).to(device)
    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion)
    lr_scheduler = LambdaLR(optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, 
                                    factor=1.0, warmup=400))

    for epoch in range(epochs):
         # 模型使用训练模式, 所有参数将被更新
        model.train()
        # data_generator(V, batch_size, 20).to("cuda")
        run_epoch(epoch, data_generator(V, batch_size, 20, device), model, loss,
            optimizer, lr_scheduler, mode="train")
        
        # 模型使用评估模式, 参数将不会变化 
        model.eval()
        run_epoch(epoch, data_generator(V, batch_size, 5, device), model, loss,
            DummyOptimizer(), DummyScheduler(), mode="eval")[0]
        file_path = "tmp_model/copt_task_model_%.2d.pt" % (epoch)
        torch.save(model.state_dict(), file_path)

    file_path = "tmp_model/copt_task_model_final.pt"
    torch.save(model.state_dict(), file_path)
    
    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    result = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(result)

def copy_task_eval(device):
    # device = "cpu"
    V = 11
    batch_size = 128

    model = make_model(V, V, N=2, d_model=256, d_ff=512, h=4, dropout=0.2)
    model.load_state_dict(torch.load("tmp_model/copt_task_model_final.pt"))
    model.to(device)

    # model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    result = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(result)


if __name__ == '__main__' :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # 
    model_path = "tmp_model/copt_task_model_final.pt"
    if not os.path.exists(model_path):
        copy_task_train(60, device)

    copy_task_eval(device)