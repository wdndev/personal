
from transformer_model import make_model
from transformer_utils import LabelSmoothing, SimpleLossCompute, greedy_decode, run_epoch, rate, DummyScheduler, DummyOptimizer

from copy_task import *
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt


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

def run(model, loss, epochs=10):
    """
    """
    for epoch in range(epochs):
        model.train()
        run_epoch()

def copy_task_train(epochs):
    """
    """
    V = 11
    batch_size = 80
    model = make_model(V, V, N=2)
    # 获得模型优化器
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    # 使用LabelSmoothing获得标签平滑对象
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    # 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法
    loss = SimpleLossCompute(model.generator, criterion)
    lr_scheduler = LambdaLR(optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=model.src_embed[0].d_model, 
                                    factor=1.0, warmup=400))

    for epoch in range(epochs):
         # 模型使用训练模式, 所有参数将被更新
        model.train()
        run_epoch(data_generator(V, batch_size, 20), model, loss,
            optimizer, lr_scheduler, mode="train")
        
        # 模型使用评估模式, 参数将不会变化 
        model.eval()
        run_epoch(data_generator(V, batch_size, 5), model, loss,
            DummyOptimizer(), DummyScheduler(), mode="eval")[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    result = greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(result)



if __name__ == '__main__' :

    copy_task_train(20)