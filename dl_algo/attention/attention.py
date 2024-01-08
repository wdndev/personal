

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_mask(x, mask, mode='mul'):
    """ 通用mask函数
        这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        mask = mask.float()
        for _ in range(x.dim() - mask.dim()):
            mask = mask.unsqueeze(mask.dim())
        if mode == 'mul':
            return x * mask
        else:
            print("11: ", x)
            print("222: ", mask)
            return x - (1 - mask) * 1e10
        
def extract_seq_patches(x, kernel_size, rate):
    """ x.shape = [None, seq_len, seq_dim]
        滑动地把每个窗口的x取出来, 为做局部attention作准备。
    """
    # B, seq_len, seq_dim = x.shape
    seq_dim = x.size(-1)
    seq_len = x.size(1)
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    
    print(x.shape)
    # 在序列两端进行填充
    x = torch.nn.functional.pad(x, (p_right, p_left), mode='constant', value=0)
    print(x.shape)
    
    # 按照 rate 滑动地获取每个窗口的 x
    xs = [x[:, :, i: i + seq_len] for i in range(0, k_size, rate)]
    print(x.shape)
    
    # 在第三维上进行连接
    x = torch.cat(xs, 2)
    print(x.shape)
    
    # 重新塑形成期望的形状
    x = x.view(-1, seq_len, kernel_size, seq_dim)
    return x












def test_to_mask():
    # 创建一个输入张量 x 和一个 mask
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])

    # 测试 mask 为 None 的情况
    result_no_mask = to_mask(x, None)
    assert torch.all(torch.eq(result_no_mask, x))

    # 测试 mode='mul' 的情况
    result_mul = to_mask(x, mask, mode='mul')
    expected_result_mul = torch.tensor([[1, 0, 3], [0, 5, 6]])
    assert torch.all(torch.eq(result_mul, expected_result_mul))

    # 测试 mode='sub' 的情况
    result_sub = to_mask(x, mask, mode='sub').float()
    expected_result_sub = torch.tensor([[1, 2 - 1e10, 3], [4 - 1e10, 5, 6]])
    print(result_sub)
    assert torch.all(torch.eq(result_sub, expected_result_sub))

    print("所有测试用例都通过！")

def test_to_extract():
    # 创建一个测试输入张量
    batch_size = 5
    seq_len = 361
    seq_dim = 64
    kernel_size = 3
    rate = 2

    # 随机生成输入张量
    x = torch.randn(batch_size, seq_len, seq_dim)

    print(x.shape)

    # 执行函数
    result = extract_seq_patches(x, kernel_size, rate)

    # 打印结果形状
    print(result.shape)

if __name__ == "__main__":
    # 随机初始化一个 2x4x19x19 的张量
    tensor = torch.randn(5, 5, 19, 19)
    test_to_mask()
