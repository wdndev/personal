import numpy as np
import torch

def np_sqrt2(lr=0.01, max_iters=1000, tolerance=1e-7):
    # 初始化x的值，这里选择1作为初始猜测值
    x = 1.0
    for _ in range(max_iters):
        fx = x**2 - 2
        loss = fx ** 2
        if abs(fx) < tolerance:
            break
        
        # 求解梯度
        d_loss_fx = 2 * fx
        d_fx_x = 2 * x
        d_loss_x = d_loss_fx * d_fx_x
        # 更新 x
        x = x - lr * d_loss_x
        
    return x

def torch_sqrt2(lr=0.01, max_iters=1000, tolerance=1e-7):
    x = torch.tensor(1.0, requires_grad=True)
    for _ in range(max_iters):
        fx = x**2 - 2
        loss = fx ** 2
        if abs(fx) < tolerance:
            break

        loss.backward()
        with torch.no_grad():
            x -= lr * x.grad
        # 手动清零梯度
        x.grad.zero_()

    return x.item()


# 使用梯度下降法求解根号2
np_sqrt2_v = np_sqrt2()
print(f"np sqrt(2): {np_sqrt2_v}")

# 使用梯度下降法求解根号2
torch_sqrt2_v = torch_sqrt2()
print(f"torch sqrt(2): {torch_sqrt2_v}")

# 对比Python内置库的精确值
exact_sqrt2 = np.sqrt(2)
print(f"Exact value of sqrt(2): {exact_sqrt2}")