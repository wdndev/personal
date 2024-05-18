import numpy as np
import matplotlib.pyplot as plt

def linear_model(x, w, b):
    """线性模型预测"""
    return w * x + b

def loss_function(y_true, y_pred):
    """均方误差损失函数"""
    return np.mean((y_true - y_pred)**2)

def gradient_descent(x, y_true, y_pred, w, b, learning_rate):
    """梯度下降更新权重"""
    m = len(x)
    grad_w = (1 / m) * np.sum((y_pred - y_true) * x)  # 注意这里对x的乘积
    grad_b = (1 / m) * np.sum(y_pred - y_true)
    
    # 确保梯度更新时形状一致，这里假设w和b都是标量或与x的行数兼容
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    
    return w, b


x = np.arange(0.0, 10.0, 0.05)
print(x)
# y = 20 * np.sin(2 * np.pi * x)
y = 2 * x + 3 + np.random.randn(len(x))
# 初始化权重和偏置
w = np.random.randn()
b = np.random.randn()

# 设置学习率和迭代次数
learning_rate = 0.05
epochs = 1000
losses = []  # 用于记录每一轮的损失

# 训练模型
for epoch in range(epochs):
    y_pred = linear_model(x, w, b)
    loss = loss_function(y, y_pred)
    losses.append(loss)
    
    # 每100次迭代打印一次损失
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
    
    # 更新权重
    w, b = gradient_descent(x, y, y_pred, w, b, learning_rate)

print(f"Final w: {w}, b: {b}")

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), losses)
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 绘制真实值与预测值对比图
plt.subplot(1, 2, 2)
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, linear_model(x, w, b), color='red', label='Predicted')
plt.title('Actual vs. Predicted Values')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()