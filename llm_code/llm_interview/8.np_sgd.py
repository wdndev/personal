import numpy as np

def random_gradient_descent(X, y, learning_rate=0.01, n_iters=1000):
    """
    使用随机梯度下降法实现线性回归
    :param X: 输入特征，形状为 (n_samples, n_features)
    :param y: 目标变量，形状为 (n_samples,)
    :param learning_rate: 学习率
    :param n_iters: 迭代次数
    :return: 权重向量 (w)，截距 b
    """
    n_samples, n_features = X.shape
    
    # 初始化权重向量和截距
    w = np.zeros(n_features)
    b = 0
    
    for _ in range(n_iters):
        for idx in np.random.permutation(n_samples):  # 随机选取样本的索引以实现随机梯度下降
            # 预测值
            y_pred = np.dot(X[idx], w) + b
            
            # 计算梯度
            dw = (y_pred - y[idx]) * X[idx]
            db = y_pred - y[idx]
            
            # 更新权重
            w -= learning_rate * dw
            b -= learning_rate * db
    
    return w, b

# 示例数据生成
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # 生成100个样本，1个特征
y = 2 * X.squeeze() + 3 + np.random.randn(100) * 1  # 线性关系加上噪声

# 拟合模型
weights, bias = random_gradient_descent(X, y)

# 打印结果
print(f'Weights: {weights}, Bias: {bias}')

# 可选：绘制拟合直线与数据点
import matplotlib.pyplot as plt
plt.scatter(X.squeeze(), y, color='blue', label='Data points')
plt.plot(X.squeeze(), weights*X.squeeze() + bias, color='red', label='Linear Regression Line')
plt.legend()
plt.show()