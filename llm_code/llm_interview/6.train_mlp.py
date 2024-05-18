import torch
from torch import nn
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.w1(x)
        x = self.act(x)
        x = self.w2(x)
        return x

if __name__ == "__main__":
    model = MLP(1, 20, 1)
    print(model)

    # 1.准备数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    y = x.pow(3) + 0.1 * torch.randn(x.size())

    # 2.定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 3.训练模型
    losses = []
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 前向传播
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 每隔一定周期打印损失
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 4. 保存模型
    torch.save(model.state_dict(), 'mlp_model.pth')
    print("Model has been saved.")

    # 5. 预测并绘制结果
    with torch.no_grad():
        y_pred = model(x)
        plt.figure(figsize=(12, 6))
        plt.plot(x.detach().numpy(), y.numpy(), 'b-', label='Actual')
        plt.plot(x.detach().numpy(), y_pred.detach().numpy(), 'r-', label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted')
        plt.show()

    # 6. 绘制损失变化曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, num_epochs+1, 1), losses, 'g-', label='Training Loss')
    plt.legend()
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()



