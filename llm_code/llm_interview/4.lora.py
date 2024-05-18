import torch
from torch import nn

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super(LoRALayer, self).__init__()

        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x 
    
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank=4, alpha=1):
        super().__init__()

        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


if __name__ == "__main__":
    linear = nn.Linear(512, 512)
    linear_with_lora = LinearWithLoRA(linear, rank=8, alpha=0.9)
    x = torch.randn(32, 512)
    linear_out = linear(x)
    lora_out = linear_with_lora(x)
    print("linear_out: ", linear_out )
    print("linear_out: ", linear_out)

