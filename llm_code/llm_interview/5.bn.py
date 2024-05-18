import torch
from torch import nn

def batch_norm(x, epsilon=1e-6):
    mean = x.mean(dim=0)
    var = x.var(dim=0)
    x_hat = (x - mean) / torch.sqrt(var + epsilon)

    return x_hat

class CustomLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(CustomLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta
    
if __name__ == "__main__":
    # N,L,D
    x = torch.randn(5, 10, 64)
    ln = CustomLayerNorm(64)
    ln_x = ln(x)
    print(ln_x)

    ln_torch = nn.LayerNorm(normalized_shape=[64], eps=1e-6)
    normalized_x_torch = ln_torch(x)
    print(normalized_x_torch)

