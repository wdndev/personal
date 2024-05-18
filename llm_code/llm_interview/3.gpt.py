import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size) -> None:
        super(FFN, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = nn.ReLU()

    def forward(self, x):
        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_proj = self.down_proj(intermediate)
        return down_proj