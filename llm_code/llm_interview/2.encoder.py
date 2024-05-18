import torch
from torch import nn

class MHA(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MHA, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.w_o = nn.Linear(hidden_size, hidden_size)

    def split_head(self, x):
        batch_size = x.size()[0]

        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        # 过线性层
        q = self.w_q(hidden_state)
        k = self.w_k(hidden_state)
        v = self.w_v(hidden_state)
        # 分割头
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.hidden_size))
        # mask
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        # 计算注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 每个头注意力输出
        output = torch.matmul(attention_probs, v)
        # 还原形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
        # 过输出层
        output = self.w_o(output)

        return output
    
class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size) -> None:
        super(FFN, self).__init__()

        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.w1(x)
        x = self.act_fn(x)
        x = self.w2(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size=None):
        super(TransformerBlock, self).__init__()

        self.attention = MHA(hidden_size, num_heads)
        self.norm_att = nn.LayerNorm(hidden_size)
        self.norm_ffn = nn.LayerNorm(hidden_size)

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        self.ffn = FFN(hidden_size, intermediate_size)

    def forward(self, x):
        att = self.attention(x)
        x = self.norm_att(att) + x

        ffn = self.ffn(x)
        x = self.norm_ffn(ffn) + x

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == "__main__":
    x = torch.rand(2, 10, 64)
    # print(x.size())


    ############ MHA ###########
    mha = MHA(hidden_size=64, num_heads=4)
    output = mha(x)
    print(output.size())

    encoder = TransformerEncoder(hidden_size=64, num_layers=2, num_heads=4)
    output = encoder(x)
    print(output.size())