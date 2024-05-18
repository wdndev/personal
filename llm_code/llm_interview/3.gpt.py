import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6) -> None:
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x:torch.Tensor):
        input_type = x.dtype
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.sqrt(variance + self.eps)
        return x * self.weight

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, hidden_size)
        # 初始化一个绝对位置矩阵, 在我们这里，词汇的绝对位置就是用它的索引去表示. 
        # 所以我们首先使用arange方法获得一个连续自然数向量，然后再使用unsqueeze方法拓展向量维度使其成为矩阵， 
        # 又因为参数传的是1，代表矩阵拓展的位置，会使向量变成一个max_len x 1 的矩阵， 
        position = torch.arange(0, max_len).unsqueeze(1)

        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加入到位置编码矩阵中，
        # 最简单思路就是先将max_len x 1的绝对位置矩阵， 变换成max_len x d_model形状，然后覆盖原来的初始位置编码矩阵即可， 
        # 要做这种矩阵变换，就需要一个1xd_model形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        # 还希望它能够将自然数的绝对位置编码缩放成足够小的数字，有助于在之后的梯度下降过程中更快的收敛.  这样我们就可以开始初始化这个变换矩阵了.
        # 首先使用arange获得一个自然数矩阵， 但是细心的同学们会发现， 我们这里并没有按照预计的一样初始化一个1xd_model的矩阵， 
        # 而是有了一个跳跃，只初始化了一半即1xd_model/2 的矩阵。 为什么是一半呢，其实这里并不是真正意义上的初始化了一半的矩阵，
        # 我们可以把它看作是初始化了两次，而每次初始化的变换矩阵会做不同的处理，第一次初始化的变换矩阵分布在正弦波上， 第二次初始化的变换矩阵分布在余弦波上， 
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(torch.log(torch.tensor(10000.0)) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 这样我们就得到了位置编码矩阵pe, pe现在还只是一个二维矩阵，要想和embedding的输出（一个三维张量）相加，
        # 就必须拓展一个维度，所以这里使用unsqueeze拓展维度.
        pe = pe.unsqueeze(0)
        # 最后把pe位置编码矩阵注册成模型的buffer，什么是buffer呢，
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要随着优化步骤进行更新的增益对象. 
        # 注册之后我们就可以在模型保存后重加载时和模型结构与参数一同被加载.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x, 表示文本序列的词嵌入表示 """
        # 在相加之前我们对pe做一些适配工作， 将这个三维张量的第二维也就是句子最大长度的那一维将切片到与输入的x的第二维相同即x.size(1)，
        # 因为我们默认max_len为5000一般来讲实在太大了，很难有一条句子包含5000个词汇，所以要进行与输入张量的适配. 
        # 最后使用Variable进行封装，使其与x的样式相同，但是它是不需要进行梯度求解的，因此把requires_grad设置成false.
        x = x + self.pe[:, :x.size(1)]
        return x

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

    def forward(self, x, mask_attention=None):
        batch_size = x.size()[0]
        # 过线性层
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # 分割头
        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        # mask
        if mask_attention != None:
            attention_scores += mask_attention * -1e9
        # 计算注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 每个头注意力输出
        output = torch.matmul(attention_probs, v)
        # 还原形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        # 过输出层
        output = self.w_o(output)

        return output
    
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size=None):
        super(TransformerDecoder, self).__init__()
        self.attention = MHA(hidden_size, num_heads)
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
        self.ffn = FFN(hidden_size, intermediate_size)
        
        self.norm_att = RMSNorm(hidden_size)
        self.norm_ffn = RMSNorm(hidden_size)

    def forward(self, x):
        h = x + self.attention(self.norm_att(x))
        out = h + self.ffn(self.norm_ffn(h))
        return out
    
class GPT(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList (
            [TransformerDecoder(hidden_size, num_heads) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        h = self.embed_tokens(x)
        h = self.position_encoding(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)

        return logits


if __name__ == "__main__":
    x = torch.randint(0, 1000, (10, 20)) 
    print(x.size())

    gpt = GPT(vocab_size=1000, hidden_size=64, num_layers=2, num_heads=4)
    
    output = gpt(x)
    print(output.size())





