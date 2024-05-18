import torch
from torch import nn

class MHA(nn.Module):
    """ 多头注意力
    """
    def __init__(self, hidden_size, num_heads) -> None:
        super(MHA, self).__init__()

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
        attention_scores = torch.matmul(q, k.transpose(-1 ,-2)) / torch.sqrt(torch.tensor(self.head_dim))
        # mask
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        # 计算注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 每个头注意力输出
        output = torch.matmul(attention_probs, v)
        # 还原形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        # 过输出线性层
        output = self.w_o(output)

        return output
    
class MQA(nn.Module):
    """ 多查询注意力
    """
    def __init__(self, hidden_size, num_heads) -> None:
        super(MQA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.w_q = nn.Linear(hidden_size, hidden_size)
        # w_k, w_v 和 MHA不一样
        self.w_k = nn.Linear(hidden_size, self.head_dim)
        self.w_v = nn.Linear(hidden_size, self.head_dim)

        self.w_o = nn.Linear(hidden_size, hidden_size)

    def split_head(self, x, head_num=None):
        batch_size = x.size()[0]
        if head_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            return x.view(batch_size, -1, head_num, self.head_dim).transpose(1, 2)
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        # 过线性层
        q = self.w_q(hidden_state)
        k = self.w_k(hidden_state)
        v = self.w_v(hidden_state)
        # 分割头，
        # 注意：不一样了，k,v 只需要一个
        q = self.split_head(q)
        k = self.split_head(k, 1)
        v = self.split_head(v, 1)
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        # mask
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        # 计算注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 每个头的输出
        output = torch.matmul(attention_probs, v)
        # 还原形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        # 过输出层
        output = self.w_o(output)

        return output

class GQA(nn.Module):
    """ 分组查询注意力
    """
    def __init__(self, hidden_size, num_heads, group_num) -> None:
        super(GQA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.group_num = group_num

        self.w_q = nn.Linear(hidden_size, hidden_size)
        # w_k, w_v 和 MHA不一样
        self.w_k = nn.Linear(hidden_size, self.group_num * self.head_dim)
        self.w_v = nn.Linear(hidden_size, self.group_num * self.head_dim)

        self.w_o = nn.Linear(hidden_size, hidden_size)

    def split_head(self, x, group_num=None):
        batch_size, seq_len, _ = x.size()
        if group_num == None:
            return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # torch.Size([2, 10, 32])
            x = x.view(batch_size, -1, group_num, self.head_dim).transpose(1, 2)
            # torch.Size([2, 2, 10, 16])
            x = x[:, :, None, :, :].expand(batch_size, group_num, self.num_heads//group_num, seq_len, self.head_dim) \
                .reshape(batch_size, self.num_heads//group_num * group_num, seq_len, self.head_dim)
            # torch.Size([2, 4, 10, 16])
            return x
        
    def forward(self, hidden_state, attention_mask=None):
        batch_size = hidden_state.size()[0]
        # 过线性层
        q = self.w_q(hidden_state)
        k = self.w_k(hidden_state)
        v = self.w_v(hidden_state)
        # 分割头，
        # 注意：不一样了，k,v 只需要一个
        q = self.split_head(q)
        k = self.split_head(k, self.group_num)
        v = self.split_head(v, self.group_num)
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        # mask
        if attention_mask != None:
            attention_scores += attention_mask * -1e-9
        # 计算注意力权重
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # 每个头的输出
        output = torch.matmul(attention_probs, v)
        # 还原形状
        output = output.transpose(-1, -2).contiguous().view(batch_size, -1, self.head_dim*self.num_heads)
        # 过输出层
        output = self.w_o(output)

        return output


if __name__ == "__main__":
    x = torch.rand(2, 10, 64)
    # print(x.size())


    ############ MHA ###########
    mha = MHA(hidden_size=64, num_heads=4)
    output = mha(x)
    print(output.size())

    ############ MQA ###########
    mqa = MQA(hidden_size=64, num_heads=4)
    output = mqa(x)
    print(output.size())

    ############ GQA ###########
    gqa = GQA(hidden_size=64, num_heads=4, group_num=2)
    output = gqa(x)
    print(output.size())
    

