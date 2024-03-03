from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from mha import MultiHeadAttention


class Model(nn.Module):
    
    def __init__(self, nhead, vocab_size, embedding_dim, num_labels, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.attn = MultiHeadAttention(nhead, embedding_dim, dropout)
        self.fc = nn.Linear(embedding_dim, num_labels)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        h = self.emb(x)
        attn_score, h = self.attn(h)
        h = F.avg_pool1d(h.permute(0, 2, 1), seq_len, 1)
        h = h.squeeze(-1)
        logits = self.fc(h)

        return attn_score, logits

if __name__ == '__main__':
    # 参数设置
    vocab_size: int = 5000
    embedding_dim: int = 512
    num_heads: int = 16
    dropout: float = 0.1

    num_labels: int = 2

    max_seq_len: int = 512

    num_epochs: int = 10

    model = Model(num_heads, vocab_size, embedding_dim, num_labels, dropout)

    x = torch.randint(0, 5000, (3, 30))

    attn, logits = model(x)
    print(attn.shape, logits.shape)