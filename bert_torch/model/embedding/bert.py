import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """
    """
    bert的3种embedding分别有什么意义，如果实现的? 3种embedding分别是:
        1. token embedding: 输入文本在送入token embeddings层之前要先进行tokenization处理。
            此外，两个特殊的token会被插入到tokenization的结果的开头([CLS])和结尾(ISEP])。
            tokenization使用的方法是WordPiece tokenization.
        2. segment embedding: Bert在处理句子对的任务时，需要区分句子对中的上句/下句，
            因此用segment embedding来做识别。上句所有token的segment embedding均相同，
            下句所有token的segment embedding也相同;换句话说，
            如果是句子对,segment embedding只会有两种值。
            如果任务为单句的话，那segment embedding只有一个值。
        3. position embedding: 与transformer不同的是，Bert的position embedding是学习出来的。
            相同的是，positionembedding仍然用的是相对位置编码。

    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
