# -*- coding: utf-8 -*-

import torch
from torch import nn
from transformer import EncoderBlock

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """ 获取时间序列的词元及其片段索引
    """
    """  BERT输入序列明确地表示单个文本和文本对。
         当输入为单个文本时，BERT输入序列是 特殊类别词元“<cls>”、文本序列的标记、以及特殊分隔词元“<sep>”的连结。
         当输入为文本对时，BERT输入序列是“<cls>”、第一个文本序列的标记、“<sep>”、第二个文本序列标记、以及“<sep>”的连结。
         我们将始终如一地将术语“BERT输入序列”与其他类型的“序列”区分开来。
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 和 1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)

    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments

class BERTEncoder(nn.Module):
    """ BERT编码器
    """

    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, 
                ffn_num_hiddens, num_heads, num_layers, dropout,
                max_len=1000, key_size=768, query_size=768, value_size=768,
                **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)

        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
            # 在BERT中，位置嵌入是可学习的，因此，创建一个足够长的位置嵌入参数
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))


    def forward(self, tokens, segments, valid_lens):
        """ 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        """
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)

        return X
    
class MaskLM(nn.Module):
    """ BERT的掩蔽语言模型任务
    """
    """
        掩蔽语言模型：为了双向编码上下文以表示每个词元，BERT随机掩蔽词元并使用来自双向上下文的词元以自监督的方式预测掩蔽词元。

        在这个预训练任务中，将随机选择15%的词元作为预测的掩蔽词元。要预测一个掩蔽词元而不使用标签作弊，
        一个简单的方法是总是用一个特殊的“<mask>”替换输入序列中的词元。然而，人造特殊词元“<mask>”不会出现在微调中。
        为了避免预训练和微调之间的这种不匹配，如果为预测而屏蔽词元（例如，在“this movie is great”中选择掩蔽和预测“great”），则在输入中将其替换为：
        - 80%时间为特殊的“<mask>“词元（例如，“this movie is great”变为“this movie is<mask>”；
        - 10%时间为随机词元（例如，“this movie is great”变为“this movie is drink”）；
        - 10%时间内为不变的标签词元（例如，“this movie is great”变为“this movie is great”）。
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)

        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)

        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """ BERT的下一句预测任务
    """
    """
        为了帮助理解两个文本序列之间的关系，BERT在预训练中考虑了一个二元分类任务——下一句预测。
        在为预训练生成句子对时，有一半的时间它们确实是标签为“真”的连续句子；
        在另一半的时间里，第二个句子是从语料库中随机抽取的，标记为“假”。
    """
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)

        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """ X的形状：(batchsize, num_hiddens)
        """
        return self.output(X)

class BERTModel(nn.Module):
    """ BERT模型
    """
    """
        在预训练BERT时，最终的损失函数是掩蔽语言模型损失函数和下一句预测损失函数的线性组合
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input, 
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()

        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                                   ffn_num_hiddens, num_heads, num_layers, dropout, max_len=max_len,
                                   key_size=key_size, query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),
            nn.Tanh()
        )
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pre_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pre_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pre_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat



def test_bert_encoder():
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                        ffn_num_hiddens, num_heads, num_layers, dropout)
    
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(encoded_X.shape)

def test_MaskLM():
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                        ffn_num_hiddens, num_heads, num_layers, dropout)
    
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)

    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(mlm_Y_hat.shape)

    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(mlm_l.shape)

def main():
    test_MaskLM()

if __name__ == '__main__' :
    main()

