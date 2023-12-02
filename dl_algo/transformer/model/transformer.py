# -*- coding: utf-8 -*-
#  @file        - transformer.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - Transformer模型
#  @version     - 0.0
#  @date        - 2023.06.06
#  @copyright   - Copyright (c) 2023
import math
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from torch import nn 
from torch.autograd import Variable
import copy

from .multihead_attention import MultiHeadAttention
from .utils import clones



class Embeddings(nn.Module):
    """ 实现文本嵌入层
        无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转表为向量表示，
        希望在这样的高维空间捕捉词汇间的关系。
    """
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        """ 初始化
            - d_model：词嵌入的维度
            - vocab：词表的大小
        """

        # 获得词嵌入对象self.lut
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, X):
        """ 前向传播
        """
        return self.lut(X) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """ 位置编码类
        Transformer结构中，没有针对词汇位置信息的处理，因此需要在Embeding层后加入位置编码，
        将词汇位置不同可能会产生不同语义的信息加入到词嵌入张量中，以弥补位置信息的缺失。
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        """ 初始化
            - d_model: 词嵌入维度
            - dropout: 置0比例
            - max_len: 每个句子的最大长度
        """

        # 实例化nn中预定的Dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵，大小为max_len*d_model
        pe = torch.zeros(max_len, d_model)

        # 初始化一个绝对位置矩阵，在这里，词汇的绝对位置就是用它的索引取表示。
        # 首先使用arange方法获得一个连续自然数向量，然后再使用unsqeeze方法拓展向量维度
        # 参数为1，代表矩阵拓展的位置，会使向量变成一个max_len * 1的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 考虑如何将这些位置信息加入到位置编码矩阵中
        # 最简单的思路就是先将max_lemn*1的绝对位置矩阵，
        # 变换成max_len*d_model形状，然后覆盖原来的初始位置编码即可。
        # 变换矩阵 1*d_model -> max_len*d_model，
        # 还希望将自然数的绝对位置编码缩放成足够小的数，有助于后续梯度下降中的收敛。
        # torch.arange(0, d_model, 2), 初始化了一半，可以看作使初始化了两次，
        # 每次初始化做不同的处理，填充再位置编码矩阵的偶数和奇数位置上
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             - (math.log(10000.0) / d_model))
        # 将前面定义变换矩阵div_term，对奇数和偶数不同赋值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe为二维矩阵，而embedding的输出为三维，扩展矩阵
        pe = pe.unsqueeze(0)

        # 把pe位置编码矩阵注册成模型的buffer
        # buffer，对模型有效果，但不是超参，不需要随优化器调整，注册之后可以随模型加载和保存
        self.register_buffer("pe", pe)

    def forward(self, X):
        """ 前向传播
            -X: 文本序列的词嵌入表示
        """
        # self.pe[:, :X.size(1) 截取部分pe，与输入匹配
        # 明确pe的编码太长了，将第二个维度，
        # 也就是max_len对应的那个维度缩小成x的句子长度同等的长度
        # X = X + Variable(self.pe[:, :X.size(1)], requires_grad=False)
        X = X +  self.pe[:, : X.size(1)].requires_grad_(False)

        return self.dropout(X)

class PositionwiseFeedForward(nn.Module):
    """ 实现前馈全连接层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
            - d_model: 线性层的输入维度也是第二个线性层的输出维度，
                    因为我们希望输入通过前馈全连接层后输入和输出的维度不变. 
           - d_ff: 第二个线性层的输入维度和第一个线性层的输出维度. 
           - dropout: 置0比率
        """
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 首先经过第一个线性层，然后使用Funtional中relu函数进行激活,
        # 之后再使用dropout进行随机置0，最后通过第二个线性层w2，返回最终结果.
        return self.w2(self.dropout(self.w1(x).relu()))
    
class LayerNorm(nn.Module):
    """ 实现规范化层的类
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        """ 初始化参数
            - features: 表示词嵌入的维度,
            - eps: 一个足够小的数, 在规范化公式的分母中出现,防止分母为0.默认是1e-6.
        """

        # 根据features的形状初始化两个参数张量a2，和b2，第一个初始化为1张量，
        # 也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，
        # 这两个张量就是规范化层的参数，因为直接对上一层得到的结果做规范化公式计算，
        # 将改变结果的正常表征，因此就需要有参数作为调节因子，使其即能满足规范化要求，
        # 又能不改变针对目标的表征.最后使用nn.parameter封装，代表他们是模型的参数。
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):
        """
        """
        # 在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致.
        # 接着再求最后一个维度的标准差，然后就是根据规范化公式，
        # 用x减去均值除以标准差获得规范化的结果，
        # 最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进行乘法操作，
        # 加上位移参数b2.返回即可.
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """ 子层连接结构的类
    """
    def __init__(self, size, dropout=0.1):
        """ 初始化
            - size: 词嵌入维度的大小， 
            - dropout: 是对模型结构中的节点数进行随机抑制的比率， 
        """
        super(SublayerConnection, self).__init__()
        # 实例化了规范化对象self.norm
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ 前向逻辑函数中, 接收上一个层或者子层的输入作为第一个参数，
           将该子层连接中的子层函数作为第二个参数
        """

        # 我们首先对输出进行规范化，然后将结果传给子层处理，之后再对子层进行dropout操作，
        # 随机停止一些网络中神经元的作用，来防止过拟合. 最后还有一个add操作， 
        # 因为存在跳跃连接，所以是将输入x与dropout后的子层输出结果相加作为最终的子层连接输出.
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """ 编码器层
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """ 初始化
            - size: 词嵌入维度的大小，它也将作为我们编码器层的大小, 
            - self_attn: 传入多头自注意力子层实例化对象, 并且是自注意力机制, 
            - eed_froward: 传入前馈全连接层实例化对象,
            - dropout: 置0比率
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        # 如图所示, 编码器层中有两个子层连接结构, 所以使用clones函数进行克隆
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        """ x和mask，分别代表上一层的输出，和掩码张量mask
        """
        #  首先通过第一个子层连接结构，其中包含多头自注意力子层，
        # 然后通过第二个子层连接结构，其中包含前馈全连接子层. 最后返回结果.
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, self.feed_forward)
    
class Encoder(nn.Module):
    """ 实现编码器
    """
    def __init__(self, layer, N):
        """ 初始化
            - layer: 编码器层
            - N: 编码器层的个数
        """
        super(Encoder, self).__init__()
        # 使用clones函数克隆N个编码器层放在self.layers中
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """forward函数的输入和编码器层相同, 
            - x: 上一层的输出, 
            - mask: 掩码张量
        """
        # 首先就是对我们克隆的编码器层进行循环，每次都会得到一个新的x，
        # 这个循环的过程，就相当于输出的x经过了N个编码器层的处理. 
        # 最后再通过规范化层的对象self.norm进行处理，最后返回结果. 
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """ 解码器层
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """ 初始化
            - size：代表词嵌入的维度大小, 同时也代表解码器层的尺寸，
            - self_attn： 多头自注意力对象，也就是说这个注意力机制需要Q=K=V， 
            - src_attn：多头注意力对象，这里Q!=K=V， 
            - feed_forward： 前馈全连接层对象，
            - droupout：置0比率.
        """
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        # 按照结构图使用clones函数克隆三个子层连接对象.
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """forward函数中的参数有4个，
            - x: 来自上一层的输入x，
            - mermory: 来自编码器层的语义存储变量 
            - src_mask: 源数据掩码张量
            - tgt_mask: 目标数据掩码张量.
        """
        # 将memory表示成m方便之后使用
        m = memory

        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，
        # 因为是自注意力机制，所以Q,K,V都是x，最后一个参数是目标数据掩码张量，
        # 这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，
        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，
        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，
        # 同样生成第二个字符或词汇时，
        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.
        x = self.sublayer[0](x, lambda x : self.self_attn(x, x, x, tgt_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory， 
        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，
        # 而是遮蔽掉对结果没有意义的字符而产生的注意力值，
        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.
        x = self.sublayer[1](x, lambda x : self.src_attn(x, m, m, src_mask))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    """ 实现解码器
    """
    def __init__(self, layer, N):
        """ 初始化
            - layer： 解码器层
            - N：解码器层个数
        """
        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层. 
        # 因为数据走过了所有的解码器层后最后要做规范化处理. 
        self.layers = clones(layer, N)

        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
            - x: 来自上一层的输入x，
            - mermory: 来自编码器层的语义存储变量 
            - src_mask: 源数据掩码张量
            - tgt_mask: 目标数据掩码张量.
        """
        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，
        # 得出最后的结果，再进行一次规范化返回即可.
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    """ 生成器类
        将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构
        因此把类的名字叫做Generator, 生成器类
    """

    def __init__(self, d_model, vocab_size):
        """
            - d_model: 词嵌入维度
            - vocab_size: 词表的总大小
        """
        super(Generator, self).__init__()

        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
            - x: 上一层的输出张量
        """
        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化, 
        # 然后使用F中已经实现的log_softmax进行的softmax处理.
        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.
        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数, 
        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.
        return log_softmax(self.project(x), dim=-1)

class EncoderDecoder(nn.Module):
    """ 实现编码器-解码器结构
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """ 初始化
            - encoder： 编码器对象
            - decoder： 解码器对象
            - src_embed：源数据嵌入函数
            - tgt_embed： 目标数据嵌入函数
            - generator：输出部分类别生成器
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
            - src：源数据
            - tgt：目标数据
            - src_mask：源数据的掩码张量
            - tgt_mask：目标数据的掩码张量
        """
        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,
        # 与source_mask，target，和target_mask一同传给解码函数.
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        """ 编码函数
            - src：源数据
            - src_mask：源数据的掩码张量
        """
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """ 解码函数
            - memory：经历编码器编码后的输出张量
            - src_mask：源数据的掩码张量
            - tgt：目标数据
            - tgt_mask：目标数据的掩码张量
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """ 构建transformer模型
        - src_vocab: 源数据特征(词汇)总数
        - tgt_vocab: 目标数据特征(词汇)总数
        - N: 编码器和解码器堆叠数
        - d_model: 词向量映射维度
        - d_ff: 前馈全连接网络中变换矩阵的维度
        - h : 多头注意力结构中的多头数
        - dropout: 置零比率
    """
    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
    # 来保证他们彼此之间相互独立，不受干扰.
    c = copy.deepcopy
    # 实例化了多头注意力类，得到对象attn
    attn = MultiHeadAttention(h, d_model)
    # 然后实例化前馈全连接类，得到对象ff 
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化位置编码类，得到对象position
    position = PositionalEncoding(d_model, dropout)
    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层. 
    # 在编码器层中有attention子层以及前馈全连接子层，
    # 在解码器层中有两个attention子层以及前馈全连接层.
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model

class TransformerModelMy(nn.Module):
    """ transform 类
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        """ 构建transformer模型
            - src_vocab: 源数据特征(词汇)总数
            - tgt_vocab: 目标数据特征(词汇)总数
            - N: 编码器和解码器堆叠数
            - d_model: 词向量映射维度
            - d_ff: 前馈全连接网络中变换矩阵的维度
            - h : 多头注意力结构中的多头数
            - dropout: 置零比率
        """
        super(TransformerModelMy, self).__init__()

        # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，
        # 来保证他们彼此之间相互独立，不受干扰.
        c = copy.deepcopy
        # 实例化了多头注意力类，得到对象attn
        attn = MultiHeadAttention(h, d_model)
        # 然后实例化前馈全连接类，得到对象ff 
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 实例化位置编码类，得到对象position
        position = PositionalEncoding(d_model, dropout)
        # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，
        # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，
        # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层. 
        # 在编码器层中有attention子层以及前馈全连接子层，
        # 在解码器层中有两个attention子层以及前馈全连接层.
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model, src_vocab_size), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab_size), c(position)),
            Generator(d_model, tgt_vocab_size)
        )
        
        self.init_weights()
        

    def init_weights(self):
        """
        """
        # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵
        # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
            - src：源数据
            - tgt：目标数据
            - src_mask：源数据的掩码张量
            - tgt_mask：目标数据的掩码张量
        """
        return self.model(src, tgt, src_mask, tgt_mask)

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    


    