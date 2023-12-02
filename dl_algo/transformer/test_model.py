from model.transformer import *
from model.utils import subsequent_mask
import matplotlib.pyplot as plt
import copy


def test_subsequent_mask():
    # print(np.triu([[1,2,3], [4,5,6], [7,8,9],[10,11,12]], k=-1))
    # print(np.triu([[1,2,3], [4,5,6], [7,8,9],[10,11,12]], k=0))
    # print(np.triu([[1,2,3], [4,5,6], [7,8,9],[10,11,12]], k=1))
    size = 5
    sm = subsequent_mask(size)
    print("sm: ", sm)

    plt.figure(figsize=(5,5))
    plt.imshow(subsequent_mask(20)[0])

    plt.show()

def show_positional():
    plt.figure(figsize=(15, 5))
    pe = PositionalEncoding(20, 0)
    # 向pe传入被Vaiiable封装的tensor，这样pe会执行forward函数
    # 且这个tensor里的数值都是0，被处理后相当于位置编码张量
    y = pe.forward(torch.zeros(1, 100, 20))

    # 横坐标到100的长度，纵坐标是某一个词汇中的某维特征在不同长度下对应的值
    # 因为总共有20维之多，我们这里只查看4，5，6，7维的值
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # 画布上填写维度提示信息
    plt.legend(["dim %d" % p for p in [4,5,6,7]])

    # 输出效果分析:
    ## 每条颜色的曲线代表某一个词汇中的特征在不同位置的含义.
    ## 保证同一词汇随着所在位置不同它对应位置嵌入向量会发生变化.
    ## 正弦波和余弦波的值域范围都是1到-1这又很好的控制了嵌入数值的大小,有助于梯度的快速计算.

    plt.show()

def test_Positional_encoding():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)
    print("embr: ", embr)
    print(embr.shape)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(embr)
    print("pe_res: ", pe_res)
    print(pe_res.shape)

    show_positional()

def test_positionwiseffn():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(embr)

    head = 8
    embedding_dim = 512
    dropout = 0.2

    query = key = value = pe_res

    mask = Variable(torch.zeros(8, 4, 4))

    mha = MultiHeadAttention(head, embedding_dim, dropout)
    mha_res = mha(query, key, value, mask)

    d_ff = 64
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    print(mha_res.shape)
    ff_res = ff(mha_res)

    print(ff_res)
    print(ff_res.shape)

def test_layer_norm():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_res = pe(embr)

    head = 8
    embedding_dim = 512
    dropout = 0.2

    query = key = value = pe_res

    mask = Variable(torch.zeros(8, 4, 4))

    mha = MultiHeadAttention(head, embedding_dim, dropout)
    mha_res = mha(query, key, value, mask)

    d_ff = 64
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    
    ff_result = ff(mha_res)

    features = d_model = 512
    eps = 1e-6

    
    x = ff_result
    ln = LayerNorm(features, eps)
    print(mha_res.shape)
    ln_result = ln(x)
    print(ln_result)
    print(ln_result.shape)

def test_sublayer_connection():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(embr)

    head = 8
    embedding_dim = 512
    dropout = 0.2

    x = pe_result

    mask = Variable(torch.zeros(8, 4, 4))

    mha = MultiHeadAttention(head, embedding_dim, dropout)

    sublayer = lambda x : mha(x, x, x, mask)
    size = 512

    sc = SublayerConnection(size, dropout)
    print(x.shape)
    sc_result = sc(x, sublayer)
    print(sc_result)
    print(sc_result.shape)

def test_encode_layer():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(embr)

    size = d_model = 512
    head = 8
    d_ff = 64
    x = pe_result
    dropout = 0.2

    self_attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    
    el = EncoderLayer(size, self_attn, ff, dropout)
    print(x.shape)
    el_result = el(x, mask)
    print(el_result)
    print(el_result.shape)

def test_encoder():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(embr)

    size = d_model = 512
    head = 8
    d_ff = 64
    x = pe_result
    dropout = 0.2
    N = 8

    self_attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    
    layer = EncoderLayer(size, copy.deepcopy(self_attn), copy.deepcopy(ff), dropout)

    encoder = Encoder(layer, N)

    print(x.shape)
    en_result = encoder(x, mask)
    print(en_result)
    print(en_result.shape)

    return en_result, encoder

def test_decoder_layer():

    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(embr)

    size = d_model = 512
    head = 8
    d_ff = 64
    x = pe_result
    dropout = 0.2
    N = 8

    self_attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    
    layer = EncoderLayer(size, copy.deepcopy(self_attn), copy.deepcopy(ff), dropout)

    encoder = Encoder(layer, N)
    en_result = encoder(x, mask)

    size = d_model = 512
    head = 8
    d_ff = 64
    dropout = 0.2

    self_attn = src_attn = MultiHeadAttention(head, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    x = pe_result

    memory = en_result

    mask = Variable(torch.zeros(8, 4, 4))
    src_mask = tgt_mask = mask

    dl = DecoderLayer(size, self_attn, src_attn, ff, dropout)
    print(x.shape)
    dl_result = dl(x, memory, src_mask, tgt_mask)
    print(dl_result)
    print(dl_result.shape)

    return dl_result

def test_deocder():
    d_model = 512
    dropout = 0.1
    max_len = 60

    vocab = 1000
    emb = Embeddings(d_model, vocab)
    input = Variable(torch.LongTensor([[1,2,4,5], [4,3,2,9]]))
    embr = emb(input)

    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(embr)

    size = d_model = 512
    head = 8
    d_ff = 64
    x = pe_result
    dropout = 0.2
    N = 8

    self_attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    mask = Variable(torch.zeros(8, 4, 4))
    
    layer = EncoderLayer(size, copy.deepcopy(self_attn), copy.deepcopy(ff), dropout)

    encoder = Encoder(layer, N)
    en_result = encoder(x, mask)

    size = d_model = 512
    head = 8
    d_ff = 64
    dropout = 0.2

    attn = MultiHeadAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    layer = DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout)

    N = 8
    x = pe_result
    memory = en_result
    mask = Variable(torch.zeros(8, 4, 4))
    src_mask = tgt_mask = mask

    de = Decoder(layer, N)

    print(x.shape)
    de_result = de(x, memory, src_mask, tgt_mask)
    print(de_result)
    print(de_result.shape)

    return de_result, de


def test_generator():
    de_result = test_deocder()

    d_model = 512
    vocab_size = 1000
    x = de_result

    gen = Generator(d_model, vocab_size)
    print(x.shape)
    gen_result = gen(x)
    print(gen_result)
    print(gen_result.shape)

    return gen_result, gen

def test_encoder_decoder():
    _, en = test_encoder()
    _, de = test_deocder()
    gen = Generator(d_model, vocab_size)

    vocab_size = 1000
    d_model = 512
    encoder = en
    decoder = de
    src_embed = nn.Embedding(vocab_size, d_model)
    tgt_embed = nn.Embedding(vocab_size, d_model)
    generator = Generator(d_model, vocab_size)

    src = tgt = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    src_mask = tgt_mask = Variable(torch.zeros(8, 4, 4))

    ed = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    ed_result = ed(src, tgt, src_mask, tgt_mask)
    print(ed_result)
    print(ed_result.shape)

def test_transformer_model():
    src_vocab = 11
    tgt_vocab = 11

    model = make_model(src_vocab, tgt_vocab)
    # model.eval()
    print(model)

def test_inference():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


if __name__ == '__main__' :
    
    for _ in range(10):
        test_inference()