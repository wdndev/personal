# 自注意力：Self-Attention

自注意力是目前应用最广泛的注意力机制之一，self-attention及其变体广泛应用与自然语言处理、图像处理及语音识别的各个领域，特别是NLP领域，基于self-attention的Transformer结构已经成为NLP技术的基石。CV领域的self-attention也来源于NLP，甚至在某些分割、识别任务上直接套用NLP的Transformer结构并且取得了非常好的结果。

自注意力的结构下图所示，它是从NLP中借鉴过来的思想，因此仍然保留了Query, Key和Value等名称。对应图中自上而下分的三个分支，计算时通常分为三步：

1.  第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等；
2.  第二步一般是使用一个softmax函数对这些权重进行归一化，转换为注意力；
3.  第三步将权重和相应的键值value进行加权求和得到最后的attention。
