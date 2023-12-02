# Transformer 实现

# 1.Description

Transformer算法pytorch实现

Transformer架构解析可参考如下文章：

[Transformer架构解析](https://wdndev.github.io/2023/05/25/Transformer/Transformer%E6%9E%B6%E6%9E%84%E8%A7%A3%E6%9E%90)

# 2.Environment

- Python 3
- Pytorch

# 3.Directory

```shell
  ├── .data: wikitext-2数据集目录，用于后续NLP任务
  ├── copy_task: Transformer实现COPY任务源码
  ├── language_demo: 语言任务
  ├── multihead_attention: 多头注意力源码
  ├── positional_encoding: 位置编码源码
  ├── test_copy_task: Copy任务测试文件
  ├── test_model: Model 测试文件
  ├── transformer_model： transformer模型实现
  └── transformer_utils: transformer模型中需要的一些共有函数
```
# 4.Running

## 4.1 test_model

用于测试各个模块是否编写正常，可以在`test_model.py`主函数中调用任意模块测试函数

## 4.2 Copy task

训练
```shell
Epoch Step:      1 | Accumulation Step:   2 | Loss:   3.36 | Tokens / Sec:   943.6 | Learning Rate: 5.5e-06
Epoch Step:      1 | Accumulation Step:   2 | Loss:   2.17 | Tokens / Sec:   940.5 | Learning Rate: 6.1e-05
Epoch Step:      1 | Accumulation Step:   2 | Loss:   1.80 | Tokens / Sec:  1065.6 | Learning Rate: 1.2e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   1.54 | Tokens / Sec:   923.8 | Learning Rate: 1.7e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   1.08 | Tokens / Sec:  1065.7 | Learning Rate: 2.3e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.67 | Tokens / Sec:  1164.4 | Learning Rate: 2.8e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.45 | Tokens / Sec:  1163.5 | Learning Rate: 3.4e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.20 | Tokens / Sec:   941.9 | Learning Rate: 3.9e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.18 | Tokens / Sec:  1229.9 | Learning Rate: 4.5e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.13 | Tokens / Sec:  1111.1 | Learning Rate: 5.0e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.26 | Tokens / Sec:  1210.0 | Learning Rate: 5.6e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.14 | Tokens / Sec:   952.8 | Learning Rate: 6.1e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.19 | Tokens / Sec:  1042.8 | Learning Rate: 6.7e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.14 | Tokens / Sec:  1084.0 | Learning Rate: 7.2e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.11 | Tokens / Sec:  1207.2 | Learning Rate: 7.8e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.07 | Tokens / Sec:  1188.8 | Learning Rate: 8.3e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.16 | Tokens / Sec:  1184.5 | Learning Rate: 8.9e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.12 | Tokens / Sec:  1143.7 | Learning Rate: 9.4e-04
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.05 | Tokens / Sec:  1218.7 | Learning Rate: 1.0e-03
Epoch Step:      1 | Accumulation Step:   2 | Loss:   0.13 | Tokens / Sec:  1203.0 | Learning Rate: 1.1e-03
```

运行结果
```shell
tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
```

## 4.3 Language Demo

使用wikitext-2数据集训练transformer

训练
```shell
| epoch   1 |   200/ 2981 batches | lr 5.00 | ms/batch 30.03 | loss  7.68 | ppl  2158.52
| epoch   1 |   400/ 2981 batches | lr 5.00 | ms/batch 28.90 | loss  5.26 | ppl   193.39
| epoch   1 |   600/ 2981 batches | lr 5.00 | ms/batch 28.90 | loss  4.07 | ppl    58.44
| epoch   1 |   800/ 2981 batches | lr 5.00 | ms/batch 28.88 | loss  3.41 | ppl    30.26
| epoch   1 |  1000/ 2981 batches | lr 5.00 | ms/batch 28.89 | loss  2.98 | ppl    19.72
| epoch   1 |  1200/ 2981 batches | lr 5.00 | ms/batch 28.90 | loss  2.79 | ppl    16.30
| epoch   1 |  1400/ 2981 batches | lr 5.00 | ms/batch 28.91 | loss  2.67 | ppl    14.38
| epoch   1 |  1600/ 2981 batches | lr 5.00 | ms/batch 28.92 | loss  2.58 | ppl    13.19
| epoch   1 |  1800/ 2981 batches | lr 5.00 | ms/batch 28.91 | loss  2.43 | ppl    11.32
| epoch   1 |  2000/ 2981 batches | lr 5.00 | ms/batch 28.92 | loss  2.39 | ppl    10.93
| epoch   1 |  2200/ 2981 batches | lr 5.00 | ms/batch 28.91 | loss  2.33 | ppl    10.24
| epoch   1 |  2400/ 2981 batches | lr 5.00 | ms/batch 28.91 | loss  2.36 | ppl    10.59
| epoch   1 |  2600/ 2981 batches | lr 5.00 | ms/batch 28.90 | loss  2.33 | ppl    10.31
| epoch   1 |  2800/ 2981 batches | lr 5.00 | ms/batch 28.92 | loss  2.26 | ppl     9.54
-----------------------------------------------------------------------------------------
| end of epoch   1 | time: 90.01s | valid loss  1.32 | valid ppl     3.73
-----------------------------------------------------------------------------------------
| epoch   2 |   200/ 2981 batches | lr 4.75 | ms/batch 29.08 | loss  2.18 | ppl     8.83
| epoch   2 |   400/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  2.11 | ppl     8.24
| epoch   2 |   600/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  1.98 | ppl     7.23
| epoch   2 |   800/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  2.00 | ppl     7.39
| epoch   2 |  1000/ 2981 batches | lr 4.75 | ms/batch 28.94 | loss  1.94 | ppl     6.96
| epoch   2 |  1200/ 2981 batches | lr 4.75 | ms/batch 28.92 | loss  1.97 | ppl     7.15
| epoch   2 |  1400/ 2981 batches | lr 4.75 | ms/batch 28.94 | loss  1.98 | ppl     7.28
| epoch   2 |  1600/ 2981 batches | lr 4.75 | ms/batch 28.92 | loss  1.97 | ppl     7.16
| epoch   2 |  1800/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  1.92 | ppl     6.84
| epoch   2 |  2000/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  1.96 | ppl     7.11
| epoch   2 |  2200/ 2981 batches | lr 4.75 | ms/batch 28.93 | loss  1.92 | ppl     6.80
| epoch   2 |  2400/ 2981 batches | lr 4.75 | ms/batch 28.94 | loss  1.94 | ppl     6.93
| epoch   2 |  2600/ 2981 batches | lr 4.75 | ms/batch 28.76 | loss  1.91 | ppl     6.76
| epoch   2 |  2800/ 2981 batches | lr 4.75 | ms/batch 28.75 | loss  1.89 | ppl     6.64
-----------------------------------------------------------------------------------------
| end of epoch   2 | time: 89.71s | valid loss  1.01 | valid ppl     2.74
-----------------------------------------------------------------------------------------
| epoch   3 |   200/ 2981 batches | lr 4.51 | ms/batch 28.88 | loss  1.78 | ppl     5.96
| epoch   3 |   400/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.89 | ppl     6.59
| epoch   3 |   600/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.72 | ppl     5.58
| epoch   3 |   800/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.73 | ppl     5.63
| epoch   3 |  1000/ 2981 batches | lr 4.51 | ms/batch 28.73 | loss  1.65 | ppl     5.22
| epoch   3 |  1200/ 2981 batches | lr 4.51 | ms/batch 28.74 | loss  1.69 | ppl     5.40
| epoch   3 |  1400/ 2981 batches | lr 4.51 | ms/batch 28.74 | loss  1.73 | ppl     5.66
| epoch   3 |  1600/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.75 | ppl     5.73
| epoch   3 |  1800/ 2981 batches | lr 4.51 | ms/batch 28.74 | loss  1.67 | ppl     5.33
| epoch   3 |  2000/ 2981 batches | lr 4.51 | ms/batch 28.74 | loss  1.69 | ppl     5.41
| epoch   3 |  2200/ 2981 batches | lr 4.51 | ms/batch 28.74 | loss  1.66 | ppl     5.26
| epoch   3 |  2400/ 2981 batches | lr 4.51 | ms/batch 28.76 | loss  1.69 | ppl     5.43
| epoch   3 |  2600/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.71 | ppl     5.55
| epoch   3 |  2800/ 2981 batches | lr 4.51 | ms/batch 28.75 | loss  1.72 | ppl     5.58
-----------------------------------------------------------------------------------------
| end of epoch   3 | time: 89.26s | valid loss  0.85 | valid ppl     2.33
-----------------------------------------------------------------------------------------
```

运行结果
```shell
=========================================================================================
| End of training | test loss  0.83 | test ppl     2.30
=========================================================================================
```



