""" DeepSpeed框架分布式训练
"""

import json
import time
import random
import torch
import deepspeed
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.distributed as dist

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW

def set_seed(seed=121):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_datasets(file_path):
    """ 加载数据集
    """
    with open(file_path, 'r', encoding='utf-8') as fp:
        data = fp.read()
    data = json.loads(data)
    text_label = []
    # [(文本， 标签id)]
    for d in data:
        text = d[0]
        label = d[1]
        text_label.append(("".join(text.split(" ")).strip(), label))
    
    return text_label

class ClsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class DataCollate:
    """ 数据处理类
        - 利用tokenizer对每条文本进行编码，得到input_ids、token_type_ids和attention_mask；
        - 将每批数据的编码结果统一规整为固定长度，并填充或截断至max_seq_len；
        - 将处理过的数据转换为PyTorch张量，方便后续送入模型训练；
        - 最后将这些张量打包成字典形式返回，供数据加载器传递给模型训练。
    """
    def __init__(self, tokenizer, max_seq_len) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def collate_fn(self, batch):
        input_ids_all = []
        token_type_ids_all = []
        attention_mask_all = []
        label_all = []
        
        for data in batch:
            text = data[0]
            label = data[1]
            
            inputs = self.tokenizer.encode_plus(text=text,
                                                max_length=self.max_seq_len,
                                                padding="max_length",
                                                truncation="longest_first",
                                                return_attention_mask=True,
                                                return_token_type_ids=True)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            
            input_ids_all.append(input_ids)
            token_type_ids_all.append(token_type_ids)
            attention_mask_all.append(attention_mask)
            label_all.append(label) 
        
        input_ids_all = torch.tensor(input_ids_all, dtype=torch.long)
        token_type_ids_all = torch.tensor(token_type_ids_all, dtype=torch.long)
        attention_mask_all = torch.tensor(attention_mask_all, dtype=torch.long)
        label_all = torch.tensor(label_all, dtype=torch.long)
        collate_data = {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "token_type_ids": token_type_ids_all,
            "label": label_all
        }
        return collate_data  

# def build_optimizer(model, args):
#     """ 构建优化器
#     """
#     no_decay = ['bias', 'LayerNorm.weight']
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#          'weight_decay': args.weight_decay},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#          'weight_decay': 0.0}
#     ]

#     # optimizer = AdamW(model.parameters(), lr=learning_rate)
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
#     return optimizer         

class Trainer:
    """ Trainer类，用于训练、验证和测试模型
    """
    # 初始化方法，接收训练参数args、配置config、模型model、损失函数criterion和优化器optimizer
    def __init__(self, args, model_engine, criterion, optimizer):
        self.args = args  # 训练参数
        self.model_engine = model_engine  # 待训练的模型
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器

    # 处理单个训练步长，获取logits和真实标签
    def one_step(self, batch_data):
        """ 计算一步
        """
        label = batch_data["label"].cuda()
        input_ids = batch_data["input_ids"].cuda()
        token_type_ids = batch_data["token_type_ids"].cuda()
        attention_mask = batch_data["attention_mask"].cuda()
        
        # 模型前向传播计算输出
        output = self.model_engine.forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=label
            )
        
        # 提取logits（预测值）用于计算损失
        logits = output[1]
        
        return logits, label

    def loss_reduce(self, loss):
        rt = loss.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= torch.cuda.device_count()
        return rt

    def output_reduce(self, outputs, targets):
        output_gather_list = [torch.zeros_like(outputs) for _ in range(torch.cuda.device_count())]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(output_gather_list, outputs)

        outputs = torch.cat(output_gather_list, dim=0)
        target_gather_list = [torch.zeros_like(targets) for _ in range(torch.cuda.device_count())]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(target_gather_list, targets)
        targets = torch.cat(target_gather_list, dim=0)
        return outputs, targets

    def train(self, train_loader, valid_loader=None):
        """ 训练
        """
        global_step = 1  # 全局训练步数
        best_acc = 0.  # 最佳验证准确率
        if self.args.local_rank == 0:
            start = time.time()  # 记录开始时间
        
        # 遍历所有训练轮次
        for epoch in range(1, self.args.epochs + 1):
            for step, batch_data in enumerate(train_loader):
                # 设置模型为训练模式
                self.model_engine.train()
                # 调用on_step方法获取logits和真实标签
                logits, label = self.one_step(batch_data)
                
                # 损失计算
                loss = self.criterion(logits, label)
                
                # 更新梯度
                self.model_engine.backward(loss)
                self.model_engine.step()
                loss = self.loss_reduce(loss)
                
                # 输出训练日志
                if self.args.local_rank == 0:
                    print("[train] epoch: {}/{} step：{}/{} loss：{:.6f}".format(
                        epoch, self.args.epochs, global_step, self.args.total_step, loss.item()
                    ))
                
                global_step += 1
                
                # 若开启验证，在特定步数下进行验证
                if self.args.valid and global_step % self.args.eval_step == 0:
                    loss_vaild, accuracy = self.valid(valid_loader)
                    if self.args.local_rank == 0:
                        print("[valid] loss：{:.6f} accuracy：{:.4f}".format(loss_vaild, accuracy))
                    
                    # 若当前验证准确率高于历史最佳，则保存模型
                    if accuracy > best_acc:
                        best_acc = accuracy
                        self.model_engine.save_checkpoint(self.args.ckpt_path, save_latest=True)
                        if self.args.local_rank == 0:
                            print("【best accuracy】 {:.4f}".format(best_acc))
        
        if self.args.local_rank == 0:
            end = time.time()  # 记录结束时间
            print("耗时：{}秒".format(end - start))
        
        # 若未开启验证，在训练结束后保存模型
        if not self.args.valid:
            self.model_engine.save_checkpoint(self.args.ckpt_path, save_latest=True)

    def valid(self, valid_loader):
        """ 验证阶段
        """
        self.model_engine.eval()  # 设置模型为评估模式
        correct_total = 0  # 累计正确预测数量
        num_total = 0  # 总样本数
        loss_total = 0.  # 累计损失
        
        # 在无需计算梯度的情况下进行评估
        with torch.no_grad():
            for step, batch_data in enumerate(valid_loader):
                logits, label = self.one_step(batch_data)
                loss = self.criterion(logits, label)
                
                loss = self.loss_reduce(loss)
                logits, label = self.output_reduce(logits, label)
                loss_total += loss.item()
                
                # 将数据转换为numpy数组并在CPU上计算精度
                logits = logits.detach().cpu().numpy()
                label = label.view(-1).detach().cpu().numpy()
                num_total += len(label)
                preds = np.argmax(logits, axis=1).flatten()
                correct_num = (preds == label).sum()
                correct_total += correct_num
                
        # 返回平均损失和准确率
        return loss_total, correct_total / num_total

    def test(self, model, test_loader, labels):
        """ 测试
        """
        self.model_engine = model  # 更新模型（可加载已保存的最佳模型）
        self.model_engine.eval()  # 设置模型为评估模式
        
        # 存储预测结果和真实标签
        preds = []
        trues = []
        
        # 在无需计算梯度的情况下进行预测
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                logits, label = self.one_step(batch_data)
                logits, label = self.output_reduce(logits, label)
                label = label.view(-1).detach().cpu().numpy().tolist()
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1).flatten().tolist()
                trues.extend(label)
                preds.extend(pred)
        
        # 输出真值和预测值形状，计算并打印分类报告
        print(np.array(trues).shape, np.array(preds).shape)
        report = classification_report(trues, preds, target_names=labels)
        return report

@dataclass
class TrainArgs:
    # model_path = "model_hub/chinese-bert-wwm-ext"
    model_path = "model/chinese-bert-wwm-ext"
    ckpt_path = "output/bert_cls_deepspeed_4"
    dataset_path = "dataset/train.json"
    max_seq_len = 128
    ratio = 0.92
    epochs = 10
    eval_step = 100
    valid = False
    
    local_rank = None

deepspeed_config = {
    "train_micro_batch_size_per_gpu": 64,
    "gradient_accumulation_steps": 1,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-5
        }
    },
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8
    },
    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True
    },
    "wall_clock_breakdown": True,
    "log_dist": False,
}


def main():
    # =======================================
    # 定义相关参数
    set_seed()
    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }
    args = TrainArgs()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    # =======================================
    # 加载数据集
    data = load_datasets(args.dataset_path)
    # 取1万条数据出来
    data = data[:10000]
    random.shuffle(data)
    train_num = int(len(data) * args.ratio)
    train_data = data[:train_num]
    valid_data = data[train_num:]

    collate = DataCollate(tokenizer, args.max_seq_len)
    train_dataset = ClsDataset(train_data)
    train_loader = DataLoader(train_data,
                              batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate.collate_fn)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    valid_data = ClsDataset(valid_data)
    valid_loader = DataLoader(valid_data,
                            batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn)
    test_loader = valid_loader
    
    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    model_engine, optimizer, _, _ = deepspeed.initialize(
            config=deepspeed_config,
            model=model,
            model_parameters=model.parameters()
        )

    args.local_rank = model_engine.local_rank

    # =======================================
    # 定义训练器
    trainer = Trainer(args, model_engine, criterion, optimizer)
    
    # 训练和验证
    trainer.train(train_loader, valid_loader)

    # 测试
    labels = list(label2id.keys())
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda()
    
    # 需要重新初始化引擎
    model_engine, optimizer, _, _ = deepspeed.initialize(
            config=deepspeed_config,
            model=model,
            model_parameters=model.parameters()
        )
    model_engine.load_checkpoint(args.ckpt_path, load_module_only=True)
    report = trainer.test(model_engine, test_loader, labels)
    if args.local_rank == 0:
        print(report)
    print("===============end=====================")
    
    
# nohup deepspeed --master_port 1122 --num_gpus=2 bert_cls_deepspeed.py > logs/bert_cls_deepspeed_2.log 2>&1 &
if __name__ == '__main__':
    main()
    