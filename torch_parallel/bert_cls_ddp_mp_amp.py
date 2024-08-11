""" 分布式数据并行 (DistributedDataParallel) 分布式训练, 使用混合精度训练
"""

import os
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.multiprocessing as mp

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
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

def build_optimizer(model, args):
    """ 构建优化器
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer         

class Trainer:
    """ Trainer类，用于训练、验证和测试模型
    """
    # 初始化方法，接收训练参数args、配置config、模型model、损失函数criterion和优化器optimizer
    def __init__(self, args, model, criterion, optimizer):
        self.args = args  # 训练参数
        self.model = model  # 待训练的模型
        self.criterion = criterion  # 损失函数
        self.optimizer = optimizer  # 优化器

    # 处理单个训练步长，获取logits和真实标签
    def one_step(self, batch_data):
        """ 计算一步
        """
        # ===========  第五步：根据local_rank将数据分发给指定的GPU  ===========
        label = batch_data["label"].cuda()
        input_ids = batch_data["input_ids"].cuda()
        token_type_ids = batch_data["token_type_ids"].cuda()
        attention_mask = batch_data["attention_mask"].cuda()
        
        # 模型前向传播计算输出
        output = self.model(
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
        rt /= self.args.local_world_size
        return rt
    
    def output_reduce(self, outputs, targets):
        output_gather_list = [torch.zeros_like(outputs) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(output_gather_list, outputs)

        outputs = torch.cat(output_gather_list, dim=0)
        target_gather_list = [torch.zeros_like(targets) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(target_gather_list, targets)
        targets = torch.cat(target_gather_list, dim=0)
        return outputs, targets


    def train(self, train_loader, valid_loader=None, train_sampler=None):
        """ 训练
        """
        global_step = 1  # 全局训练步数
        best_acc = 0.  # 最佳验证准确率
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        if self.args.local_rank == 0:
            start = time.time()  # 记录开始时间
        
        # 遍历所有训练轮次
        for epoch in range(1, self.args.epochs + 1):
            # ===========  第四步：训练时每一个epoch打乱数据  ===========
            train_sampler.set_epoch(epoch)
            for step, batch_data in enumerate(train_loader):
                # 设置模型为训练模式
                self.model.train()
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        logits, label = self.one_step(batch_data)
                        loss = self.criterion(logits, label)
                        torch.distributed.barrier()
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    # 调用on_step方法获取logits和真实标签
                    logits, label = self.one_step(batch_data)
                    # 损失计算
                    loss = self.criterion(logits, label)
                    # ===========  第六步：等待所有GPU  ===========
                    torch.distributed.barrier()
                    # 更新梯度
                    # self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # ===========  第七步：reduce计算损失  ===========
                loss = self.loss_reduce(loss)
                # ===========  第八步：在主rank打印指标  ===========
                if self.args.local_rank == 0:
                    # 输出训练日志
                    print("[train] epoch: {}/{} step：{}/{} loss：{:.6f}".format(
                        epoch, self.args.epochs, global_step, self.args.total_step, loss.item()
                    ))
                
                global_step += 1
                
                # 若开启验证，在特定步数下进行验证
                if self.args.valid and global_step % self.args.eval_step == 0:
                    loss_vaild, accuracy = self.valid(valid_loader)
                    # ===========  第九步：在主rank保存模型  ===========
                    if self.args.local_rank == 0:
                        print("[valid] loss：{:.6f} accuracy：{:.4f}".format(loss_vaild, accuracy))
                        
                        # 若当前验证准确率高于历史最佳，则保存模型
                        if accuracy > best_acc:
                            best_acc = accuracy
                            print("【best accuracy】 {:.4f}".format(best_acc))
                            torch.save(self.model.state_dict(), self.args.ckpt_path)
        if self.args.local_rank == 0:
            end = time.time()  # 记录结束时间
            print("耗时：{}秒".format(end - start))
        
        # 若未开启验证，在训练结束后保存模型
        if not self.args.valid and self.args.local_rank == 0:
            torch.save(self.model.state_dict(), self.args.ckpt_path)

    def valid(self, valid_loader):
        """ 验证阶段
        """
        self.model.eval()  # 设置模型为评估模式
        correct_total = 0  # 累计正确预测数量
        num_total = 0  # 总样本数
        loss_total = 0.  # 累计损失
        
        # 在无需计算梯度的情况下进行评估
        with torch.no_grad():
            for step, batch_data in enumerate(valid_loader):
                logits, label = self.one_step(batch_data)
                loss = self.criterion(logits, label)
                
                torch.distributed.barrier()
                loss = self.loss_reduce(loss)
                loss_total += loss
                # ===========  第十步：reduce得到结果  ===========
                logits, label = self.output_reduce(logits, label)
                
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
        self.model = model  # 更新模型（可加载已保存的最佳模型）
        self.model.eval()  # 设置模型为评估模式
        
        # 存储预测结果和真实标签
        preds = []
        trues = []
        
        # 在无需计算梯度的情况下进行预测
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                logits, label = self.one_step(batch_data)
                torch.distributed.barrier()
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
    ckpt_path = "output/bert_cls_ddp_mp_amp_4.pt"
    dataset_path = "dataset/train.json"
    max_seq_len = 128
    ratio = 0.92
    train_batch_size = 64
    valid_batch_size = 64
    weight_decay = 0.01
    epochs = 10
    learning_rate = 3e-5
    eval_step = 100
    valid = False
    
    local_rank = None
    local_world_size = None
    device_ids = None
    rank = None
    
    use_amp = True

def main_worker(local_rank, local_world_size):
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
    
    # ===========  第一步：初始化  ===========
    dist.init_process_group(backend="nccl", 
                            init_method="tcp://localhost:12345", 
                            world_size=local_world_size,
                            rank=local_rank)
    n = torch.cuda.device_count() // local_world_size
    device_ids = [local_rank]
    print(
        f"[{os.getpid()}] rank = {local_rank}, "
        + f"world_size = {local_world_size}, n = {n}, device_ids = {device_ids} \n", end=''
    )

    torch.cuda.set_device(local_rank)
    
    args = TrainArgs()
    args.local_world_size = local_world_size
    args.local_rank = local_rank
    args.device_ids = device_ids
    args.rank = local_rank
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
    # ===========  第二步：DistributedSampler  ===========
    train_dataset = ClsDataset(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              collate_fn=collate.collate_fn,
                              sampler=train_sampler)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    valid_dataset = ClsDataset(valid_data)
    vaild_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset,
                            batch_size=args.valid_batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn,
                            sampler=vaild_sampler)
    test_loader = valid_loader
    
    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    # ===========  第三步：封装模型  ===========
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_ids)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)

    # =======================================
    # 定义训练器
    trainer = Trainer(args, model, criterion, optimizer)
    
    # 训练和验证
    trainer.train(train_loader, valid_loader, train_sampler)

    # 测试
    labels = list(label2id.keys())
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_ids)
    model.load_state_dict(torch.load(args.ckpt_path))
    report = trainer.test(model, test_loader, labels)
    if args.local_rank == 0:
        print(report)
        
    # =======================================
    # ===========  第十一步  ===========
    dist.destroy_process_group()
    # =======================================
    
    print("===============end=====================")

# nohup python bert_cls_ddp_mp_amp.py --local_world_size=2  > logs/bert_cls_ddp_mp_amp_2.log 2>&1 &
if __name__ == '__main__':
    # =======================================
    # 输入参数定义
    import argparse
    parser = argparse.ArgumentParser()
    # ===========  第零步：需要定义一个参数local-rank  ===========
    parser.add_argument("--local_world_size", type=int, default=1)
    p_args = parser.parse_args()
    
    # 启动进程
    mp.spawn(main_worker, nprocs=p_args.local_world_size, args=(p_args.local_world_size,))
    
    
    