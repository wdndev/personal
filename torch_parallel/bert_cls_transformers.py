""" Transformers 框架分布式训练
"""
import os
# # 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2,3"
# os.environ["WORLD_SIZE"] = "2"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5678'
import json
import time
import random
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.distributed as dist

from accelerate import Accelerator
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from collections import Counter
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from transformers import TrainingArguments, Trainer

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
            "labels": label_all
        }
        return collate_data  
    
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    correct_num = (predictions == labels).sum()
    correct_total = len(labels)
    return {"accuracy：": correct_num / correct_total}


@dataclass
class TrainArgs:
    # model_path = "model_hub/chinese-bert-wwm-ext"
    model_path = "model/chinese-bert-wwm-ext"
    ckpt_path = "output/bert_cls_transformers_4"
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
    
    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    
    # =======================================
    training_args = TrainingArguments(
        output_dir=args.ckpt_path,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_step,
        eval_steps=args.eval_step,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.valid_batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=args.eval_step,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
        seed=123,
        report_to="none",   # wandb不显示
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=collate.collate_fn,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    start = time.time()
    # 训练模型
    trainer.train()
    end = time.time()  # 记录结束时间
    print("耗时：{}秒".format(end - start))

    # 评估模型
    print(trainer.evaluate())
    
def test():
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
    valid_loader = DataLoader(valid_data,
                            batch_size=args.valid_batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn)
    test_loader = valid_loader

    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.ckpt_path + "/checkpoint-100", config=config)
    model.cuda()

    labels = list(label2id.keys())

    def test(model, test_loader, labels):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                label = batch_data["labels"].cuda()
                input_ids = batch_data["input_ids"].cuda()
                token_type_ids = batch_data["token_type_ids"].cuda()
                attention_mask = batch_data["attention_mask"].cuda()
                output = model.forward(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask,
                                                   labels=label)
                logits = output[1]
                label = label.view(-1).detach().cpu().numpy().tolist()
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1).flatten().tolist()
                trues.extend(label)
                preds.extend(pred)
        # print(trues, preds, labels)
        print(np.array(trues).shape, np.array(preds).shape)
        report = classification_report(trues, preds, target_names=labels)
        return report

    print(test(model, test_loader, labels))
    
    
# nohup python bert_cls_transformers.py > logs/bert_cls_transformers_2.log 2>&1 &
if __name__ == '__main__':
    
    main()
    test()
    