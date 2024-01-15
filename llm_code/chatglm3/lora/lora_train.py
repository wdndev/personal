import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import pandas as pd
from peft import TaskType, get_peft_model, LoraConfig

def process_func(example):
    """ 数据处理流程
        训练的数据是需要经过格式化、编码之后再输入给模型进行训练的
        参考GLM3仓库:https://github.com/THUDM/ChatGLM3/blob/main/finetune_chatmodel_demo/preprocess_utils.py
    """
    MAX_LENGTH = 512
    input_ids, labels = [], []
    instruction = tokenizer.encode(text="\n".join(["<|system|>", "现在你要扮演原神开放世界中探险的向导--派蒙", "<|user|>", 
                                    example["instruction"] + example["input"] + "<|assistant|>"]).strip() + "\n",
                                    add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    response = tokenizer.encode(text=example["output"], add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    input_ids = instruction + response + [tokenizer.eos_token_id]
    labels = [tokenizer.pad_token_id] * len(instruction) + response + [tokenizer.eos_token_id]
    pad_len = MAX_LENGTH - len(input_ids)
    # print()
    input_ids += [tokenizer.pad_token_id] * pad_len
    labels += [tokenizer.pad_token_id] * pad_len
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def eval_test(model):
    """ 模型推理
    """
    model.eval()
    model = model.cuda()
    ipt = tokenizer("<|system|>\n现在你要扮演原神开放世界中探险的向导--派蒙\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
    tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

args = TrainingArguments(
    output_dir="output/ChatGLM",    # 模型的输出路径
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 梯度累加，如果显存比较小，那可以把 `batch_size` 设置小一点，梯度累加增大一些。
    logging_steps=20,               # 多少步，输出一次log
    num_train_epochs=1
)

if "__main__" == __name__:
    # 将JSON文件转换为CSV文件,处理数据集
    df = pd.read_json('dataset/paimeng.json')
    ds = Dataset.from_pandas(df)
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", 
                                              trust_remote_code=True)
    # 将数据集变化为token形式
    tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

    # 创建模型
    model = AutoModelForCausalLM.from_pretrained("model/chatglm3-6b",
                                                 torch_dtype=torch.half, 
                                                 trust_remote_code=True, 
                                                 low_cpu_mem_usage=True)

    # 创建loRA参数
    config = LoraConfig(task_type=TaskType.CAUSAL_LM,       # 模型类型
                        target_modules={"query_key_value"}, # 需要训练的模型层的名字，主要就是`attention`部分的层，不同的模型对应的层的名字不同，可以传入数组，也可以字符串，也可以正则表达式。
                        r=8,                                # lora 的秩
                        # 归一化超参数，lora参数ΔW会被以 alpha/r 归一化，以便减少改变r时需要重新训练的计算量
                        lora_alpha=32,                      # Lora alaph
                        lora_dropout=0.0                    # Dropout 比例
                        )

    # 模型合并
    model = get_peft_model(model, config)

    # 指定GLM的Data collator
    # Data collator GLM源仓库从新封装了自己的data_collator,在这里进行沿用。
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    # 指定训练参数。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()




