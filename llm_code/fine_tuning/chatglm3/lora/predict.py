# 通过PEFT所微调的模型，都可以使用下面的方法进行重新加载，并推理:

# - 加载源model与tokenizer；
# - 使用`PeftModel`合并源model与PEFT微调后的参数。

from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("model/chatglm3-6b", trust_remote_code=True, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("model/chatglm3-6b", use_fast=False, trust_remote_code=True)

p_model = PeftModel.from_pretrained(model, model_id="output/ChatGLM/checkpoint-1000/")  # 将训练所得的LoRa权重加载起来

ipt = tokenizer("<|system|>\n现在你要扮演原神开放世界中探险的向导--派蒙\n<|user|>\n {}\n{}".format("你是谁？", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
tokenizer.decode(p_model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)