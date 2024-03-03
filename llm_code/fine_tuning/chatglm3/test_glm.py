
from transformers import AutoTokenizer, AutoModel
model_dir = "model/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
# model = AutoModel.from_pretrained(model_dir,trust_remote_code=True).quantize(4).cuda()
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float()
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
