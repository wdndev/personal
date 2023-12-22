from modelscope import snapshot_download
# from modelscope
model_dir = snapshot_download("ZhipuAI/chatglm3-6b", cache_dir='model/chatglm3-6b', revision="v1.0.0")