from modelscope import snapshot_download

# 默认模型会下载到~/.cache/modelscope/hub中
model_dir = snapshot_download(model_id="ZhipuAI/chatglm3-6b", 
                              cache_dir='model/chatglm3-6b', 
                              revision="v1.0.0")