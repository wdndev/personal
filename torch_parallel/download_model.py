#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('dienstag/chinese-bert-wwm-ext', cache_dir='./model/')