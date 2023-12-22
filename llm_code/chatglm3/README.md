# ChatGLM3

### 安装

```shell
pip install -r requirements.txt
```

- `transformers` 库版本应该 `4.30.2` 以及以上的版本 ，`torch` 库版本应为 2.0 及以上的版本，以获得最佳的推理性能。
- 为了保证 `torch` 的版本正确，请严格按照 [官方文档](https://pytorch.org/get-started/locally/) 的说明安装。
- `gradio` 库版本应该为 `3.x` 的版本。

### 模型下载

使用 `modelscope` 中的`snapshot_download`函数下载模型，第一个参数为模型名称，参数`cache_dir`为模型的下载路径。

新建 `download_glm3.py` 文件并在其中输入以下内容，粘贴代码后记得保存文件，如下图所示。并运行 `python download_glm3.py`执行下载，模型大小为 14 GB，下载模型大概需要 10~20 分钟

```python
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('ZhipuAI/chatglm3-6b', cache_dir='model/chatglm3-6b', revision='master')
```

### 代码调用

可以通过如下代码调用 ChatGLM 模型来生成对话：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM3-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```

### Fast API 接口

实现了 OpenAI 格式的流式 API 部署，可以作为任意基于 ChatGPT 的应用的后端，比如 [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web)。可以通过运行仓库中的[openai_api.py](https://github.com/THUDM/ChatGLM3/blob/main/openai_api_demo/openai_api.py) 进行部署：

```shell
cd fast_api
python fast_api_chatglm3.py
```

同时，也书写了一个示例代码，用来测试API调用的性能。

- 使用Curl进行测试

```shell
curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
```

- 使用Python进行测试

```python
import requests
import json

def get_completion(prompt):
    headers = {'Content-Type': 'application/json'}
    data = {"prompt": prompt, "history": []}
    response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == '__main__':
    print(get_completion('你好'))
```

如果测试成功，则模型应该返回一段故事。



### OpenAI接口

实现了 OpenAI 格式的流式 API 部署，可以作为任意基于 ChatGPT 的应用的后端，比如 [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web)。可以通过运行仓库中的[openai_api.py](https://github.com/THUDM/ChatGLM3/blob/main/openai_api_demo/openai_api.py) 进行部署：

```shell
cd openai_api
python openai_api.py
```

同时，也书写了一个示例代码，用来测试API调用的性能。

- 使用Curl进行测试

```shell
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d "{\"model\": \"chatglm3-6b\", \"messages\": [{\"role\": \"system\", \"content\": \"You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\"}, {\"role\": \"user\", \"content\": \"你好，给我讲一个故事，大概100字\"}], \"stream\": false, \"max_tokens\": 100, \"temperature\": 0.8, \"top_p\": 0.8}"
```

- 使用Python进行测试

```
cd openai_api_demo
python openai_api_request.py
```

如果测试成功，则模型应该返回一段故事。

### 低成本部署

#### 模型量化

默认情况下，模型以 FP16 精度加载，运行上述代码需要大概 13GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型，使用方法如下：

```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
```

模型量化会带来一定的性能损失，经过测试，ChatGLM3-6B 在 4-bit 量化下仍然能够进行自然流畅的生成。

#### CPU 部署

如果没有 GPU 硬件的话，也可以在 CPU 上进行推理，但是推理速度会更慢。使用方法如下（需要大概 32GB 内存）

```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float()
```

#### Mac 部署

对于搭载了 Apple Silicon 或者 AMD GPU 的 Mac，可以使用 MPS 后端来在 GPU 上运行 ChatGLM3-6B。需要参考 Apple 的 [官方说明](https://developer.apple.com/metal/pytorch) 安装 PyTorch-Nightly（正确的版本号应该是2.x.x.dev2023xxxx，而不是 2.x.x）。

目前在 MacOS 上只支持[从本地加载模型](https://github.com/THUDM/ChatGLM3/blob/main/README.md#从本地加载模型)。将代码中的模型加载改为从本地加载，并使用 mps 后端：

```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).to('mps')
```

加载半精度的 ChatGLM3-6B 模型需要大概 13GB 内存。内存较小的机器（比如 16GB 内存的 MacBook Pro），在空余内存不足的情况下会使用硬盘上的虚拟内存，导致推理速度严重变慢。

#### 多卡部署

如果你有多张 GPU，但是每张 GPU 的显存大小都不足以容纳完整的模型，那么可以将模型切分在多张GPU上。首先安装 accelerate: `pip install accelerate`，然后即可正常加载模型。











