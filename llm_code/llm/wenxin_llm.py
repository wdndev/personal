# -*- coding: utf-8 -*-
#  @file        - wenxin_llm.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 基于百度文心大模型自定义 LLM 类
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

from llm.llm_base import LLMBase

from typing import Any, List, Mapping, Optional, Dict, Union, Tuple
from pydantic import Field
import json
import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun

class WenxinLLM(LLMBase):
    """ 文心大模型自定义LLM
    """
    # URL
    url : str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"
    # Secret_Key
    secret_key : str = None
    # access_token
    access_token: str = None

    def init_access_token(self):
        """ 初始化key
        """
        if self.api_key != None and self.secret_key != None:
            try:
                self.access_token = get_access_token(self.api_key, self.secret_key)
            except Exception as e:
                print(e)
                print("获取 access_token 失败，请检查 Key")
        else:
            print("API_Key 或 Secret_Key 为空，请检查 Key")

    def _call(self, 
              prompt : str, 
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        
        # 如果 access_token 为空，初始化 access_token
        if self.access_token == None:
            self.init_access_token()
        # API 调用 url
        url = self.url.format(self.access_token)
        # 配置 POST 参数
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",# user prompt
                    "content": "{}".format(prompt)# 输入的 prompt
                }
            ],
            'temperature' : self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }

        # 发起请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
        if response.status_code == 200:
            # 返回的是一个 Json 字符串
            js = json.loads(response.text)
            # print(js)
            return js["result"]
        else:
            return "请求失败"
    
    @property
    def _llm_type(self) -> str:
        return "Wenxin"


def get_access_token(api_key : str, secret_key : str):
    """ 调用文心API的工具函数
    """
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")
