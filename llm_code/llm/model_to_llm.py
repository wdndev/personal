# -*- coding: utf-8 -*-
#  @file        - model_to_llm.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 将langchain封装的LLM打包
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

import sys 
sys.path.append("..")
from llm.wenxin_llm import WenxinLLM
from llm.spark_llm import SparkLLM
from llm.zhipuai_llm import ZhipuAILLM
from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key


def model_to_llm(model:str=None, 
                 temperature:float=0.0, 
                 appid:str=None, 
                 api_key:str=None,
                 Spark_api_secret:str=None,
                 Wenxin_secret_key:str=None):
        """
        星火：model,temperature,appid,api_key,api_secret
        百度问心：model,temperature,api_key,api_secret
        智谱：model,temperature,api_key
        OpenAI：model,temperature,api_key
        """
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
            if api_key == None:
                api_key = parse_llm_api_key("openai")
            llm = ChatOpenAI(model_name = model, temperature = temperature , openai_api_key = api_key)
        elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
            if api_key == None or Wenxin_secret_key == None:
                api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
            llm = WenxinLLM(model=model, temperature = temperature, api_key=api_key, secret_key=Wenxin_secret_key)
        elif model in ["Spark-1.5", "Spark-2.0"]:
            if api_key == None or appid == None and Spark_api_secret == None:
                api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
            llm = SparkLLM(model=model, temperature = temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
        elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
            if api_key == None:
                api_key = parse_llm_api_key("zhipuai")
            llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature = temperature)
        else:
            raise ValueError(f"model{model} not support!!!")
        return llm