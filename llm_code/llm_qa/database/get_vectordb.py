# -*- coding: utf-8 -*-
#  @file        - get_vectordb.py
#  @author      - dongnian.wang(dongnian.wang@outlook.com)
#  @brief       - 获取向量数据库
#  @version     - 0.0
#  @date        - 2023.12.20
#  @copyright   - Copyright (c) 2023 

import sys 
# sys.path.append("../embedding") 
# sys.path.append("../database") 

from langchain.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型
import os
from embedding.zhipuai_embedding import ZhipuAIEmbeddings

from database.create_db import create_db,load_knowledge_db
from embedding.llm_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "openai",embedding_key:str=None):
    """
    返回向量数据库对象
    输入参数：
    question：
    llm:
    vectordb:向量数据库(必要参数),一个对象
    template：提示模版（可选参数）可以自己设计一个提示模版，也有默认使用的
    embedding：可以使用zhipuai等embedding，不输入该参数则默认使用 openai embedding，注意此时api_key不要输错
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb

