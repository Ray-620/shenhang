import streamlit as st
import pickle
from typing import (Any)
from rag_main import test_ragindexer
import json
from zhipuai import ZhipuAI
import os
import pandas as pd

def load_retriever(file_path:str)->Any:
    """load_retriever
    Args:
        retriver:
        file_path:
    Returns:
        retriever
    """
    with open(file_path,'rb') as f:
        retriever = pickle.load(f)
    return retriever

# with st.sidebar:
#     openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
#     "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.subheader("💬 深航RAG增强大模型智能维修决策支持系统")
st.caption("🚀 Shenzhen Airlines AI maintenance decision system powered by RAG-LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "请输入您遇到的故障问题"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


ensemble_retriver_path = "C:\\Users\\admin\\Desktop\\shenhang\\langchain_retriever\\ensemble_retriever.pkl"#os.path.join(os.path.realpath('.'),'..','langchain_retriever','ensemble_retriever.pkl')
# save_retriever(ensemble_retriever,ensemble_retriver_path)
ensemble_retriever = load_retriever(ensemble_retriver_path)

with open('output.json', 'r') as file:
    handbook = json.load(file)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    response = rag_main(ensemble_retriever,prompt)
    msg = response
    print(msg)
    
    
    task = "现有手册中无法找到该任务"
    question = prompt
    history = str(msg['measure'])
    measure_num = []
    for i in range(len(msg['measure'])):
        if str(msg['measure_num'][i]) in handbook:
            measure_num = str(msg['measure_num'][i])
            task = str(handbook[measure_num])
            break

    output_area = st.empty()
    
    client = ZhipuAI(api_key="80455c76392763b39b00bd3bb978f153.CLwhcgyIPKXPBLAE") # 填写您自己的APIKey
    content = "根据我给出的内容返回一个详细检修推荐的中文步骤。必须使用提供的上下文来回答，不许自己给出解释!!!如果你不知道答案，就说你不知道，不要试图编造答案。给出的内容包括几个相似问题的历史检修操作，和检修手册中检修操作的上下文或只有历史检修操作。其中手册的内容上下文中可能有多个Task，或者一个Task对应多个SUBTASK+对应的编号，你需要自己去识别对应的Task对应的内容。请从上下文中选择相关的文本来回答我的输入问题，不相关的内容请忽略。prompt =开始吧!我的输入:{"+ str(question) +"}，历史检修操作：{"+ str(history) +"}给出的上下文:{"+ str(task) +"}若有手册内容，请以以下格式回复您的回复:助理回答:Task-编号:对应的全部详细操作步骤。若没有手册内容，只需要根据历史检修记录整理出具体详细的操作步骤，不要提到我是否提供手册内容。"
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[
            {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {
                "role": "user", 
                "content": content
            },
        ],
        stream=True,
    )
    
    text = '##### 检修建议\n\n'
    for chunk in response:
        text = text + chunk.choices[0].delta.content
        output_area.chat_message("assistant").write(text)

    print(text)

    st.session_state.messages.append({"role": "assistant", "content": text})
    # st.chat_message("assistant").write(text)