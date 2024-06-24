import os,json,re
import time
from datetime import datetime
import threading
import flask
from flask import request, jsonify, current_app, Response
from loguru import logger as lg
from ai import ai_blue
from ai.llm_api import LlmApi
from ai.faiss_indexer import RagIndexer
# Use a dictionary to store API usage statistics
stats = {}
# Use thread locks to ensure the safety of dictionary operations under multi-threads
stats_lock = threading.Lock()

def record_stats(api_path, date_str):
    with stats_lock:
        if api_path not in stats:
            stats[api_path] = {}
        if date_str not in stats[api_path]:
            stats[api_path][date_str] = 0
        stats[api_path][date_str] += 1


@ai_blue.before_request
def before_request_func():
    # Get the API path and date of the current request
    api_path = request.path
    if api_path == '/favicon.ico':
        return
    date_str = datetime.now().strftime('%Y-%m-%d')
    # Update statistics
    record_stats(api_path, date_str)


@ai_blue.route('/stats', methods=['GET'])
def get_stats():
    # Provide an API to query statistics
    return jsonify(stats)

@ai_blue.route('/perplexityai',methods=["Post"])
def perplexityai():
    try:
        start = time.time()
        an_api_key = "PERPLEXITY_API_KEY"
        a_base_url = "https://api.perplexity.ai"
        a_model="sonar-medium-chat"
        perplexity_api = LlmApi(base_url=a_base_url,api_key=an_api_key)
        query = str(request.json["query"])
        lg.info("query is %s"%query)
        lg.info("type of query is %s"%type(query))
        perplexit_response = perplexity_api.search(prompt=query,model = a_model)
        perplexit_response = perplexit_response.replace("\n","")
        end = time.time()
        lg.info("Total costing time is %s"%(end-start))
        return jsonify({'code':0, 'msg': 'success', 'item': {"perplexit_response":perplexit_response}})
    except Exception as e:
        flask.abort(500)

@ai_blue.route('/knowledge_retrieve',methods=["Post"])
def knowledge_retrieve():
    try:
        query = str(request.json["query"])
        knowledge_data = current_app.knowledge_data
        knowledge_index = current_app.knowledge_index
        lg.info("knowledge shape is %s"%knowledge_data.shape[0])
        rag = RagIndexer()
        retrieve_result = rag.knowledge_retrieve(query,knowledge_index,knowledge_data,top_k_hits=5)
        return jsonify({'code':0, 'msg': 'success', 'item': {"knowledge_retrieve":retrieve_result}})
    except Exception as e:
        flask.abort(500)


# @ai_blue.route('/generative_answer',methods=["Post"])
# def knowledge_retrieve():
#     try:
#         query = str(request.json["query"])
#         knowledge_data = current_app.knowledge_data
#         knowledge_index = current_app.knowledge_index
#         lg.info("knowledge shape is %s"%knowledge_data.shape[0])
#         normalize_re_result = answer_generative_answer(query,knowledge_index,knowledge_data,top_k_hits=5)
#         return jsonify({'code':0, 'msg': 'success', 'item': {"generative_response":normalize_re_result}})
#     except Exception as e:
#         flask.abort(500)
