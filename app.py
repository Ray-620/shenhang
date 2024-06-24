from flask import Flask, request, jsonify
from flask_cors import CORS
from config import *
from ai import ai_blue
import os
import pandas as pd
from loguru import logger as lg
from ai.rag_indexer import RagIndexer
#from ai import Similar_Match




def create_app():
    app = Flask(__name__)
    CORS(app)
    knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),"knowledge")
    lg.info("abs file path is %s"%(os.path.realpath(__file__)))
    lg.info("lg dir is %s"%knowledge_dir)
    knowledge_path = os.path.join(knowledge_dir,'knowledge_data.xlsx')
    knowledge_data = pd.read_excel(knowledge_path)
    rag = RagIndexer()
    knowledge_index = rag.deserialize_faiss_index(knowledge_data)
    app.knowledge_data = knowledge_data
    app.knowledge_index = knowledge_index
    #register the ai route
    app.register_blueprint(ai_blue)
    if os.getenv("APP_ENV"):
        app.config.from_object(ProductionConfig)
    # #load model
    # similar_match = Similar_Match()
    # similar_match.load_model()
    # app.similar_match = similar_match
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=3556, debug=True)
