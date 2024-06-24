from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community import docstore
from langchain_core.vectorstores import VectorStoreRetriever
import faiss
import jieba
import uuid
import os
import pickle
import numpy as np
import time
from faiss_indexer import deserialize_faiss_index
from loguru import logger as lg
from typing import (Any,List,Dict,Optional,TypeVar)

#VST = TypeVar("VST", bound="VectorStore")

def load_hf_embedding_model(embedding_model_name:str, model_kwargs:Optional[str] = {'device': 'cpu'})->Any:
    """load hugging face model into langchain

    Args:
        embedding_model_name (str): _description_
        model_kwargs (_type_, optional): _description_. Defaults to {'device': 'cpu'}.

    Returns:
        Any: _description_
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    return embedding_model



def load_csv_into_langchain(data_path:str,encoding_type:Optional[str] = "utf-8",source_column:Optional[str] = None)->List[Document]:
    """load csv into langchain

    Args:
        data_path (str): _description_
        encoding_type (Optional[str], optional): _description_. Defaults to "utf-8".
        source_column (Optional[str], optional): _description_. Defaults to None.

    Returns:
        List[Document]: _description_
    """
    custom_loader = CSVLoader(data_path,encoding=encoding_type,source_column=source_column)
    custom_docs = custom_loader.load()
    return custom_docs

def initialize_bm25_retriever(custom_docs:List[Document],bm25_retriever_topk:int = 5):
    """initialize the bm25 retriever and faiss retriever
    Args:
        custom_docs List[Document]: _description_
        bm25_retriever_topk (int): _description_. Defaults to 5.

    Returns:
        VectorStoreRetriever: _description_
    """
    
    bm25_retriever = BM25Retriever.from_documents(
        custom_docs,preprocess_func=preprocessing_func
    )
    bm25_retriever.k = bm25_retriever_topk
    return bm25_retriever

def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))

def create_customize_docstore(custom_docs:List[Document],customize_index_to_docstore_id:Dict[int,str])->docstore:
    """create a customize docstore

    Returns:
        _type_: _description_
    """
    customize_docstore = InMemoryDocstore(
    {customize_index_to_docstore_id[i]: doc for i, doc in enumerate(custom_docs)}
)
    return customize_docstore

def create_customize_index_to_docstore_id(custom_docs:List[Document])->Dict[int,str]:
    """create a customize index to docstore id

    Args:
        custom_docstore (docstore): _description_

    Returns:
        _type_: _description_
    """
    customize_index_to_docstore_id = {i: str(uuid.uuid4()) for i in range(len(custom_docs))}
    return customize_index_to_docstore_id

def load_meta_faiss_into_langchain(
    langchain_embedding:Embeddings,
    meta_faiss_index:faiss,
    customize_docstore:docstore,
    customize_index_to_docstore_id:Dict[int,str],
    normalize_L2:Optional[bool]=True,
    customized_distance_strategy:Optional[any]=DistanceStrategy.DOT_PRODUCT 
    )->FAISS:
    """load your trained meta faiss into langchain

    Args:
        langchain_embedding (Embeddings): _description_
        meta_faiss_index (faiss): _description_
        customize_docstore (docstore): _description_
        customize_index_to_docstore_id (_type_): _description_
        normalize_L2 (Optional[bool], optional): _description_. Defaults to True.
        customized_distance_strategy (Optional[any], optional): _description_. Defaults to DistanceStrategy.DOT_PRODUCT.

    Returns:
        FAISS: _description_
    """
    #langchain_faiss = FAISS(langchain_embedding,index=meta_faiss_index,docstore=customize_docstore,index_to_docstore_id=customize_index_to_docstore_id,normalize_L2=normalize_L2,distance_strategy=customized_distance_strategy)
    langchain_faiss = FAISS(langchain_embedding,index=meta_faiss_index,docstore=customize_docstore,index_to_docstore_id=customize_index_to_docstore_id,normalize_L2=True,distance_strategy=DistanceStrategy.DOT_PRODUCT)
    return langchain_faiss

def create_langchain_faiss_vectorstore_main(embedding_model_name,embedding_model_kwargs,data_path,do_you_have_an_indexed_meta_faiss,index_dir,index_file_name,faiss_vectorstore_folder_path,faiss_vectorstore_file_name,source_column:Optional[str]=None):
    
    start_time = time.time()
    lg.info("start load model")

    embedding_model = load_hf_embedding_model(embedding_model_name,embedding_model_kwargs)
    lg.info("end load model")
    
    lg.info("start load data")
    custom_docs = load_csv_into_langchain(data_path,'utf-8',source_column)
    lg.info("end load data")
    
    lg.info("start create langchain faiss")
    if do_you_have_an_indexed_meta_faiss:
        knowledge_index = deserialize_faiss_index(index_dir,index_file_name)
        knowledge_index_to_docstore_id = create_customize_index_to_docstore_id(custom_docs)
        knowledge_docstore = create_customize_docstore(custom_docs,knowledge_index_to_docstore_id)
        faiss_vectorstore = load_meta_faiss_into_langchain(embedding_model,knowledge_index,customize_docstore=knowledge_docstore,customize_index_to_docstore_id=knowledge_index_to_docstore_id)
    else: 
        faiss_vectorstore = FAISS.from_documents(
        custom_docs, embedding_model
    )
    lg.info("save_faiss_vectorstore")
    save_langchain_faiss_vectorstore(faiss_vectorstore,faiss_vectorstore_folder_path,faiss_vectorstore_file_name)
    lg.info("end save_faiss_vectorstore")
    end_time = time.time()
    lg.info(f"create_langchain_faiss_main costing time: {end_time-start_time}")
    return faiss_vectorstore

def langchain_faiss_vectorstore_query_with_score(query:str,embedding_model:Embeddings,langchain_faiss_vectorstore:FAISS):
    query_embeded_vector = embedding_model.embed_query(query)
    query_embeded_vector = np.array(query_embeded_vector).reshape(1,1792)
    query_embeded_vector = query_embeded_vector/np.linalg.norm(query_embeded_vector,axis=1)[:,None]
    query_embeded_vector = query_embeded_vector[0].tolist()
    return langchain_faiss_vectorstore.similarity_search_with_score_by_vector(query_embeded_vector)

def faiss_as_retrieve(langchain_faiss:FAISS,search_kwargs:Optional[Dict[str,str]]=None)->VectorStoreRetriever:
    """convert langchain vector store as the retreiver

    Args:
        langchain_faiss (FAISS): _description_
        search_kwargs (_type_, optional): _description_. Defaults to None.

    Returns:
        VectorStoreRetriever: _description_
    """
    faiss_retriever = langchain_faiss.as_retriever(search_kwargs=search_kwargs)
    return faiss_retriever

def init_ensemble_retriever(bm25_retriever:BM25Retriever,faiss_retriever:VectorStoreRetriever,weights:List[float] = [0.5,0.5])->EnsembleRetriever:
    """init the ensemble retriever

    Args:
        bm25_retriever (BM25Retriever): _description_
        faiss_retriever (FAISS): _description_

    Returns:
        EnsembleRetriever: _description_
    """
    ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)
    return ensemble_retriever

def save_langchain_faiss_vectorstore(langchain_faiss:FAISS,folder_path: str, index_name: str = "index") -> None:
    """save_langchain_faiss_vectorstore

    Args:
        langchain_faiss (FAISS): _description_
        folder_path (str): _description_
        index_name (str, optional): _description_. Defaults to "index".

    Returns:
        _type_: _description_
    """
    return langchain_faiss.save_local(folder_path,index_name)

def load_langchain_faiss_vectorstore(
        embeddings: Embeddings,
        folder_path: str,
        index_name: str = "index",
        allow_dangerous_deserialization: bool = False,)-> FAISS:
    """load_langchain_faiss_vectorstore

    Args:
        embeddings (Embeddings): _description_
        folder_path (str): _description_
        index_name (str, optional): _description_. Defaults to "index".
        allow_dangerous_deserialization (bool, optional): _description_. Defaults to False.

    Returns:
        FAISS: _description_
    """
    langchain_faiss_vectorstore = FAISS.load_local(folder_path,embeddings,index_name=index_name,allow_dangerous_deserialization=allow_dangerous_deserialization)
    return langchain_faiss_vectorstore

def save_retriever(retriver,file_path:str)->None:
    """save_retriever

    Args:
        retriver:
        file_path:
    Returns:
        None
    """
    return pickle.dump(retriver,open(file_path,'wb'))

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

