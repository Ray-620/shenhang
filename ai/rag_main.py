from langchain_rag import *
import torch
from loguru import logger as lg
import time
import os
import pandas as pd


# lg.info("start load model")
# embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
# model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
# embedding_model = load_hf_embedding_model(embedding_model_name,model_kwargs)
# lg.info("end load model")
#f3n_data_path = "C:\\Users\\admin\\Desktop\\shenhang\\knowledge\\test_f3n_data.csv"
# f3n_data_path = os.path.join(os.path.realpath('.'),'..','knowledge','test_f3n_data.csv')
# f3n_data = pd.read_csv(f3n_data_path)




#def run_create_langchain_faiss_vectorstore_main():
    ### generate create_langchain_faiss_vectorstore_main
    # embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
    # #embedding_model_name = "/Users/libo/.cache/torch/sentence_transformers/aspire_acge_text_embedding"
    # embedding_model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    # f3n_data_path = "C:\\Users\\admin\\Desktop\\shenhang\\knowledge\\test_f3n_data.csv"
    # #f3n_data_path = os.path.join(os.path.realpath('.'),'..','knowledge','f3n_data.csv')
    # do_you_have_an_indexed_meta_faiss = True
    # faiss_vectorstore_folder_path = "C:\\Users\\admin\\Desktop\\shenhang\\langchain_vectorstore"
    # faiss_vectorstore_file_name = "langchain_faiss_f3n_vstore"
    # meta_faiss_index_dir = "C:\\Users\\admin\\Desktop\\shenhang\\index"
    # meta_faiss_index_file_name = "test_acge_index"
    # create_langchain_faiss_vectorstore(embedding_model_name,embedding_model_kwargs,f3n_data_path,do_you_have_an_indexed_meta_faiss,meta_faiss_index_dir,meta_faiss_index_file_name,faiss_vectorstore_folder_path,faiss_vectorstore_file_name)

# initialize the bm25 retriever and faiss retriever
bm25_retriever = load_retriever(os.path.join("C:\\Users\\admin\\Desktop\\shenhang\\",'langchain_retriever','bm25_retriever'))
faiss_retriever = load_retriever(os.path.join("C:\\Users\\admin\\Desktop\\shenhang\\",'langchain_retriever','faiss_retriever'))
keyword_data = pd.read_csv(os.path.join("C:\\Users\\admin\\Desktop\\shenhang\\",'knowledge','73NG-34.csv'))
historical_data = pd.read_csv(os.path.join("C:\\Users\\admin\\Desktop\\shenhang\\",'knowledge','test_f3n_data.csv'))

@lg.catch
def customize_rerank(query:str)->pd.DataFrame:
    """ Rerank the faiss result base on using bm25 retrieve result. bm25 will match the keyword of 4bit chapter with input question. 
    Args: query: input query
    Returns: return faiss result sorted by bm25
    """
    lg.info("bm25 retrieve")
    bm25_retriever_result = bm25_retriever.invoke(query)
    bm25_retriever_result_row_ls = [doc.metadata["row"] for doc in bm25_retriever_result]
    bm25_retriever_keyword_data = keyword_data.loc[bm25_retriever_result_row_ls,:]  
    lg.info(bm25_retriever_keyword_data["4位章节号"])  
    lg.info("fasiss retrieve")
    faiss_retrieve_result = faiss_retriever.invoke(query)
    faiss_retriever_result_row_ls = [doc.metadata["row"] for doc in faiss_retrieve_result]
    faiss_retriever_historical_data = historical_data.loc[faiss_retriever_result_row_ls,:]
    lg.info("check if there're multiple 4bit measure num in bm25 result")
    if len(np.unique(bm25_retriever_keyword_data["4位章节号"]))!=bm25_retriever_keyword_data.shape[0]:
        lg.info("yes,there're multiple 4bit measure num")
        bm25_retriever_4bit_num_sort_values = bm25_retriever_keyword_data.value_counts(subset="4位章节号")
        bm25_retriever_keyword_4bit_num = bm25_retriever_4bit_num_sort_values.index.to_list()
    
        faiss_retriever_historical_data_in_keyword = faiss_retriever_historical_data[faiss_retriever_historical_data["4_bit_measure_num"].isin(bm25_retriever_keyword_4bit_num)].copy()
        faiss_retriever_historical_data_notin_keyword = faiss_retriever_historical_data[~faiss_retriever_historical_data["4_bit_measure_num"].isin(bm25_retriever_keyword_4bit_num)].copy()
        
        if not faiss_retriever_historical_data_in_keyword.empty:
            lg.info("4bit measure num also in faiss result")
        # Create a dictionary mapping integers to their positions in the custom order
            order_mapping = {val: i for i, val in enumerate(bm25_retriever_keyword_4bit_num)}

            # Create a temporary column for sorting based on the custom order
            faiss_retriever_historical_data_in_keyword.loc[:,'sort_temp'] = faiss_retriever_historical_data_in_keyword['4_bit_measure_num'].map(order_mapping).copy()

            # Sort the DataFrame by the temporary column
            faiss_retriever_historical_data_in_keyword_sorted = faiss_retriever_historical_data_in_keyword.sort_values('sort_temp').copy()
            faiss_retriever_historical_data_in_keyword_sorted.drop(columns=['sort_temp'], inplace=True)

            faiss_retriever_historical_data = pd.concat([faiss_retriever_historical_data_in_keyword_sorted,faiss_retriever_historical_data_notin_keyword])
    lg.info("customize_rerank done")
    return faiss_retriever_historical_data

def post_process(faiss_retriever_historical_data:pd.DataFrame,top_k:Optional[int]=5)->Dict[str,List[str]]:
    """Post process the result of sorted faiss result
    Args:
        faiss_retriever_historical_data: the result of sorted faiss result
        top_k: return topk
    Return:
    """
    final_result={}
    historical_question_ls = []
    historical_measure_num_ls = []
    historical_measure_ls = []
    n=0
    for idx,row_data in faiss_retriever_historical_data.iterrows():
        n+=1
        if n>top_k:
            break
        historical_question_ls.append(row_data["knowledge_question"])
        historical_measure_num_ls.append(row_data["measure_num"])
        historical_measure_ls.append(row_data["measure"])

    final_result["measure"] = historical_measure_ls
    final_result["historical_question"] = historical_question_ls        
    final_result["measure_num"] = historical_measure_num_ls
    return final_result
@lg.catch
def rag_main(query:str,topk=5)->Dict[str,List[str]]:
    start_time = time.time()
    lg.info(f"rag_main: {query}")

    lg.info("start answer question")
    faiss_retriever_historical_data = customize_rerank(query)
    lg.info("start post proces")
    final_result = post_process(faiss_retriever_historical_data,topk) 
    lg.info("end answer question")
    
    end_time = time.time()
    lg.info(f"rag_main costing time: {end_time-start_time}")
    return final_result

if __name__=="__main__":
    
    ###
    query="过站FLB反馈巡航8900m时左组件灯亮"
    top_k=2
    final_result = rag_main(query,top_k)
    # final_result = rag_main(query)
    print(final_result)
    
    
    # embedding_model_name = "C:\\Users\\admin\\.cache\\torch\\sentence_transformers\\aspire_acge_text_embedding"
    # model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    # source_column = "index"
    # rag_index = RagIndexer()
    # index_dir = os.path.join(os.path.realpath('.'),'..','..','index')
    # index_file_name = "test_acge_index"
    # knowledge_index = rag_index.deserialize_faiss_index(index_dir,index_file_name)
    # knowledge_docstore = None
    # knowledge_index_to_docstore_id = None
    # is_load_meta_faiss =False
    

    