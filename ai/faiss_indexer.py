from langchain import PromptTemplate
from langchain.output_parsers import RegexParser
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from sentence_transformers import SentenceTransformer
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from loguru import logger as lg
import re,os
import faiss
import time
import math
import numpy as np
from typing import Optional,List, Dict,Any

class RagIndexer():
    
    def load_model(self, model_name:Optional[str]="distiluse-base-multilingual-cased-v2"):
        """
        Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

        :param model_name: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
            default: "distiluse-base-multilingual-cased-v2"
            - https://www.sbert.net/docs/pretrained_models.html
        :return: SentenceTransformer
        """
        lg.debug("in load_model")
        model = SentenceTransformer(model_name)
        self.model = model
        lg.debug("success load_model")
        return model
    
    def serialize_faiss_index(self, index, directory_path:str, file_name:str)->bool:
        """
        Serialize a Faiss index and write it to Amazon EFS.

        :param index: Faiss index object.
        :param directory_path: the directory path where save index file.
        :param file_name: Name of the file to write.
        :return: True if writing is successful, False if writing fails.
        """
        try:
            # Specify the file path to write to
            file_path = os.path.join(directory_path, file_name)
            faiss.write_index(index, file_path)
            lg.info(f"Faiss index successfully written to {file_path}")
            return True
        except Exception as e:
            lg.info(f"Error writing Faiss index: {str(e)}")
            return False

    def deserialize_faiss_index(self, directory_path:str, file_name:str):
        """
        Read a Faiss index and deserialize it.

        :param directory_path:  the directory path where save index file.
        :param file_name: Name of the file to read.
        :return: Faiss index object or None if reading fails.
        """
        try:
            # Specify the file path to read from
            file_path = os.path.join(directory_path, file_name)
            # Deserialize the Faiss index
            index = faiss.read_index(file_path)
            return index
        except Exception as e:
            print(f"Error reading Faiss index: {str(e)}")
            return None

    def check_dataframe_valid(self,df:pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise Exception('df should be pandas.DataFrame')
        return

    def create_faiss_index(self,data:pd.DataFrame, embedding_size:int=512, n_clusters:int=10, nprobe:int=5):
        """create faiss index for input data

        Args:
            data (pd.DataFrame): _description_
            embedding_size (int, optional): _description_. Defaults to 512.
            n_clusters (int, optional): _description_. Defaults to 10.
            nprobe (int, optional): _description_. Defaults to 5.

        Returns:
            _type_: _description_
        """
        start_time = time.time()
        self.check_dataframe_valid(data)
        n_clusters = int(math.sqrt(len(data)))
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        lg.info("Encode the corpus. This might take a while")
        corpus_sentences = list()
        for i, row in data.iterrows():
            corpus_sentences.append(row['knowledge_question'])
        corpus_sentences = list(corpus_sentences)
        corpus_embeddings = self.model.encode(corpus_sentences, convert_to_numpy=True)
        embs = [np.array(i) for i in corpus_embeddings]
        lg.info("finish encoding for knowledge_question, time cost "+str(time.time() - start_time))
        # Create the FAISS index
        lg.info("Start creating FAISS index")
        # First, we need to normalize vectors to unit length
        try:
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
            # Then we train the index to find a suitable clustering
            index.train(corpus_embeddings)
            # Finally we add all embeddings to the index
            index.add(corpus_embeddings)
        except Exception as e:
            lg.debug(e)
        lg.info("finish creating FAISS index, time cost "+str(time.time() - start_time))
        return index,embs
    
    def knowledge_retrieve(self,question,knowledge_index,knowledge_data,top_k_hits=5,embedding_size=384):
        """Get topk answer from knowledge base

        Args:
            question (_type_): _description_
            knowledge_index (_type_): _description_
            knowledge_data (_type_): _description_
            top_k_hits (int, optional): _description_. Defaults to 5.
            embedding_size (int, optional): _description_. Defaults to 384.

        Returns:
            _type_: _description_
        """
        question_embedding = self.model.encode([question], convert_to_numpy=True)
        question_embedding = np.array(question_embedding).reshape(1,embedding_size)
        if knowledge_index is None:
            lg.info("There's no knowledge_index")
            return None
        distances, corpus_ids = knowledge_index.search(question_embedding, top_k_hits)
        response = []
        for i in range(len(corpus_ids)):
            hits = [{'corpus_id': corpus_id, 'score': score} for corpus_id, score in zip(corpus_ids[i], distances[i])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for hit in hits[0:top_k_hits]:
                row = knowledge_data.iloc[hit['corpus_id'],]
                index = str(row["index"])
                DE_num = row["DE Number"]
                knowledge_question = str(row["knowledge_question"])
                plan_measure = str(row.plan_measure)
                measure = str(row.measure)
                sim = int(hit['score'] * 1000) / 10.0
                response_dict = {'question': question,
                                    'index':index,
                                    'DE Number':DE_num,
                                    'knowledge_question': knowledge_question,
                                    'plan_measure': plan_measure,
                                    'measure': measure,
                                    'score': sim,
                                    'algorithm': 'FAISS-KnowledgeQuestion'
                                    }
                response.append(response_dict)
        final_response = {"FAISS-KnowledgeQuestion":response}
        return final_response

    def answer_generative_answer(self,question,knowledge_index,knowledge_data,top_k_hits=5,embedding_size=512,temperature=0,refuse_statement = "Unfortunately, I don't have the information to answer that question at the moment. However, I want to ensure you receive the best possible support. May I suggest contacting our support department?  They should be able to provide more detailed assistance. Used Control: None"):
        retrieval_results = self.knowledge_retrieve(question,knowledge_index,knowledge_data,top_k_hits,embedding_size)
        lg.info("length of retrieval_results is %s"%len(retrieval_results))
        if len(retrieval_results)==0:
            return None
        instruction = """
        You are an AI assistant designed to provide helpful, respectful, and honest responses. Your answers must adhere to ethical standards and avoid any harmful, unethical, or discriminatory content. Please ensure all responses are socially unbiased and positive in nature. 
        Suppose you are a company now and you need to answer user-entered questions in the first person, based on the knowledge you have.
        If a question is unclear or lacks coherence, kindly explain the issue instead of providing an incorrect response. If you are unsure of an answer, refrain from guessing.
        The knowledge format is json format. Each element in it is a knowledge, the key is knowledge name and value is knowledge text. 
        Your current {topk} knowledge areas related the questions are listed below:
        {knowledge_dict}
        Response Format Requirements:
        When responding to a question, begin by analyzing the input question and then provide a final answer. The final answer should be a conclusive statement without using uncertain terms like "probably" or "maybe."
        Offer a comprehensive and well-reasoned response that addresses the core query, supported by logical reasoning or evidence where applicable in the final answer.
        I hope you not only answer Yes or No, but you also need to briefly answer why you do this in the final answer.
        For questions requiring additional information, decide if it's necessary to acquire more knowledge. If confident in providing a response, answer with "Yes" along with your reasoning in the first person. 
        If you are unable to answer the question for security reasons, please briefly explain why and Used ID is None, Otherwise go straight back in Final Answer: {refuse_statement}
        Don't repeat the question in the final answer.
        Avoid words like "probably" or "maybe" in the final answer.
        Refrain from using phrases like "based on " in final answer, please convert it in the first person.
        Do not include statements like "there is no specific ability mentioned" in the final answer.
        Please use we to replace the company in final answer.
        Please list the konwledge name you used in composing your final answer in the Used ID.
        Always remember to answer in the first person!!!!!
        Practical Input Question:
        QUESTION: {question}
        Please adhere to the following format for your response:
        Your Analysis:
        Final Answer:
        Used ID:
        """
        output_parser = RegexParser(
            regex=r"Your Analysis:([\s\S]*)Final Answer:([\s\S]*)",
            output_keys=["Analysis","Final Answer"],
        )
        prompt = PromptTemplate(input_variables=["question","topk","knowledge_dict","refuse_statement"],template=instruction,output_parser=output_parser)
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '.env')
        load_dotenv(dotenv_path=env_path, verbose=True)
        openai_api_key = os.getenv("openai_api_key")
        llm=ChatOpenAI(model_name = "gpt-4",temperature=temperature,api_key=openai_api_key)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            verbose=False,
        )
        #topk_retrieval_dict = {{"knowledge_name":r["control"],"knowledge_text":r["answer"]} for r in retrieval_results}
        topk_retrieval_dict = {r["id"]:r["indexed_row_as_text"] for r in retrieval_results}
        lg.info("topk_retrieval_dict is %s"%topk_retrieval_dict)
        topk = len(topk_retrieval_dict)
        topk_retrieval_content = "\n\n"+str(topk_retrieval_dict)+"\n\n"
        final_outputs = llm_chain.invoke({"question":question,"topk":topk,"knowledge_dict":topk_retrieval_content,"refuse_statement":refuse_statement})["text"]
        re_result = re.findall(output_parser.regex,final_outputs)
        lg.info("re_result is %s"%re_result)
        normalize_re_result = {}
        if len(re_result)>0 and len(re_result[0])==2:
            for i,r in enumerate(re_result[0]):
                normalize_re_result[output_parser.output_keys[i]] = " ".join(r.split())
            normalize_re_result["Retrieval_Results"]=retrieval_results
            normalize_re_result["Response"] = final_outputs
            normalize_re_result["Question"] = question
            normalize_re_result["Knowledge_Info"] = topk_retrieval_dict
        else:
            normalize_re_result = {"Question":question, "Analysis":"None","Final Answer":"None","Retrieval_Results":retrieval_results,"Knowledge_Info":topk_retrieval_dict,"Response":final_outputs}
        #lg.info("normalize_re_result is %s"%normalize_re_result)
        return normalize_re_result