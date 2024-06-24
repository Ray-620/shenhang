from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from loguru import logger as lg


class LlmApi():
    def __init__(self,base_url=None,api_key="openai_api_key"):
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), '.env')
        load_dotenv(dotenv_path=env_path, verbose=True)
        self.api_key = os.getenv(api_key)
        self.client = OpenAI(api_key=self.api_key,base_url=base_url)


    def search(self, prompt="",model="gpt-4"):
        try:
            openai_response = self.client.chat.completions.create(
                model = model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                frequency_penalty=0.01,
                )
            result = (openai_response.choices[0].message.content)

            return result.lstrip()
        except Exception as e:
            lg.exception("An exception occurred in function '{}': {}".format("PromptSearchV2", str(e)))
            return None
    
    def openai_embedding(self,sentence,embedding_model="text-embedding-ada-002"):
        return self.client.embeddings.create(input=[sentence], model=embedding_model).data[0].embedding
        