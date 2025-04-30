import os
import openai
from typing import Union, List

class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        model_path: str, 
        url: str = None,
        api_key: str = None
    ):
        if url or api_key:
            # API模式
            if not all([url, api_key]):
                raise ValueError("API mode need url and api_key")
            
            self.mode = "api"
            self.client = openai.Client(base_url=url, api_key=api_key)
            self.model_name = model_name
            
        else:
            # 本地模式
            self.model_path = model_path
            self.mode = "local"
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("install sentence-transformers：pip install sentence-transformers")
            
            self.model = SentenceTransformer(self.model_path)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings_batch(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if self.mode == "api":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        else:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        
    def generate_embeddings(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
            
        BATCH_SIZE = 1000
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            batch_embeddings = self.generate_embeddings_batch(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '7'
    model_name = "bge-m3"
    model_path =  "/data01/tqbian/modelPATH/Xorbits/bge-m3"
    # url = ""
    # api_key = ""
    url = "http://127.0.0.1:9997/v1"
    api_key = "EMPTY"
    myembedding = EmbeddingModel(model_name=model_name, model_path=model_path, url=url, api_key=api_key)
    query = ['btq'] * 2000

    res = myembedding.generate_embeddings(query)
    
    print(len(res))
    
    