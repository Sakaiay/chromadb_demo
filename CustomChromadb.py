import chromadb
import uuid
from chromadb.config import Settings
from typing import Union, List, Dict
from CustomEmbedding import EmbeddingModel

class ChromaDBManager:
    def __init__(self, 
                 embedding_model, 
                 persist_directory="./chroma_db"):
        '''
            初始化Chroma数据库客户端
        '''
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = embedding_model
    
    def get_collections_list(self) -> List[str]:
        """
            获取所有集合的名称
        """
        return [collection.name for collection in self.client.list_collections()]
    
    def get_collection(self, collection_name: str):
        """
            获取指定名称的集合
        """
        collections = {collection.name: collection for collection in self.client.list_collections()}
        return collections.get(collection_name)
    
    def create_collection(self, collection_name: str):
        """
            创建一个新的集合
        """
        if collection_name not in self.get_collections_list():
            self.client.create_collection(name=collection_name)
        return self.get_collection(collection_name)
    
    def delete_collection(self, collection_name: str):
        ''' 
           删除指定集合
        '''
        if collection_name not in self.get_collections_list():
            raise ValueError(f"Delete Error: Collection '{collection_name}' does not exist.")
        self.client.delete_collection(name=collection_name)
    
    def add_texts(self, 
                  collection_name: str, 
                  texts: List[str], 
                  metadatas: List[Dict]) -> List[Dict]:
        """
            添加文本及其嵌入向量到指定集合中，并返回生成的ID
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist. Please create it first.")

        doc_infos = []
        ids = [str(uuid.uuid1()) for _ in range(len(texts))]
        embeddings = self.embedding_model.generate_embeddings(texts)
        for _id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            collection.add(
                ids=_id,
                embeddings=embedding,
                metadatas=metadata,
                documents=text
            )
            doc_infos.append({"id": _id, "metadata": metadata})
        return doc_infos
    
    
    def get_collection_texts(self, collection_name: str) -> List[Dict]:
        """
            获得指定集合所有数据
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        results = collection.get()
        return [
            {
                "id": id_,
                "text": document,
                "metadata": metadata
            }
            for id_, document, metadata in zip(results["ids"], results["documents"], results["metadatas"])
        ]
    
    def delete_texts(self, 
                     collection_name: str, 
                     ids: List[str]):
        """
            从指定集合中删除指定ID的文本
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        collection.delete(ids=ids)
    
    def upsert_texts(self,
                     collection_name: str, 
                     texts: List[str], 
                     metadatas: List[Dict],
                     ids: List[str] = None,
                     ):
        ''' 
           更新指定集合中文本
           如果ID存在就更新存在的文本，如果ID不存在就插入新的文本
        '''
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        if not ids:
            ids = [str(uuid.uuid1()) for _ in range(len(texts))]
        doc_infos = []
        embeddings = self.embedding_model.generate_embeddings(texts)
        for _id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            collection.upsert(
                ids=_id,
                embeddings=embedding,
                metadatas=metadata,
                documents=text
            )
            doc_infos.append({"id": _id, "metadata": metadata})
        return doc_infos
    
    
    def query_texts(self, 
                    collection_name: str, 
                    query_text: str, 
                    n_results: int = 3) -> List[Dict]:
        """
            根据查询文本检索指定集合中的相似文本
        """
        collection = self.get_collection(collection_name)
        if not collection:
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        query_embedding = self.embedding_model.generate_embeddings(query_text)
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        return [
            {
                "id": id_,
                "text": document,
                "metadata": metadata
            }
            for id_, document, metadata in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])
        ]
        
        
if __name__ == '__main__':
    model_name = "bge-m3"
    model_path =  "/data01/tqbian/modelPATH/Xorbits/bge-m3"
    url = "http://127.0.0.1:9997/v1"
    api_key = "EMPTY"
    myembedding = EmbeddingModel(model_name=model_name, 
                   model_path=model_path, 
                   url=url, 
                   api_key=api_key)

    texts_sports = [
            "2011年7月20日，姚明正式宣布退役",
             "1998年4月，姚明入选王非执教的国家队，开始了职业篮球生涯。",
             "沙奎尔·奥尼尔（Shaquille O'Neal），全名沙奎尔·雷肖恩·奥尼尔（Shaquille Rashaun O'Neal），昵称沙克（Shaq）",
            "奥尼尔在1992年NBA选秀中于第1轮第1位以状元秀的身份被奥兰多魔术队选中"
    ]
    metadatas_sports = [{'domain': 'sports', 'source': idx} for idx in range(len(texts_sports))]


    texts_city = [
        "北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市",
        "2023年，北京市全年实现地区生产总值43760.7亿元，按不变价格计算，比上年增长5.2%",
        "纽约（英语：New York），位于美国纽约州南端（非纽约州首府），为美国人口最多的城市、纽约都会区的核心",
        "纽约位于美国东北部，滨临大西洋海岸，坐拥世界上最大天然港口之一的纽约和新泽西港"
    ]
    metadatas_city = [{'domain': 'city', 'source': idx} for idx in range(len(texts_city))]
    
    db_manager = ChromaDBManager(embedding_model=myembedding)
    
    print("==========创建sports集合==========")
    collection_sports = "sports"
    db_manager.create_collection(collection_sports)
    doc_infos_sports = db_manager.add_texts(collection_sports, 
                                  texts_sports, 
                                  metadatas_sports)
    print("==========创建city集合==========")
    collection_city = "city"
    db_manager.create_collection(collection_city)
    doc_infos_city = db_manager.add_texts(collection_city, 
                                  texts_city, 
                                  metadatas_city)
    
    print("==========打印所有集合==========")
    print(db_manager.get_collections_list())
    
    
    
    query = '奥尼尔选秀'
    print(f"查询测试，问题：{query}")
    
    query_res = db_manager.query_texts(
        collection_name='sports',
        query_text=query,
        n_results=1
    )
    print(f"查询结果：{query_res}")
    
    print("===========删除集合==========")
    db_manager.delete_collection(collection_name='sports')
    
    print(db_manager.get_collections_list())
    
    print("==========获得集合中所有元素==========")
    collection_results = db_manager.get_collection_texts(collection_name='city')
    for collection_result in collection_results:
        print(collection_result)
    
    print("==========删除指定集合中的文档==========")
    id_list = [collection_results[i]['id'] for i in range(2)]
    db_manager.delete_texts('city', id_list)

    collection_results = db_manager.get_collection_texts(collection_name='city')
    for collection_result in collection_results:
        print(collection_result)
        
    print("==========修改集合中的文本==========")
    texts_city2 = [
    "上海市地处长江三角洲冲积平原，地势坦荡低平， 属亚热带季风性气候，最大河流为黄浦江。",
    "2024年，上海市实现地区生产总值53926.71亿元，按不变价格计算，比上年增长5.0%。",
    "上海市是中国国际经济、金融、贸易、航运、科技创新中心，第三产业为其支柱产业，有着外贸物流、金融保险业、信息服务业、旅游业、房地产业和其他新兴服务业， 成为拉动经济增长“主动力”"
    ]
    metadatas_city2 = [{'domain': 'city2', 'source': idx} for idx in range(len(texts_city2))]
    db_manager.upsert_texts(collection_name='city', texts=texts_city2, metadatas=metadatas_city2)
    collection_results = db_manager.get_collection_texts(collection_name='city')
    for collection_result in collection_results:
        print(collection_result)
        
    ids = [collection_results[i]['id'] for i in range(2)]
    texts_city3 = ['北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市',
                   '2023年，北京市全年实现地区生产总值43760.7亿元，按不变价格计算，比上年增长5.2%']
    metadatas_city3 = [{'domain': 'city', 'source': 2}, {'domain': 'city', 'source': 3}]

    db_manager.upsert_texts(collection_name='city', texts=texts_city3, metadatas=metadatas_city3, ids=ids)
    
    collection_results = db_manager.get_collection_texts(collection_name='city')
    for collection_result in collection_results:
        print(collection_result)