# Chroma简单介绍和教程
## 介绍

Chroma 是一款**AI 原生开源的向量数据库**，专为机器学习和大语言模型（LLM）应用设计，旨在高效存储、检索和匹配高维向量数据

## 安装

使用 pip 安装 Chroma

```Bash
pip install chromadb  
```

## 基础使用命令

在**Chroma**中，**集合（Collection）** 是核心存储单元，用于管理文档、向量及元数据。每个集合（Collection）中存储的数据格式如下图所示：



![](./img/image1.png)



**`documents`** **（文档内容）：** 存储原始文本或非结构化数据（如段落、句子等）。

**`embeddings`** **（嵌入向量）：** 文档内容的高维向量表示，用于支持基于相似度的检索（如余弦相似度）。Chroma 内置多种嵌入模型（如 HuggingFace、OpenAI），也可自定义嵌入函数。

**`metadatas`** **（元数据）：** 附加的结构化信息，例如文档来源、页码、时间戳等。这些信息由用户手动添加。

**`ids`** **（唯一标识符）：** 每个文档的唯一 ID，需满足命名规则：长度 3-63 字符，以小写字母或数字开头结尾，中间可包含连字符（如`"doc-001"`）。



### 创建持久客户端

Chroma 支持持久化存储，通过`PersistentClient`指定本地路径保存数据，避免重启后数据丢失：

```Python
import chromadb  
client = chromadb.PersistentClient(path="./chroma_db")  # 持久化路径
```

### 创建集合

集合是 Chroma 的核心存储单元，类似于数据库中的表。

```Python
collection = client.create_collection(  
    name="my_collection",  
    embedding_function=embedding_model  # 可选嵌入函数
)  
```

**注意：**如果创建集合的时候不指定`embedding_function`，那么添加数据的时候必须手动加入`embeddings`。

### 删除集合

```Python
client.delete_collection(name="my_collection")  
```

### 添加数据

通过`.add()`方法插入文档、向量、元数据及唯一 ID：

```Python
collection.add(  
    documents=["人工智能是未来的核心技术。", "Chroma 是高效的向量数据库。"],  
    metadatas=[{"source": "book1"}, {"source": "article2"}],  # 可选元数据 
    ids=["doc1", "doc2"]  # 唯一标识符，需符合命名规则
)  
```

**注意：**如果创建集合的时候没有指定`embedding_function`，添加数据的时候必须加入embeddings参数。

### 删除数据

通过`.delete()`方法移除指定 ID 或匹配条件的文档：

```Python
collection.delete(ids=["doc1"])  # 按 ID 删除 
# 或按条件删除  
collection.delete(where={"source": "article2"})  # 按元数据过滤删除  
```

### 查询数据

通过`.query()`实现相似性搜索或条件过滤：

```Python
results = collection.query(  
    query_texts=["向量数据库"],  # 查询文本  
    n_results=2,  # 返回最相似的 2 条结果 
    where={"source": "book1"}  # 可选过滤条件 
)  
print(results["documents"])  # 输出匹配的文档  
```



## 进阶使用

Chroma现在大多和LLM结合使用用于构建RAG，Langchain就封装了Chroma方便用户使用，具体可以参考官网：[Chroma | 🦜️🔗 LangChain](https://python.langchain.com/docs/integrations/vectorstores/chroma/)

为了方便使用我也对chroma进行二次封装，结合了embedding模型。

首先我封装了embedding类，支持离线和在线

```Python
# CustomEmbedding.py
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
    
```

然后封装chroma，用于与chromadb交互

```Python
import chromadb
from CustomEmbedding import EmbeddingModel

import uuid
from chromadb.config import Settings
from typing import Union, List, Dict

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
        
    
   
```

### 初始化embedding，chroma和数据

使用的时候必须初始化`embedding`类。

```Python
model_name = "bge-m3"
model_path =  "xxxxx/bge-m3"
url = "http://127.0.0.1:9997/v1"
api_key = "EMPTY"
myembedding = EmbeddingModel(model_name=model_name, 
               model_path=model_path, 
               url=url, 
               api_key=api_key)
```

初始化两种数据，**sports**和**city**

```Python
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
```

初始化`chroma`类

```Python
db_manager = ChromaDBManager(embedding_model=myembedding)
```

### 添加集合

添加`collection`，添加两个`collection`，分为为**sports**和**city**

```Python
collection_sports = "sports"
db_manager.create_collection(collection_sports)
doc_infos_sports = db_manager.add_texts(collection_sports, 
                                  texts_sports, 
                                  metadatas_sports)

collection_city = "city"
db_manager.create_collection(collection_city)
doc_infos_city = db_manager.add_texts(collection_city, 
                                  texts_city, 
                                  metadatas_city)


```

### 查看集合

查看一下现在有什么集合

```Python
collection_list = db_manager.get_collections_list()
print(collection_list)
# ['city', 'sports']
```

### 查询集合

查询集合，问：** '奥尼尔选秀'**，并且指定查询**sports**集合

```Python
query = '奥尼尔选秀'

query_res = db_manager.query_texts(
    collection_name='sports',
    query_text=query,
    n_results=1
)
print(query_res)
'''
 [{'id': 'da433748-258c-11f0-9f61-3cecefb2262e', 
'text': '奥尼尔在1992年NBA选秀中于第1轮第1位以状元秀的身份被奥兰多魔术队选中',
 'metadata': {'domain': 'sports', 'source': 3}}]
'''
```

### 删除集合

删除**sports**集合

```Python
db_manager.delete_collection(collection_name='sports')
```

我们再看一下有什么**collection**，**sports**集合被删除了，现在还有**city**集合


```Python
collection_list = db_manager.get_collections_list()
print(collection_list)
# ['city']
```

### 获得collection中所有的数据

返回id、text和metadata


```Python
collection_results = db_manager.get_collection_texts(collection_name='city')
for collection_result in collection_results:
    print(collection_result)

{'id': '0f04c0be-258d-11f0-9f61-3cecefb2262e', 'text': '北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市', 'metadata': {'domain': 'city', 'source': 0}}  
{'id': '0f04c172-258d-11f0-9f61-3cecefb2262e', 'text': '2023年，北京市全年实现地区生产总值43760.7亿元，按不变价格计算，比上年增长5.2%', 'metadata': {'domain': 'city', 'source': 1}}  
{'id': '0f04c1d6-258d-11f0-9f61-3cecefb2262e', 'text': '纽约（英语：New York），位于美国纽约州南端（非纽约州首府），为美国人口最多的城市、纽约都会区的核心', 'metadata': {'domain': 'city', 'source': 2}}  
{'id': '0f04c226-258d-11f0-9f61-3cecefb2262e', 'text': '纽约位于美国东北部，滨临大西洋海岸，坐拥世界上最大天然港口之一的纽约和新泽西港', 'metadata': {'domain': 'city', 'source': 3}}
```

### 删除collection中的指定数据

```Python
id_list = ['0f04c0be-258d-11f0-9f61-3cecefb2262e', '0f04c172-258d-11f0-9f61-3cecefb2262e']
db_manager.delete_texts('city', id_list)

collection_results = db_manager.get_collection_texts(collection_name='city')
for collection_result in collection_results:
    print(collection_result)
{'id': '0f04c1d6-258d-11f0-9f61-3cecefb2262e', 'text': '纽约（英语：New York），位于美国纽约州南端（非纽约州首府），为美国人口最多的城市、纽约都会区的核心', 'metadata': {'domain': 'city', 'source': 2}}
{'id': '0f04c226-258d-11f0-9f61-3cecefb2262e', 'text': '纽约位于美国东北部，滨临大西洋海岸，坐拥世界上最大天然港口之一的纽约和新泽西港', 'metadata': {'domain': 'city', 'source': 3}}


```

### 修改collection中的文本

如果提供ID就更新ID对应的文本，如果没有提供ID就插入新的文本  
下面首先演示一下没有提供ID的情况：需要提供新的文本和metadata信息

```Python
texts_city2 = [
    "上海市地处长江三角洲冲积平原，地势坦荡低平， 属亚热带季风性气候，最大河流为黄浦江。",
    "2024年，上海市实现地区生产总值53926.71亿元，按不变价格计算，比上年增长5.0%。",
    "上海市是中国国际经济、金融、贸易、航运、科技创新中心，第三产业为其支柱产业，有着外贸物流、金融保险业、信息服务业、旅游业、房地产业和其他新兴服务业， 成为拉动经济增长“主动力”"
    ]
metadatas_city2 = [{'domain': 'city2', 'source': idx} for idx in range(len(texts_city2))]
db_manager.upsert_texts(collection_name='city', texts=texts_city2, metadatas=metadatas_city2)

[{'id': 'cce06eb6-258e-11f0-9f61-3cecefb2262e',
  'metadata': {'domain': 'city2', 'source': 0}},
 {'id': 'cce06f92-258e-11f0-9f61-3cecefb2262e',
  'metadata': {'domain': 'city2', 'source': 1}},
 {'id': 'cce06ff6-258e-11f0-9f61-3cecefb2262e',
  'metadata': {'domain': 'city2', 'source': 2}}]
```

插入新数据后再查看一下集合内的所有文本，发现插入进去了

```Python
collection_results = db_manager.get_collection_texts(collection_name='city')
for collection_result in collection_results:
    print(collection_result)

{'id': '0f04c1d6-258d-11f0-9f61-3cecefb2262e', 'text': '纽约（英语：New York），位于美国纽约州南端（非纽约州首府），为美国人口最多的城市、纽约都会区的核心', 'metadata': {'domain': 'city', 'source': 2}}
{'id': '0f04c226-258d-11f0-9f61-3cecefb2262e', 'text': '纽约位于美国东北部，滨临大西洋海岸，坐拥世界上最大天然港口之一的纽约和新泽西港', 'metadata': {'domain': 'city', 'source': 3}}
{'id': 'cce06eb6-258e-11f0-9f61-3cecefb2262e', 'text': '上海市地处长江三角洲冲积平原，地势坦荡低平， 属亚热带季风性气候，最大河流为黄浦江。', 'metadata': {'domain': 'city2', 'source': 0}}
{'id': 'cce06f92-258e-11f0-9f61-3cecefb2262e', 'text': '2024年，上海市实现地区生产总值53926.71亿元，按不变价格计算，比上年增长5.0%。', 'metadata': {'domain': 'city2', 'source': 1}}
{'id': 'cce06ff6-258e-11f0-9f61-3cecefb2262e', 'text': '上海市是中国国际经济、金融、贸易、航运、科技创新中心，第三产业为其支柱产业，有着外贸物流、金融保险业、信息服务业、旅游业、房地产业和其他新兴服务业， 成为拉动经济增长“主动力”', 'metadata': {'domain': 'city2', 'source': 2}}

```

下面试一下提供ID，修改现有的文本


```Python
ids = ['0f04c1d6-258d-11f0-9f61-3cecefb2262e', '0f04c226-258d-11f0-9f61-3cecefb2262e']
texts_city3 = ['北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市',
               '2023年，北京市全年实现地区生产总值43760.7亿元，按不变价格计算，比上年增长5.2%']
metadatas_city3 = [{'domain': 'city', 'source': 2}, {'domain': 'city', 'source': 3}]

db_manager.upsert_texts(collection_name='city', texts=texts_city3, metadatas=metadatas_city3, ids=ids)
[{'id': '0f04c1d6-258d-11f0-9f61-3cecefb2262e',
  'metadata': {'domain': 'city', 'source': 2}},
 {'id': '0f04c226-258d-11f0-9f61-3cecefb2262e',
  'metadata': {'domain': 'city', 'source': 3}}]

```

查看一下修改的结果，我们把纽约的信息，全部修改成了北京


```Python
collection_results = db_manager.get_collection_texts(collection_name='city')
for collection_result in collection_results:
    print(collection_result)
{'id': '0f04c1d6-258d-11f0-9f61-3cecefb2262e', 'text': '北京市（Beijing），简称“京”，古称燕京、北平，是中华人民共和国首都、直辖市、国家中心城市、超大城市', 'metadata': {'domain': 'city', 'source': 2}}
{'id': '0f04c226-258d-11f0-9f61-3cecefb2262e', 'text': '2023年，北京市全年实现地区生产总值43760.7亿元，按不变价格计算，比上年增长5.2%', 'metadata': {'domain': 'city', 'source': 3}}
{'id': 'cce06eb6-258e-11f0-9f61-3cecefb2262e', 'text': '上海市地处长江三角洲冲积平原，地势坦荡低平， 属亚热带季风性气候，最大河流为黄浦江。', 'metadata': {'domain': 'city2', 'source': 0}}
{'id': 'cce06f92-258e-11f0-9f61-3cecefb2262e', 'text': '2024年，上海市实现地区生产总值53926.71亿元，按不变价格计算，比上年增长5.0%。', 'metadata': {'domain': 'city2', 'source': 1}}
{'id': 'cce06ff6-258e-11f0-9f61-3cecefb2262e', 'text': '上海市是中国国际经济、金融、贸易、航运、科技创新中心，第三产业为其支柱产业，有着外贸物流、金融保险业、信息服务业、旅游业、房地产业和其他新兴服务业， 成为拉动经济增长“主动力”', 'metadata': {'domain': 'city2', 'source': 2}}

```
## 参考

[Introduction - Chroma Docs](https://docs.trychroma.com/docs/overview/introduction)