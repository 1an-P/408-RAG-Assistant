import json
import logging
from tqdm import tqdm
from embedding_apis import OpenAIEmbedding
from langchain_community.vectorstores import Chroma


class VectorDatabase:
    def __init__(self, embedding=None, persist_directory=None):
        self.embedding = embedding if embedding else OpenAIEmbedding()
        self.persist_directory = persist_directory
        self.vectordb = None

    def create_from_documents(
        self, documents, collection_name="rag_collection", persist_directory=None
    ):
        """从文档创建向量数据库"""
        if persist_directory:
            self.persist_directory = persist_directory

        # 使用Chroma创建向量数据库
        self.vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

        # 持久化数据库
        self.vectordb.persist()
        logging.info(f"知识库构建完成，包含 {len(documents)} 个文档块")
        return self.vectordb

    def load_existing(self, persist_directory):
        """加载已有的向量数据库"""
        self.persist_directory = persist_directory
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        return self.vectordb

    def similarity_search(self, query, k=3, collection_name="rag_collection"):
        """相似度搜索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")

        # Chroma的similarity_search方法直接返回Document对象列表
        results = self.vectordb.similarity_search(
            query=query,
            k=k
        )
        return results

    def mmr_search(self, query, k=3, fetch_k=10, lambda_param=0.5, collection_name="rag_collection"):
        """使用MMR（Maximum Marginal Relevance）进行搜索，平衡相关性和多样性"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")

        # 使用Chroma的max_marginal_relevance_search方法
        results = self.vectordb.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_param
        )
        return results

    def get_collection_count(self, collection_name="rag_collection"):
        """获取向量库中的文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        # Chroma没有直接获取文档数量的方法，这里返回近似值
        # 实际应用中可能需要根据具体情况调整
        return len(self.vectordb.get()['ids'])
