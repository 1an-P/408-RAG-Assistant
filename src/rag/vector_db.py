import json
import logging
from tqdm import tqdm
from src.rag.embedding_apis import OpenAIEmbedding
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import string
import nltk
from nltk.corpus import stopwords

# 下载nltk停用词
nltk.download('stopwords', quiet=True)


class VectorDatabase:
    def __init__(self, embedding=None, persist_directory=None):
        self.embedding = embedding if embedding else OpenAIEmbedding()
        self.persist_directory = persist_directory
        self.vectordb = None
        self.bm25 = None
        self.documents = []

    def _tokenize(self, text):
        """简单的文本分词"""
        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        # 分词
        tokens = text.split()
        # 移除停用词
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        return tokens

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
        
        # 保存文档
        self.documents = documents
        
        # 初始化BM25
        try:
            # 分词
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
            # 创建BM25实例
            self.bm25 = BM25Okapi(tokenized_corpus)
            logging.info(f"BM25检索器初始化完成，包含 {len(documents)} 个文档")
        except Exception as e:
            logging.warning(f"初始化BM25检索器失败: {e}")
            # 如果初始化失败，设置为None
            self.bm25 = None
        
        logging.info(f"知识库构建完成，包含 {len(documents)} 个文档块")
        return self.vectordb

    def load_existing(self, persist_directory):
        """加载已有的向量数据库"""
        self.persist_directory = persist_directory
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
        
        # 从向量数据库中获取文档内容，初始化BM25
        try:
            # 获取所有文档
            collection = self.vectordb.get()
            ids = collection['ids']
            documents = []
            
            # 为每个文档创建Document对象
            for i, id in enumerate(ids):
                # 获取文档内容和元数据
                content = collection['documents'][i]
                metadata = collection['metadatas'][i]
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            
            # 保存文档
            self.documents = documents
            
            # 初始化BM25
            # 分词
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
            # 创建BM25实例
            self.bm25 = BM25Okapi(tokenized_corpus)
            logging.info(f"BM25检索器初始化完成，包含 {len(documents)} 个文档")
        except Exception as e:
            logging.warning(f"初始化BM25检索器失败: {e}")
            # 如果初始化失败，设置为None
            self.bm25 = None
            self.documents = []
        
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

    def hybrid_search(self, query, k=3, alpha=0.5):
        """混合检索：BM25 + 向量检索"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        
        # 如果BM25未初始化，回退到向量检索
        if not self.bm25 or not self.documents:
            logging.warning("BM25检索器未初始化，回退到向量检索")
            return self.similarity_search(query, k=k)

        # 1. BM25检索
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 按得分排序，获取前k*2个文档
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        bm25_results = [self.documents[i] for i in bm25_ranked_indices]
        
        # 2. 向量检索
        vector_results = self.similarity_search(query, k=k*2)
        
        # 3. 线性加权融合
        combined_results = self._linear_weighted_fusion(bm25_results, vector_results, k, alpha)
        
        return combined_results

    def _linear_weighted_fusion(self, bm25_results, vector_results, k, alpha):
        """线性加权融合两种检索结果"""
        # 构建文档内容到文档对象和得分的映射
        doc_scores = {}
        doc_map = {}
        
        # 为BM25结果评分
        for i, doc in enumerate(bm25_results):
            content = doc.page_content
            score = 1.0 / (i + 1)  # 排名越高，得分越高
            if content not in doc_scores:
                doc_scores[content] = 0
                doc_map[content] = doc
            doc_scores[content] += alpha * score
        
        # 为向量检索结果评分
        for i, doc in enumerate(vector_results):
            content = doc.page_content
            score = 1.0 / (i + 1)  # 排名越高，得分越高
            if content not in doc_scores:
                doc_scores[content] = 0
                doc_map[content] = doc
            doc_scores[content] += (1 - alpha) * score
        
        # 按得分排序
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前k个文档
        return [doc_map[content] for content, score in sorted_items[:k]]

    def get_collection_count(self, collection_name="rag_collection"):
        """获取向量库中的文档数量"""
        if not self.vectordb:
            raise ValueError("Vector database not initialized")
        # Chroma没有直接获取文档数量的方法，这里返回近似值
        # 实际应用中可能需要根据具体情况调整
        return len(self.vectordb.get()['ids'])
