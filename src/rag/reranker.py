from typing import List, Dict, Any, TypeVar
import numpy as np

# 定义Document类型变量
Document = TypeVar('Document')


class SimpleReranker:
    """简单的重排序器，基于规则和语义相似度进行重排序"""
    
    def __init__(self, embedding_model=None):
        """
        初始化重排序器
        
        Args:
            embedding_model: 嵌入模型，用于计算语义相似度
        """
        self.embedding_model = embedding_model
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            top_k: 返回的文档数量
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []
        
        # 计算每个文档的得分
        scored_docs = []
        for doc in documents:
            score = self._calculate_score(query, doc)
            scored_docs.append((doc, score))
        
        # 根据得分排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个文档
        return [doc for doc, score in scored_docs[:top_k]]
    
    def _calculate_score(self, query: str, doc: Document) -> float:
        """
        计算文档的得分
        
        Args:
            query: 用户查询
            doc: 文档
            
        Returns:
            文档的得分
        """
        score = 0.0
        
        # 1. 基于关键词匹配的得分
        query_tokens = query.lower().split()
        doc_content = doc.page_content.lower()
        
        # 计算关键词匹配数量
        matched_tokens = 0
        for token in query_tokens:
            if token in doc_content:
                matched_tokens += 1
        
        # 关键词匹配得分
        if query_tokens:
            keyword_score = matched_tokens / len(query_tokens)
            score += keyword_score * 0.5  # 关键词匹配占50%权重
        
        # 2. 基于语义相似度的得分（如果有嵌入模型）
        if self.embedding_model:
            try:
                # 计算查询和文档的嵌入
                query_embedding = self.embedding_model.embed_query(query)
                doc_embedding = self.embedding_model.embed_query(doc.page_content)
                
                # 计算余弦相似度
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                score += similarity * 0.5  # 语义相似度占50%权重
            except Exception as e:
                # 如果嵌入模型调用失败，只使用关键词匹配得分
                pass
        
        # 3. 基于文档长度的惩罚（可选）
        # 过长或过短的文档可能不是最佳选择
        doc_length = len(doc.page_content)
        if 100 <= doc_length <= 1000:
            # 长度适中的文档加分
            score += 0.1
        
        return score
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)