import os
import logging
from dotenv import load_dotenv, find_dotenv
from src.rag.document_processor import DocumentProcessor
from src.rag.vector_db import VectorDatabase
from src.rag.llm_apis import LLMClient
from src.rag.reranker import SimpleReranker

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 加载环境变量
load_dotenv(find_dotenv())


class RAGSystem:
    def __init__(
        self,
        persist_dir,
        strategy="default",
        alpha=0.5,
        use_ocr=True,
        use_table_extraction=True,
    ):
        self.strategy = strategy
        self.alpha = alpha  # 设置默认的alpha值
        self.document_processor = DocumentProcessor(
            strategy=self.strategy,
            use_ocr=use_ocr,
            use_table_extraction=use_table_extraction,
        )
        self.vector_db = VectorDatabase(persist_directory=persist_dir)
        self.llm_client = LLMClient()
        self.reranker = SimpleReranker(embedding_model=self.vector_db.embedding)
        self.persist_dir = persist_dir

    def build_knowledge_base(self, data_dir):
        """构建知识库"""
        if len(os.listdir(os.path.dirname(self.persist_dir))) > 0:
            logging.info("知识库已存在，跳过构建。")
            return
        # 获取所有文档路径
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(data_dir)
            for file in files
        ]

        # 处理文档
        processed_docs = self.document_processor.process_documents(file_paths)

        # 保存处理后的文档以供检查
        output_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "output",
            self.strategy,
        )

        # Clear existing files in the directory
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        os.makedirs(output_dir, exist_ok=True)

        for doc in processed_docs:
            source_filename = os.path.basename(
                doc.metadata.get("source", "unknown_file")
            )
            filename_base, _ = os.path.splitext(source_filename)

            file_output_dir = os.path.join(output_dir, filename_base)
            os.makedirs(file_output_dir, exist_ok=True)

            # Find the next available chunk number
            chunk_num = 1
            while os.path.exists(
                os.path.join(file_output_dir, f"chunk_{chunk_num}.txt")
            ):
                chunk_num += 1

            with open(
                os.path.join(file_output_dir, f"chunk_{chunk_num}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(doc.page_content)

        logging.info(f"切割后的文档已保存到 {output_dir}")

        # 构建向量数据库
        self.vector_db.create_from_documents(processed_docs)
        logging.info(
            f"知识库构建完成，包含 {self.vector_db.get_collection_count()} 个文档块"
        )

    def query(self, question, k=3, use_mmr=False, use_hybrid=True, alpha=None, fallback_to_llm=True, test_mode=False):
        """查询知识库并生成答案"""
        if not os.path.exists(self.persist_dir):
            # 知识库不存在，直接调用大模型
            logging.warning("知识库不存在，直接调用大模型回答")
            history = self.llm_client.get_history()
            rewritten_question = self.llm_client.rewrite_question(question, history)
            answer = self.llm_client.generate_answer(rewritten_question, [], history, use_knowledge=False, test_mode=test_mode)
            self.llm_client.add_to_history(question, answer)
            return answer

        if not self.vector_db.vectordb:
            self.vector_db.load_existing(self.persist_dir)

        # 如果没有提供alpha，使用实例的默认值
        if alpha is None:
            alpha = self.alpha

        # 重写问题，结合对话历史
        history = self.llm_client.get_history()
        rewritten_question = self.llm_client.rewrite_question(question, history)
        logging.info(f"原始问题: {question}")
        logging.info(f"重写问题: {rewritten_question}")

        # 检索相关文档
        if use_hybrid:
            # 使用混合检索
            retrieved_docs = self.vector_db.hybrid_search(rewritten_question, k=k*2, alpha=alpha)
        elif use_mmr:
            # 使用MMR检索
            retrieved_docs = self.vector_db.mmr_search(rewritten_question, k=k*2)
        else:
            # 使用传统相似度检索
            retrieved_docs = self.vector_db.similarity_search(rewritten_question, k=k*2)
        
        # 对检索到的文档进行重排序
        reranked_docs = self.reranker.rerank(rewritten_question, retrieved_docs, top_k=k)
        
        context = [doc.page_content for doc in reranked_docs]

        logging.info(f"找到 {len(retrieved_docs)} 个相关文档块，重排序后保留 {len(reranked_docs)} 个.")

        # 检查知识库信息是否充足
        knowledge_available = bool(context and any(doc.strip() for doc in context))
        
        # 生成答案
        if knowledge_available or not fallback_to_llm:
            # 使用知识库信息生成答案
            answer = self.llm_client.generate_answer(rewritten_question, context, history, use_knowledge=True, test_mode=test_mode)
        else:
            # 知识库信息不足，直接调用大模型
            logging.warning("知识库信息不足，直接调用大模型回答")
            answer = self.llm_client.generate_answer(rewritten_question, [], history, use_knowledge=False, test_mode=test_mode)
        
        logging.info(f"生成的答案: {answer}")

        # 添加到对话历史
        self.llm_client.add_to_history(question, answer)

        return answer

    def multi_turn_query(self, questions, alpha=None, fallback_to_llm=True, test_mode=False):
        """多轮对话查询"""
        # 如果没有提供alpha，使用实例的默认值
        if alpha is None:
            alpha = self.alpha

        answers = []
        for question in questions:
            answer = self.query(question, alpha=alpha, fallback_to_llm=fallback_to_llm, test_mode=test_mode)
            answers.append(answer)
            print(f"用户: {question}")
            print(f"助手: {answer}")
            print("-" * 50)
        return answers


if __name__ == "__main__":
    # 定义项目根目录
    project_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # 定义数据路径
    persist_directory = os.path.join(project_dir, "data_base/vector_db/408.db")
    knowledge_base_dir = os.path.join(project_dir, "data_base/knowledge_db")

    # 确保目录存在
    os.makedirs(os.path.dirname(persist_directory), exist_ok=True)

    # 初始化RAG系统
    # strategy: "default", "paper", "chapter"
    # "default": 默认切割方式，使用RecursiveCharacterTextSplitter
    # "paper": 按论文结构切割，使用PaperTextSplitter
    # "chapter": 按章节标题切割，使用ChapterTitleSplitter
    rag_system = RAGSystem(persist_dir=persist_directory, strategy="chapter")

    # 构建知识库
    logging.info("开始构建知识库...")
    rag_system.build_knowledge_base(data_dir=knowledge_base_dir)
    logging.info("知识库构建完成。")

    # 执行示例查询
    logging.info("执行示例查询...")
    answer = rag_system.query("什么是操作系统？")
    logging.info(f"最终答案: {answer}")
    logging.info("查询完成。")

    # 执行多轮对话示例
    logging.info("执行多轮对话示例...")
    questions = [
        "什么是操作系统？",
        "它有哪些主要功能？",
        "请详细解释其中的进程管理功能"
    ]
    rag_system.multi_turn_query(questions)
    logging.info("多轮对话完成。")
