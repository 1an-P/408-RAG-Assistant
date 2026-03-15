import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


class LLMClient:
    def __init__(self):
        load_dotenv(find_dotenv())
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen3-max")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation_history = []

    def rewrite_question(self, question, history=[]):
        """
        重写用户问题，结合对话历史提供更完整的上下文
        """
        history_str = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in history]) if history else "无"
        prompt = f"请根据对话历史重写用户的问题，使其更加完整和明确，突出408考研相关的知识点：\n\n对话历史：\n{history_str}\n\n用户当前问题：{question}\n\n重写后的问题："
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个408考研问题重写助手，负责将用户的问题结合对话历史重写为更加完整和明确的问题，突出考研相关的知识点。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content

    def compress_context(self, context, query):
        """
        压缩上下文，提取与问题相关的关键信息
        """
        compressed_context = []
        for i, doc in enumerate(context):
            prompt = f"请针对408考研问题 '{query}'，从以下文本中提取最相关的关键信息，重点关注教材中的定义、原理和核心知识点：\n\n{doc}\n\n提取的关键信息："
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个408考研知识提取助手，擅长从教材内容中提取与问题相关的关键信息。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )
            compressed_info = response.choices[0].message.content
            if compressed_info.strip():
                compressed_context.append(compressed_info)
        return compressed_context

    def generate_answer(self, question, context, history=[]):
        """
        Generates an answer using the LLM based on the provided question and context.
        """
        # 压缩上下文，提取关键信息
        compressed_context = self.compress_context(context, question)
        context_str = "\n\n".join(compressed_context) if compressed_context else "\n\n".join(context)
        history_str = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in history]) if history else "无"
        
        prompt = f"请根据以下提供的408考研相关知识回答问题，参考教材表述，突出核心概念和考试重点：\n\n{context_str}\n\n对话历史：\n{history_str}\n\n问题：{question}"
        print(f"LLM Input: {prompt}")

        messages = [
            {
                "role": "system",
                "content": "你是一个408考研智能辅导助手，精通数据结构、计算机组成原理、操作系统和计算机网络四大科目。请根据提供的背景知识和对话历史，以考研教材的标准表述回答问题，突出核心概念和考试重点，解释清晰准确，避免使用过于前沿的技术术语，保持与教材内容一致。",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def add_to_history(self, question, answer):
        """
        添加对话到历史记录
        """
        self.conversation_history.append((question, answer))
        # 限制历史记录长度，避免过长
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]

    def get_history(self):
        """
        获取对话历史
        """
        return self.conversation_history

    def clear_history(self):
        """
        清空对话历史
        """
        self.conversation_history = []
