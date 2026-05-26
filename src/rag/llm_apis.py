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

    def generate_answer(self, question, context, history=[], use_knowledge=True, test_mode=False):
        """
        Generates an answer using the LLM based on the provided question and context.
        
        Args:
            question: 用户问题
            context: 知识库上下文
            history: 对话历史
            use_knowledge: 是否使用知识库信息
            test_mode: 是否为测试模式，测试模式下只输出答案
        """
        # 压缩上下文，提取关键信息
        compressed_context = self.compress_context(context, question) if use_knowledge else []
        
        # 准备上下文
        context_str = "\n\n".join(compressed_context) if compressed_context else "\n\n".join(context) if use_knowledge else "无"
        history_str = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in history]) if history else "无"
        
        # 检查知识库信息是否充足
        knowledge_available = bool(context_str.strip() and context_str != "无")
        
        # 构建提示词
        if knowledge_available and use_knowledge:
            if test_mode:
                # 测试模式：只输出答案
                prompt = f"""请根据以下提供的408考研相关知识回答问题：

背景知识：
{context_str}

问题：{question}

重要要求：
- 结合背景知识和你的知识回答
- 如果背景知识与你的知识冲突，以背景知识为准
- 只输出正确答案的选项字母，不要输出任何其他内容
- 例如：A
"""
                system_message = "你是一个408考研智能辅导助手，精通数据结构、计算机组成原理、操作系统和计算机网络四大科目。请根据提供的背景知识和你的知识回答问题，只输出正确答案的选项字母。"
            else:
                # 正常模式：输出完整回答
                prompt = f"""请根据以下提供的408考研相关知识回答问题，参考教材表述，突出核心概念和考试重点：

背景知识：
{context_str}

对话历史：
{history_str}

问题：{question}

请按照以下步骤回答：
1. 分析问题：明确问题的核心知识点和考察方向
2. 知识回顾：回顾相关的教材知识点
3. 推理过程：逐步推导答案
4. 最终答案：给出清晰、准确的结论
5. 考点分析：指出该问题涉及的考试重点和容易出错的地方
6. 知识溯源：对关键知识点标注来源，格式为[来源: 教材名称]

重要要求：
- 结合背景知识和你的知识回答
- 如果背景知识与你的知识冲突，以背景知识为准
- 保持回答结构清晰、逻辑严谨，符合考研教材的标准表述
"""
                system_message = "你是一个408考研智能辅导助手，精通数据结构、计算机组成原理、操作系统和计算机网络四大科目。请根据提供的背景知识和你的知识回答问题，突出核心概念和考试重点，解释清晰准确，避免使用过于前沿的技术术语，保持与教材内容一致。如果背景知识与你的知识冲突，以背景知识为准。"
        else:
            if test_mode:
                # 测试模式：只输出答案
                prompt = f"""请回答以下408考研相关问题：

问题：{question}

重要要求：
- 只输出正确答案的选项字母，不要输出任何其他内容
- 例如：A
"""
                system_message = "你是一个408考研智能辅导助手，精通数据结构、计算机组成原理、操作系统和计算机网络四大科目。请回答问题，只输出正确答案的选项字母。"
            else:
                # 正常模式：输出完整回答
                prompt = f"""请回答以下408考研相关问题，参考教材表述，突出核心概念和考试重点：

对话历史：
{history_str}

问题：{question}

请按照以下步骤回答：
1. 分析问题：明确问题的核心知识点和考察方向
2. 知识回顾：回顾相关的教材知识点
3. 推理过程：逐步推导答案
4. 最终答案：给出清晰、准确的结论
5. 考点分析：指出该问题涉及的考试重点和容易出错的地方
6. 知识溯源：对关键知识点标注来源，格式为[来源: 教材名称]

重要要求：
- 确保答案的准确性和权威性
- 保持回答结构清晰、逻辑严谨，符合考研教材的标准表述
"""
                system_message = "你是一个408考研智能辅导助手，精通数据结构、计算机组成原理、操作系统和计算机网络四大科目。请以考研教材的标准表述回答问题，突出核心概念和考试重点，解释清晰准确，避免使用过于前沿的技术术语，保持与教材内容一致。确保答案的准确性和权威性。"
        
        print(f"LLM Input: {prompt}")

        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.3,  # 降低温度，提高准确性
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
