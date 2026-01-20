from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

class DeepSeekChat:
    def __init__(self, api_key, api_url, system_prompt, model="deepseek-v3-250324"):
        # 处理 API URL，LangChain/OpenAI SDK 通常需要 Base URL (不带 /chat/completions)
        if api_url and api_url.endswith("/chat/completions"):
            api_url = api_url.replace("/chat/completions", "")
        
        # 初始化 ChatOpenAI
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_url,
            streaming=True,
            temperature=0.7
        )
        
        # 构建 Prompt 模板，包含系统提示词和历史记录占位符
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        # 构建处理链：Prompt -> LLM -> OutputParser (直接输出字符串)
        chain = self.prompt | self.llm | StrOutputParser()
        
        # 用于存储不同 Session 的历史记录
        self.store = {}
        
        # 包装为带历史记录管理的 Runnable
        self.runnable = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """获取指定 Session 的历史记录，如果不存在则创建新的"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def ask_stream(self, question, session_id="default"):
        """
        流式提问接口
        :param question: 用户问题
        :param session_id: 会话 ID，用于区分不同对话上下文
        :return: 生成器，产生流式文本块
        """
        try:
            for chunk in self.runnable.stream(
                {"question": question},
                config={"configurable": {"session_id": session_id}}
            ):
                yield chunk
        except Exception as e:
            yield f"[Error: {str(e)}]"
