import os
import json
import threading
import queue
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
# Fix for PydanticUserError: `ChatOpenAI` is not fully defined
import langchain_core
from langchain_core.caches import BaseCache
from langchain_core.callbacks import BaseCallbackHandler, Callbacks
try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import StructuredTool
import file_manager

class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self, token_queue):
        self.queue = token_queue
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

class chat:
    def __init__(self, api_key, api_url, system_prompt, model="deepseek-v3-250324", file_manager_instance=None):
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
        
        self.store = {}
        
        tools = []
        if file_manager_instance:
            def _query_files(query: str = "") -> str:
                return file_manager_instance.query_files(query or "")
            def _open_file(filename: str) -> str:
                return file_manager_instance.open_file(filename)
                
            tools = [
                StructuredTool.from_function(
                    func=_query_files,
                    name="query_files",
                    description="Search for files in the knowledge base. Useful when user asks 'what files do I have' or looks for a document. If query is empty, lists all files."
                ),
                StructuredTool.from_function(
                    func=_open_file,
                    name="open_file",
                    description="Open a specific file by name. Useful when user asks to 'open' a file."
                )
            ]

        # 始终使用 Agent 模式
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # 即使 tools 为空，create_tool_calling_agent 也需要至少一个 tool 吗？
        # 实际上如果 tools 为空，agent 可能无法正常工作，或者我们应该提供一个 dummy tool
        # 但用户要求始终为 agent 模式，且文件知识库工具固定有。
        # 这里假设 file_manager_instance 总是会被传入。
        # 如果没有传入，我们可以给一个空的 tools 列表，但 create_tool_calling_agent 可能会报错。
        # 为了健壮性，如果 tools 为空，我们可以回退到 Chain 模式，或者添加一个 dummy tool。
        # 但根据用户要求 "llm.py中应该始终为agent模式"，我们尽量保持 Agent。
        
        if not tools:
             # 添加一个占位工具以防止报错
            def dummy_tool(query: str) -> str:
                return "No tools available."
            tools = [StructuredTool.from_function(func=dummy_tool, name="dummy_tool", description="A dummy tool.")]

        agent = create_tool_calling_agent(self.llm, tools, self.prompt)
        # AgentExecutor 负责执行 Agent 循环
        self.executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # 包装为带历史记录管理的 Runnable
        self.runnable = RunnableWithMessageHistory(
            self.executor,
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
            # 使用回调 + 线程队列实现 Agent 的流式输出
            token_queue = queue.Queue()
            handler = StreamCallbackHandler(token_queue)

            def run_agent():
                try:
                    self.runnable.invoke(
                        {"question": question},
                        config={
                            "configurable": {"session_id": session_id},
                            "callbacks": [handler]
                        }
                    )
                except Exception as e:
                    token_queue.put(f"[Error in agent: {str(e)}]")
                finally:
                    token_queue.put(None) # Signal end

            t = threading.Thread(target=run_agent)
            t.start()

            while True:
                token = token_queue.get()
                if token is None:
                    break
                yield token

            t.join()
        except Exception as e:
            yield f"[Error: {str(e)}]"
