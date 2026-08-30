import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.graph import create_agent

load_dotenv()

print("⏳ 正在初始化 Agent，请稍候...")
agent = create_agent()
print("✅ Agent 初始化完成！")


# ============================================================
# 自定义回调：记录 Token 消耗
# ============================================================
class TokenCostHandler(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        try:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = token_usage.get('prompt_tokens', 0)
            self.completion_tokens = token_usage.get('completion_tokens', 0)
            self.total_tokens = token_usage.get('total_tokens', 0)
        except Exception as e:
            print(f"Token追踪出错: {e}")


# 全局 token 处理器（每次调用重置）
token_handler = TokenCostHandler()


def chat_fn(message, history):
    state = {"messages": [], "current_step": 0}

    for user_msg, bot_msg in history:
        if user_msg:
            state["messages"].append(HumanMessage(content=user_msg))

    state["messages"].append(HumanMessage(content=message))

    try:
        # 重置 token 计数
        token_handler.prompt_tokens = 0
        token_handler.completion_tokens = 0
        token_handler.total_tokens = 0

        # 调用 Agent，传入回调
        final_state = agent.invoke(state, config={"callbacks": [token_handler]})
        last_message = final_state["messages"][-1]
        reply = last_message.content if hasattr(last_message, "content") else str(last_message)

        # 在回复末尾追加 Token 消耗信息
        token_info = (
            f"\n\n---\n"
            f"📊 Token 消耗：输入 {token_handler.prompt_tokens} | 输出 {token_handler.completion_tokens} | 总计 {token_handler.total_tokens}"
        )
        return reply + token_info

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"❌ 出错了：{str(e)}"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🧠 Research Agent",
    description="基于 LangGraph 的智能研究助手，支持工具调用和知识库检索",
    # type="tuples",
)

if __name__ == "__main__":
    demo.launch()