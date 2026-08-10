import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.graph import create_agent

load_dotenv()

print("⏳ 正在初始化 Agent，请稍候...")
agent = create_agent()
print("✅ Agent 初始化完成！")


def chat_fn(message, history):
    state = {"messages": [], "current_step": 0}
    for user_msg, bot_msg in history:
        if user_msg:
            state["messages"].append(HumanMessage(content=user_msg))
    state["messages"].append(HumanMessage(content=message))
    try:
        final_state = agent.invoke(state)
        last_message = final_state["messages"][-1]
        return last_message.content if hasattr(last_message, "content") else str(last_message)
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