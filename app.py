import os
import sys
import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_agent

# 加载环境变量（ModelScope 会通过平台密钥管理注入）
load_dotenv()

# 初始化 Agent（全局只加载一次）
print("⏳ 正在初始化 Agent，请稍候...")
agent = create_agent()
print("✅ Agent 初始化完成！")


def chat_with_agent(message, history):
    """处理用户消息并返回回复"""
    state = {"messages": [], "current_step": 0}

    # 将历史消息转换为 LangChain 格式
    for user_msg, bot_msg in history:
        if user_msg:
            state["messages"].append(HumanMessage(content=user_msg))

    state["messages"].append(HumanMessage(content=message))

    try:
        final_state = agent.invoke(state)
        last_message = final_state["messages"][-1]
        reply = last_message.content if hasattr(last_message, "content") else str(last_message)
        history.append([message, reply])
        return history, ""
    except Exception as e:
        error_msg = f"❌ 出错了：{str(e)}"
        history.append([message, error_msg])
        return history, ""


# 创建 Gradio 界面
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="🧠 Research Agent",
    description="基于 LangGraph 的智能研究助手，支持工具调用和知识库检索",
    theme="soft",
)

if __name__ == "__main__":
    demo.launch()