import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_agent
from langchain_core.messages import HumanMessage


def main():
    """交互式 ReAct Agent 运行入口"""

    print("=" * 60)
    print("🧠 欢迎使用 ReAct Agent 交互终端")
    print("📌 你可以问任何问题，Agent 会自动决定是否调用工具")
    print("📌 输入 'exit' 或 'quit' 退出")
    print("=" * 60)

    # 创建 Agent
    agent = create_agent()

    # 初始化会话状态
    state = {
        "messages": [],
        "current_step": 0,
    }

    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 你: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("👋 再见！")
                break

            if not user_input:
                continue

            # 将用户消息追加到状态
            state["messages"].append(HumanMessage(content=user_input))
            state["current_step"] = 0

            print("\n" + "-" * 40)

            # 执行 Agent（LangGraph 会驱动整个 ReAct 循环）
            final_state = agent.invoke(state)

            # 提取最终回答
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                print(f"\n🤖 Agent: {last_message.content}")
            else:
                print(f"\n🤖 Agent: {str(last_message)}")

            # 将状态更新为最终状态（保持对话连续性）
            state = final_state

        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或输入 'exit' 退出")

if __name__ == "__main__":
    main()