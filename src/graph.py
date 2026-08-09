import os
import json
import openai
from typing import TypedDict, Annotated, Literal, List, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode

# 导入创建的工具
from src.tools import TOOL_MAP, TOOLS

# 加载环境变量
load_dotenv()

# ============================================================
# 第1步：定义 Agent 的状态（State）
# ============================================================
# State 让 Agent 具备"记忆现场和可恢复执行"的能力，不再是黑盒,T3
class AgentState(TypedDict):
    """Agent 的完整状态。所有节点共享这个状态对象。"""
    messages: Annotated[List[Any], add_messages]  # 对话历史（自动追加）
    current_step: int  # 当前执行步数，用于防止死循环


# ============================================================
# 第2步：初始化 LLM 并绑定工具
# ============================================================
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

if not api_key:
    raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY")

print(f"✅ 使用 DeepSeek API: {api_key[:10]}...")


# 创建原生 OpenAI 客户端（指向 DeepSeek）
client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url + "/v1",  # DeepSeek 官方要求加 /v1
)

os.environ["OPENAI_API_KEY"] = api_key

# 用 client 参数传入 ChatOpenAI
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=base_url + "/v1",   # DeepSeek 要求 /v1 后缀
    temperature=0.3,
)

# 将工具描述列表绑定到 LLM, 相当于告诉模型："你有这些工具可以用，它们的用法如下"
llm_with_tools = llm.bind_tools(TOOLS)


# ============================================================
# 第3步：定义节点（Node）函数
# ============================================================

def agent_node(state: AgentState) -> AgentState:
    """
    核心节点：Agent 的"思考"节点。
    面试题第67题：ReAct 中的 Thought 步骤在这里体现。
    LLM 根据当前对话状态，决定下一步是调用工具还是直接回答。
    """
    messages = state["messages"]
    current_step = state.get("current_step", 0)

    print(f"\n🤔 [第 {current_step + 1} 轮思考] Agent 正在分析...")

    # 调用 LLM（已绑定工具，模型会自动判断是否需要调用工具）
    response = llm_with_tools.invoke(messages)

    # 打印 LLM 的思考过程（便于调试）
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc.get("name", "未知") for tc in response.tool_calls]
        print(f"   💡 决策：调用工具 → {', '.join(tool_names)}")
    else:
        # 如果没有 tool_calls，说明 LLM 决定直接回答
        print(f"   💡 决策：直接回答用户")

    # 返回更新后的状态（消息自动追加到 messages 中）
    return {
        "messages": [response],
        "current_step": current_step + 1,
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """
    执行节点：执行 LLM 请求的工具调用。
    ReAct 中的 Action + Observation 步骤在此体现。
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 获取 LLM 请求调用的工具列表
    tool_calls = last_message.tool_calls

    if not tool_calls:
        return state

    # 逐个执行工具
    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]

        print(f"   🔧 执行工具：{tool_name}({json.dumps(tool_args, ensure_ascii=False)})")

        # 从 TOOL_MAP 中查找对应的函数
        tool_func = TOOL_MAP.get(tool_name)

        if tool_func is None:
            # 如果工具不存在，返回错误信息
            result = f"错误：未找到名为 '{tool_name}' 的工具"
        else:
            try:
                # 执行工具函数
                result = tool_func(**tool_args)
            except Exception as e:
                result = f"工具执行异常：{str(e)}"

        print(f"   📋 观察结果：{result[:100]}{'...' if len(result) > 100 else ''}")

        # 构造 ToolMessage（LangGraph 标准格式）
        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tc["id"],  # 必须与 tool_calls 中的 id 对应
            )
        )

    # 返回更新后的状态
    return {
        "messages": tool_messages,
        "current_step": state["current_step"],
    }


def should_continue(state: AgentState) -> Literal["tool_executor", END]:
    """
    条件边函数：决定下一步往哪走。
    Conditional Edge 是 LangGraph 的核心要素之一。T3

    如果 LLM 请求了工具调用 → 去 tool_executor 节点
    如果没有请求工具 → 结束（END）
    """
    messages = state["messages"]
    last_message = messages[-1]

    # 检查最后一轮是否有 tool_calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_executor"

    return END


# ============================================================
# 第4步：构建状态图（StateGraph）T4
# ============================================================
def build_agent_graph():
    """构建并返回完整的 LangGraph 状态图"""

    # 创建状态图（指定状态类型为 AgentState）
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("agent", agent_node)                  # 思考决策节点
    workflow.add_node("tool_executor", tool_executor_node)  # 工具执行节点

    # 设置入口点
    workflow.set_entry_point("agent")

    # 添加条件边：agent → 根据决策 → tool_executor 或 END
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tool_executor": "tool_executor",
            END: END,
        }
    )

    # 添加普通边：tool_executor 执行完后，必须回到 agent（继续思考）
    workflow.add_edge("tool_executor", "agent")

    # 编译状态图（编译后才能运行）
    graph = workflow.compile()

    return graph


# ============================================================
# 第5步：创建 Agent 实例（供外部调用）
# ============================================================
def create_agent():
    """工厂函数：创建并返回一个可运行的 Agent"""
    return build_agent_graph()