import datetime
import math
import os
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from src.rag.rag_tool import search_knowledge

# ============================================================
# 1. 计算器工具
# ============================================================
@tool
def calculator(expression: str) -> str:
    """
    执行数学计算。输入参数为数学表达式字符串，如 '2 + 3 * 4' 或 'sqrt(16)'。
    适用于需要精确数值运算的场景。
    """
    try:
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}，请检查表达式格式"


# ============================================================
# 2. 获取当前时间工具
# ============================================================
@tool
def get_current_time(format: str = "YYYY-MM-DD HH:MM:SS") -> str:
    """
    获取当前日期和时间。可指定输出格式。
    - format: 'YYYY-MM-DD' 返回日期，'HH:MM:SS' 返回时间，默认为完整格式。
    """
    now = datetime.datetime.now()
    if format == "YYYY-MM-DD":
        return now.strftime("%Y-%m-%d")
    elif format == "HH:MM:SS":
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# 3. 搜索引擎工具
# ============================================================
@tool
def web_search(query: str) -> str:
    """
    在互联网上搜索实时信息。当用户询问新闻、最新动态、实时数据或你不知道的实时信息时，应该使用这个工具。
    """
    # 从环境变量获取 API Key
    api_key = os.getenv("TAVILY_API_KEY")

    # 如果没有配置 Key，给出友好提示并返回模拟数据，防止程序崩溃
    if not api_key:
        print("⚠️ 未配置 TAVILY_API_KEY，使用模拟搜索数据。请在 .env 中添加 TAVILY_API_KEY")
        # 保留我们之前的模拟逻辑作为降级方案（可选）
        if "天气" in query:
            return "【模拟】上海今日晴，25°C。"
        return "【模拟】未找到实时信息（未配置搜索 API Key）。"

    try:
        # 初始化 Tavily 工具（只返回前 3 条最相关结果）
        tavily = TavilySearchResults(
            api_key=api_key,
            max_results=3,
            include_answer=True,  # 获取 AI 整理好的简要回答
            include_raw_content=False,  # 不返回原始 HTML，节省 token
        )

        # 执行搜索
        result = tavily.invoke(query)

        # Tavily 返回的是一个列表，我们把它格式化成易读的文本
        if not result:
            return f"未在互联网上找到关于 '{query}' 的相关信息。"

        # 格式化输出，让 Agent 能看懂
        formatted = []
        for item in result:
            title = item.get("title", "无标题")
            content = item.get("content", "")
            url = item.get("url", "")
            formatted.append(f"【{title}】\n{content}\n来源: {url}\n")

        return "\n".join(formatted)

    except Exception as e:
        return f"联网搜索时发生错误: {str(e)}，请稍后重试。"


# ============================================================
# 导出工具列表和映射表（供 graph.py 使用）
# ============================================================
TOOLS = [calculator, get_current_time, web_search, search_knowledge]

# 工具映射表：名称 → 函数（用于 tool_executor_node）
TOOL_MAP = {tool.name: tool for tool in TOOLS}