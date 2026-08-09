import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 从 .env 读取配置
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 检查是否配置了 API Key
if not api_key:
    print("❌ 错误：未找到 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    exit(1)
print(f"✅ 已加载 API Key: {api_key[:10]}...")

# 创建 LLM 客户端
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0
)

try:
    print("⏳ 正在调用 DeepSeek API...")
    response = llm.invoke("世界第一高峰是哪座山，在哪里？")
    print(f"✅ 调用成功！\n回复：{response.content}")
except Exception as e:
    print(f"❌ 调用失败：{e}")