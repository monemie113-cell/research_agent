import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    print("⚠️ 未检测到 TAVILY_API_KEY，将使用模拟数据。")
else:
    print(f"✅ 检测到 TAVILY_API_KEY: {api_key[:10]}...")

from src.tools import web_search

print("\n🔍 开始测试搜索功能...")
try:
    # 关键修复：使用 .invoke() 方法
    result = web_search.invoke("2026年AI Agent最新发展趋势")
    print("\n📋 搜索结果：")
    print(result)
except Exception as e:
    print(f"❌ 测试失败: {e}")