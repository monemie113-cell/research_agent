"""
FastAPI 服务入口（与 Gradio 共存）
启动命令：uvicorn main_api:app --host 0.0.0.0 --port 8000
访问文档：http://localhost:8000/docs
"""

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.graph import create_agent

# 加载环境变量
load_dotenv()



# 1. Token 追踪回调（复用）
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
        except Exception:
            pass


# 2. 初始化 Agent（全局单例，启动时加载）
print("⏳ 正在初始化 Agent，请稍候...")
agent = create_agent()
print("✅ Agent 初始化完成！")


# 3. FastAPI 应用
app = FastAPI(
    title="Research Agent API",
    description="基于 LangGraph 的智能研究助手 API 服务",
    version="1.0.0"
)

# 允许跨域（方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. 请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求体"""
    message: str
    session_id: Optional[str] = None  # 预留，暂未实现


class ChatResponse(BaseModel):
    """聊天响应体"""
    reply: str
    tokens: Dict[str, int] = {}


# 5. API 路由
@app.get("/")
def root():
    return {
        "message": "Research Agent API 服务已启动",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    """健康检查"""
    return {"status": "healthy", "agent_loaded": agent is not None}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # 创建 Token 追踪器
        token_handler = TokenCostHandler()

        # 构建状态
        state = {
            "messages": [HumanMessage(content=request.message)],
            "current_step": 0
        }

        # 调用 Agent
        final_state = agent.invoke(state, config={"callbacks": [token_handler]})

        # 提取回复
        last_message = final_state["messages"][-1]
        reply = last_message.content if hasattr(last_message, "content") else str(last_message)

        return ChatResponse(
            reply=reply,
            tokens={
                "prompt": token_handler.prompt_tokens,
                "completion": token_handler.completion_tokens,
                "total": token_handler.total_tokens,
            }
        )

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"❌ API 错误:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


# 6. 启动入口（可选）
if __name__ == "__main__":
    import uvicorn
    print("🚀 启动 Research Agent API 服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000)