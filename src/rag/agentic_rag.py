import os

from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

from src.rag.retriever import HybridRetriever
from src.rag.reranker import Reranker

from dotenv import load_dotenv
load_dotenv()

class AgenticRAG:


    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = Reranker()

        # 读取环境变量，并设置默认值
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY")

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=api_key,
            openai_api_base=base_url + "/v1",
            temperature=0.0,
        )


    def retrieve_with_feedback(self, query: str, max_retries: int = 2) -> List[Dict[str, Any]]:
        """
        Agentic RAG 核心逻辑：
        1. 执行检索
        2. 评估检索结果质量
        3. 如果质量不足，改写查询并重试
        """
        current_query = query
        all_results = []

        for attempt in range(max_retries + 1):
            # 1. 执行检索
            candidates = self.retriever.search(current_query, top_k=10)

            # 2. 重排序
            reranked = self.reranker.rerank(current_query, candidates, top_k=5)

            # 3. 质量评估
            quality = self._assess_quality(current_query, reranked)

            if quality["is_good"]:
                print(f"✅ 检索质量良好（第 {attempt + 1} 次尝试）")
                return reranked

            if attempt < max_retries:
                # 4. 查询改写
                current_query = self._rewrite_query(current_query, quality["reason"])
                print(f"🔄 检索质量不足，改写查询重试（第 {attempt + 2} 次尝试）")
                print(f"   → 新查询: {current_query}")

        # 如果全部失败，返回最后一次的结果
        print("⚠️ 已达最大重试次数，返回最后结果")
        return reranked


    def _assess_quality(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估检索结果质量
        返回: {"is_good": bool, "reason": str}
        """
        if not results:
            return {"is_good": False, "reason": "未检索到任何结果"}

        # 检查最高分是否足够高
        top_score = results[0].get("rerank_score", 0)
        if top_score < 0.3:
            return {"is_good": False, "reason": f"最高相关度偏低 ({top_score:.2f})"}

        return {"is_good": True, "reason": "相关度足够"}


    def _rewrite_query(self, original_query: str, reason: str) -> str:
        """使用 LLM 改写查询，提升检索效果"""
        prompt = f"""
        用户原始查询：{original_query}
        检索效果不佳的原因：{reason}

        请将这个查询改写为更有利于文档检索的版本，要求：
        1. 保留核心意图
        2. 使用更正式、更具体的词汇
        3. 只输出改写后的查询，不要其他内容
        """
        response = self.llm.invoke(prompt)
        return response.content.strip()