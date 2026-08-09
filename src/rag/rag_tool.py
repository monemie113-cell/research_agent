from langchain_core.tools import tool
from src.rag.agentic_rag import AgenticRAG

_rag_engine = None

def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = AgenticRAG()
    return _rag_engine


@tool
def search_knowledge(query: str) -> str:
    """
    在知识库中智能搜索与问题相关的文档片段。支持自动查询改写和多轮重试。
    适用于需要参考特定文档、知识库或历史资料的问题。
    """
    engine = get_rag_engine()
    results = engine.retrieve_with_feedback(query, max_retries=2)

    if not results:
        return "未在知识库中找到相关信息。"

    formatted = []
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0))
        formatted.append(
            f"【片段 {i}】(来源: {r['source']}, 相关度: {score:.3f})\n{r['text'][:300]}\n"
        )

    return "\n".join(formatted)