import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import load_document
from src.rag.splitter import chunk_by_paragraph
from src.rag.vector_store import VectorStore
from src.rag.agentic_rag import AgenticRAG

# 1. 确保文档入库
print("📂 正在加载文档...")
docs = load_document('../data/rag_intro.txt')
chunks = chunk_by_paragraph(docs)
store = VectorStore(collection_name='research_docs')
store.add_documents(chunks)
print(f"✅ 入库完成，共插入 {len(chunks)} 个 Chunk")

# 2. 初始化 AgenticRAG（它会使用同一个 collection）
rag = AgenticRAG()

# 3. 测试查询
query = "那个能减少幻觉的东西是什么？"
print(f"\n🔍 查询: {query}\n")
results = rag.retrieve_with_feedback(query)

print("\n📋 最终检索结果:")
for i, r in enumerate(results, 1):
    score = r.get('rerank_score', r.get('score', 0))
    print(f"{i}. [{score:.3f}] {r['text'][:100]}...")