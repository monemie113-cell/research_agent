import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import load_document
from src.rag.splitter import chunk_by_paragraph
from src.rag.vector_store import VectorStore

# 1. 加载测试文档
test_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "test.txt")
docs = load_document(test_file)
chunks = chunk_by_paragraph(docs)
print(f"📄 加载了 {len(docs)} 页，切分为 {len(chunks)} 个 Chunk")

# 2. 存入向量库
store = VectorStore(collection_name="test_collection")
store.add_documents(chunks)

# 3. 测试检索
query = "LangGraph的核心三要素是什么？"
results = store.search(query, top_k=3)

print(f"\n🔍 检索结果（查询：{query}）")
for i, r in enumerate(results):
    print(f"{i+1}. [得分: {r['score']:.4f}] {r['text'][:80]}...")