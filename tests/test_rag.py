import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import load_document
from src.rag.splitter import chunk_by_paragraph

# 准备一个测试文件：在 data/ 目录下放一个 test.txt
# 如果还没有，手动创建一个

test_file = "../data/test.txt"
with open(test_file, "w", encoding="utf-8") as f:
    f.write("""这是第一段内容。介绍了 AI Agent 的基本概念。

这是第二段，讲的是 RAG 检索增强生成系统。RAG 可以极大减少大模型的幻觉问题。

这是第三段，内容是关于向量数据库的。向量数据库是 RAG 系统的核心存储组件。
""")

# 测试加载和分割
docs = load_document(test_file)
chunks = chunk_by_paragraph(docs)

print(f"加载了 {len(docs)} 页文档，分割为 {len(chunks)} 个 Chunk")
for i, chunk in enumerate(chunks[:3]):
    print(f"\nChunk {i}: {chunk['text'][:100]}...")