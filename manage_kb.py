import os
from qdrant_client.http import models
from src.rag.vector_store import VectorStore
from src.rag.loader import load_document
from src.rag.splitter import chunk_by_paragraph

store = VectorStore(collection_name='research_docs')


def list_documents():
    """列出当前知识库中所有文档来源"""
    collection_info = store.client.get_collection(store.collection_name)
    # 简单方法：搜索一个空字符串获取所有文档
    results = store.client.scroll(
        collection_name=store.collection_name,
        limit=100,
        with_payload=True,
    )
    sources = set()
    for point in results[0]:
        if point.payload and "source" in point.payload:
            sources.add(point.payload["source"])

    print(f"📚 当前知识库包含 {len(sources)} 个文档:")
    for s in sorted(sources):
        print(f"  - {s}")


def delete_document(source_name: str):
    """删除指定来源的文档"""
    store.client.delete(
        collection_name=store.collection_name,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key='source',
                    match=models.MatchValue(value=source_name),
                )
            ]
        )
    )
    print(f"✅ 已删除: {source_name}")


def add_document(file_path: str):
    """添加新文档到知识库"""
    docs = load_document(file_path)
    chunks = chunk_by_paragraph(docs)
    store.add_documents(chunks)
    print(f"✅ 已添加: {os.path.basename(file_path)}，共 {len(chunks)} 个 Chunk")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法:")
        print("  python manage_kb.py list                # 列出所有文档")
        print("  python manage_kb.py delete <文件名>     # 删除指定文档")
        print("  python manage_kb.py add <文件路径>      # 添加新文档")
    else:
        cmd = sys.argv[1]
        if cmd == "list":
            list_documents()
        elif cmd == "delete" and len(sys.argv) > 2:
            delete_document(sys.argv[2])
        elif cmd == "add" and len(sys.argv) > 2:
            add_document(sys.argv[2])
        else:
            print("❌ 无效命令")