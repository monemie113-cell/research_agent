import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

from src.rag.embedding import encode_texts, get_embedding_model

# 向量数据库封装（使用 Qdrant）t15
class VectorStore:

    def __init__(self, collection_name: str = "research_docs", host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self):
        """确保 Collection 存在，不存在则创建"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            # 获取 Embedding 维度
            model = get_embedding_model()
            vector_size = model.get_embedding_dimension()

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ 创建 Collection: {self.collection_name} (dim={vector_size})")

    def add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        """
        批量添加文档 Chunk 到向量库。
        每个 Chunk 包含：id, text, source, page
        """
        if not chunks:
            return

        # 提取文本列表
        texts = [chunk["text"] for chunk in chunks]

        # 批量计算 Embedding
        embeddings = encode_texts(texts)

        # 准备 Qdrant 插入数据
        points = []
        # 使用全局递增计数器作为 ID（Qdrant 要求整数或 UUID）
        # 先查询当前 Collection 中已有的最大 ID
        try:
            collection_info = self.client.get_collection(self.collection_name)
            current_max = collection_info.points_count if collection_info.points_count else 0
        except Exception:
            current_max = 0

        for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = current_max + i  # 整数 ID，从当前最大值开始递增

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk.get("source", "unknown"),
                        "page": chunk.get("page", 0),
                    },
                )
            )

        # 分批插入
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        print(f"✅ 成功插入 {len(points)} 个 Chunk 到向量库")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        向量检索：根据查询向量，返回最相似的 top_k 个 Chunk。
        兼容 qdrant-client >= 1.7.0 的 query_points API。
        """
        query_vector = encode_texts([query])[0]

        # 新版 qdrant-client 使用 query_points，返回结构不同
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k,
            with_payload=True,
        )

        # 解析结果（result 是一个包含 points 列表的命名元组）
        hits = result.points if hasattr(result, 'points') else result

        return [
            {
                "text": hit.payload["text"],
                "source": hit.payload.get("source", "unknown"),
                "page": hit.payload.get("page", 0),
                "score": hit.score,
            }
            for hit in hits
        ]