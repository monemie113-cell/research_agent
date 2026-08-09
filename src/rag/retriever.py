import numpy as np
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
import jieba  # 中文分词

from src.rag.vector_store import VectorStore
from src.rag.embedding import encode_texts


# 混合检索器：向量检索 + BM25 关键词检索 + RRF 融合 + Cross-Encoder 重排
class HybridRetriever:

    def __init__(self, collection_name: str = "research_docs"):
        self.vector_store = VectorStore(collection_name=collection_name)
        self._bm25_index = None
        self._all_chunks = []  # 用于 BM25 构建时保存所有文本

    def _build_bm25_index(self, chunks: List[Dict[str, Any]]):
        """
        构建 BM25 索引（需要所有文档的文本）
        实际生产环境中，这部分应该在文档入库时同步构建
        """
        self._all_chunks = chunks
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        self._bm25_index = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（BM25 需要）"""
        return list(jieba.cut(text))

    def search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        混合检索主入口
        - vector_weight: 向量检索的权重（剩余为 BM25 权重）
        - top_k: 最终返回的文档数量
        """
        # 1. 向量检索：召回 top_k * 2 个候选（为后续重排留足空间）
        vector_candidates = self.vector_store.search(query, top_k=top_k * 2)

        # 2. BM25 检索（如果索引已构建）
        bm25_candidates = []
        if self._bm25_index is not None:
            tokenized_query = self._tokenize(query)
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            # 取 top_k * 2 个
            top_indices = np.argsort(bm25_scores)[-top_k * 2:][::-1]
            bm25_candidates = [
                {
                    "text": self._all_chunks[i]["text"],
                    "source": self._all_chunks[i].get("source", "unknown"),
                    "page": self._all_chunks[i].get("page", 0),
                    "score": bm25_scores[i],
                }
                for i in top_indices
            ]

        # 3. RRF 融合（面试题第44题：倒数排名融合）
        merged = self._rrf_fusion(vector_candidates, bm25_candidates, k=60)

        return merged[:top_k]

    def _rrf_fusion(self, list1: List[Dict], list2: List[Dict], k: int = 60) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合算法
        公式: RRF_score(d) = Σ 1 / (k + rank_i(d))
        """
        # 构建文档ID → 得分映射（用文本内容作为ID）
        scores = {}

        for rank, doc in enumerate(list1, start=1):
            doc_id = doc["text"][:100]  # 用前100字符作为标识
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

        for rank, doc in enumerate(list2, start=1):
            doc_id = doc["text"][:100]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

        # 按得分排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 构建返回结果
        result_map = {doc["text"][:100]: doc for doc in list1 + list2}
        return [result_map[doc_id] for doc_id in sorted_ids if doc_id in result_map]

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Cross-Encoder 重排序（调用重排模型）"""
        # 使用轻量级重排模型，但需要先加载模型
        # 为避免每次都加载，把模型加载放到外部
        pass