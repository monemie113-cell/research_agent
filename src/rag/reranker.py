from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from typing import List, Dict, Any

#Cross-Encoder 重排序器
class Reranker:

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        print(f"⏳ 正在加载重排模型: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        print("✅ 重排模型加载完成")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        对候选文档进行重排序，返回 top_k 个最相关的
        """
        if not candidates:
            return []

        # 构造 (query, document) 对
        pairs = [[query, doc["text"]] for doc in candidates]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = self.model(**inputs, return_dict=True).logits.view(-1,).float()

        # 按分数排序
        for i, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[i])

        sorted_candidates = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return sorted_candidates[:top_k]