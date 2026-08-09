from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer


# 全局单例模型（避免重复加载）
_model = None


def get_embedding_model(model_name: str = "BAAI/bge-small-zh-v1.5"):
    """
    懒加载 Embedding 模型。
    默认使用 bge-small-zh-v1.5（轻量，适合开发测试）。
    生产环境可切换为 bge-m3。
    """
    global _model
    if _model is None:
        print(f"⏳ 正在加载 Embedding 模型: {model_name}...")
        _model = SentenceTransformer(model_name, trust_remote_code=True)
        print("✅ Embedding 模型加载完成")
    return _model

def encode_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """
    批量将文本转为向量。
    normalize=True 时，向量归一化，便于余弦相似度计算。
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    return embeddings


def encode_query(query: str) -> np.ndarray:
    """将单个查询转为向量（用于检索）"""
    return encode_texts([query])[0]