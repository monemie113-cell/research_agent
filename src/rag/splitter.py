from typing import List, Dict, Any


def chunk_by_paragraph(documents: List[Dict[str, Any]], max_chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    按段落分割 + 长度控制（递归字符分割的简化版）T7
    """
    chunks = []
    chunk_id = 0

    for doc in documents:
        paragraphs = doc["text"].split("\n\n")

        current_chunk = ""
        for para in paragraphs:
            # 如果当前段落本身超过 max_chunk_size，强行切分
            if len(para) > max_chunk_size * 1.5:
                # 按句子分割
                sentences = para.replace("。", "。\n").replace("！", "！\n").split("\n")
                for sent in sentences:
                    if len(current_chunk) + len(sent) > max_chunk_size and current_chunk:
                        chunks.append({
                            "id": chunk_id,
                            "text": current_chunk.strip(),
                            "source": doc["source"],
                            "page": doc["page"],
                        })
                        chunk_id += 1
                        current_chunk = ""
                    current_chunk += sent
                continue

            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "text": current_chunk.strip(),
                    "source": doc["source"],
                    "page": doc["page"],
                })
                chunk_id += 1
                current_chunk = ""

            current_chunk += para + "\n\n"

        if current_chunk.strip():
            chunks.append({
                "id": chunk_id,
                "text": current_chunk.strip(),
                "source": doc["source"],
                "page": doc["page"],
            })
            chunk_id += 1

    return chunks


def semantic_chunk(documents: List[Dict[str, Any]], embedding_model, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    语义分割：在句子相似度骤降处切断。
    面试题第7题：这是比固定长度分割更高级的策略。
    注意：此函数需要传入 embedding_model，会消耗较多计算资源。
    """
    # 这是高级功能，我们后续阶段再实现
    # 先使用 paragraph 分割
    return chunk_by_paragraph(documents)