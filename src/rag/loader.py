import os
from typing import List, Dict, Any

# ============================================================
# 依赖导入（带友好的缺失提示）
# ============================================================
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    # print("⚠️ 未安装 PyMuPDF，无法解析 PDF。请执行: pip install PyMuPDF")

try:
    from docx import Document
except ImportError:
    Document = None
    print("⚠️ 未安装 python-docx，无法解析 Word。请执行: pip install python-docx")


# ============================================================
# 2. 核心解析函数（每个函数返回统一格式的列表）
# ============================================================
def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 PDF 文件，提取所有页面的文本，合并为一个整体返回。
    """
    if fitz is None:
        raise ImportError("请安装 PyMuPDF: pip install PyMuPDF")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    doc = fitz.open(file_path)
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text and text.strip():
            full_text.append(f"【第{page_num}页】\n{text.strip()}")

    doc.close()

    # 如果没有提取到任何文本，可能是扫描件
    if not full_text:
        print(f"⚠️ PDF '{file_path}' 未提取到文本，可能是扫描件，建议后续增加 OCR 处理。")

    # 返回统一格式：整个文档作为一个条目，方便后续分块
    return [{
        "page": 1,
        "text": "\n\n".join(full_text) if full_text else "",
        "source": os.path.basename(file_path),
    }]


def load_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    加载 Word 文档 (.docx)，提取所有段落文本。
    """
    if Document is None:
        raise ImportError("请安装 python-docx: pip install python-docx")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    doc = Document(file_path)
    paragraphs = []

    for para in doc.paragraphs:
        if para.text and para.text.strip():
            paragraphs.append(para.text.strip())

    full_text = "\n\n".join(paragraphs)

    return [{
        "page": 1,
        "text": full_text,
        "source": os.path.basename(file_path),
    }]


def load_text(file_path: str) -> List[Dict[str, Any]]:
    """加载纯文本文件 (.txt)"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    return [{
        "page": 1,
        "text": content,
        "source": os.path.basename(file_path),
    }]


# ============================================================
# 3. 统一入口：根据扩展名自动选择加载器（这就是你调用的函数）
# ============================================================

def load_document(file_path: str) -> List[Dict[str, Any]]:
    """
    统一入口：根据文件扩展名自动调用对应的加载器。
    支持：.pdf, .docx, .txt
    """
    if file_path.endswith(".pdf"):
        return load_pdf(file_path)
    elif file_path.endswith(".docx"):
        return load_docx(file_path)
    elif file_path.endswith(".txt"):
        return load_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}，目前仅支持 .pdf, .docx, .txt")