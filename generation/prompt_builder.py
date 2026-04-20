from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

def _resolve_raw_path(item: Dict) -> str:
    metadata = item.get("metadata", {}) or {}
    return str(
        metadata.get("raw_path")
        or metadata.get("file_path")
        or metadata.get("source_file")
        or ""
    ).strip()

def _resolve_source_name(item: Dict) -> str:
    metadata = item.get("metadata", {}) or {}

    file_name = metadata.get("file_name")
    if file_name:
        return str(file_name)

    file_path = metadata.get("file_path")
    if file_path:
        return Path(str(file_path)).name

    source_filename = metadata.get("source_filename")
    if source_filename:
        return str(source_filename)

    source_file = metadata.get("source_file")
    if source_file:
        return Path(str(source_file)).name

    return "unknown_source"


def _resolve_source_path(item: Dict) -> str:
    metadata = item.get("metadata", {}) or {}
    return str(metadata.get("file_path") or metadata.get("source_file") or "").strip()


def select_context_chunks(results: List[Dict], max_context_chars: int) -> Tuple[List[Dict], List[Dict]]:
    selected: List[Dict] = []
    dropped: List[Dict] = []
    total_chars = 0
    seen_signatures = set()

    for item in results:
        text = (item.get("text") or "").strip()
        source = _resolve_source_name(item)
        signature = (source, text[:160])

        if not text or signature in seen_signatures:
            dropped.append(item)
            continue

        estimated_size = len(text)
        if selected and total_chars + estimated_size > max_context_chars:
            dropped.append(item)
            continue

        seen_signatures.add(signature)
        total_chars += estimated_size
        selected.append(item)

    return selected, dropped


def build_context_block(results: List[Dict]) -> str:
    blocks: List[str] = []

    for idx, item in enumerate(results, start=1):
        source_file = _resolve_source_name(item)
        block = (
            f"[Nguồn {idx}]\n"
            f"source: {source_file}\n"
            f"nội dung: {item.get('text')}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)


def build_prompt(question: str, results: List[Dict]) -> str:
    context = build_context_block(results)

    return f"""
Bạn là trợ lý hỏi đáp nghiệp vụ ngân hàng theo hướng RAG (Hybrid).

Quy tắc bắt buộc:
- Nếu câu hỏi là nghiệp vụ ngân hàng thì ưu tiên trả lời dựa trên ngữ cảnh, không từ chối quá sớm.
- Chỉ dùng thông tin có trong phần NGỮ CẢNH.
- Không suy đoán, không bổ sung kiến thức bên ngoài, không tự diễn giải quá mức.
- Nếu ngữ cảnh không đủ chắc chắn, nói rõ là chưa tìm thấy đủ bằng chứng trong dữ liệu để trả lời chắc chắn.
- Nếu có nhiều nguồn nhưng mâu thuẫn hoặc không khớp nhau, cũng phải từ chối trả lời.
- Ưu tiên câu trả lời ngắn, rõ, đúng trọng tâm.
- Khi trả lời được, phải nêu các ý bám sát chứng cứ trong ngữ cảnh.
- Sau phần trả lời, liệt kê nguồn thực sự đã dùng; không liệt kê nguồn không dùng.
- Không được nhắc đến chunk_id, score_hybrid, score_dense hoặc score_lexical trong câu trả lời.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}

ĐỊNH DẠNG KẾT QUẢ:
Trả lời:
<1 đoạn ngắn hoặc các ý ngắn>

Nguồn sử dụng:
- <tên nguồn>
""".strip()


def build_sources(results: List[Dict]) -> List[Dict]:
    seen = set()
    sources: List[Dict] = []

    for item in results:
        source_name = _resolve_source_name(item)
        source_path = _resolve_source_path(item)
        key = (source_name, source_path)
        if key in seen:
            continue
        seen.add(key)
        raw_path = _resolve_raw_path(item)
        source_payload = {
            "source": source_name,
            "file_name": source_name,
            "file_path": source_path,
            "raw_path": raw_path,
            "score": item.get("score"),
            "download_url": None,
        }
        sources.append(source_payload)

    return sources
