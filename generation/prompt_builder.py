from __future__ import annotations

from typing import Dict, List, Tuple


def _resolve_source_name(item: Dict) -> str:
    metadata = item.get("metadata", {}) or {}
    return metadata.get("source_filename") or metadata.get("source_file") or "unknown_source"


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
            f"chunk_id: {item.get('chunk_id')}\n"
            f"score_hybrid: {item.get('score', 0):.4f}\n"
            f"score_dense: {item.get('dense_score', 0):.4f}\n"
            f"score_lexical: {item.get('lexical_score', 0):.4f}\n"
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
- Từ chối các câu hỏi hoặc ký tự vô nghĩa, không phụ hợp ngữ cảnh.
- Chỉ dùng thông tin có trong phần NGỮ CẢNH.
- Không suy đoán, không bổ sung kiến thức bên ngoài, không tự diễn giải quá mức.
- Nếu ngữ cảnh không đủ chắc chắn, trả lời đúng nguyên văn: "Không đủ thông tin trong dữ liệu để trả lời."
- Nếu có nhiều nguồn nhưng mâu thuẫn hoặc không khớp nhau, cũng phải từ chối trả lời.
- Ưu tiên câu trả lời ngắn, rõ, đúng trọng tâm.
- Khi trả lời được, phải nêu các ý bám sát chứng cứ trong ngữ cảnh.
- Sau phần trả lời, liệt kê nguồn thực sự đã dùng; không liệt kê nguồn không dùng.

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
        source = _resolve_source_name(item)
        key = (item.get("chunk_id"), source)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "dense_score": item.get("dense_score"),
                "lexical_score": item.get("lexical_score"),
                "source": source,
            }
        )

    return sources
