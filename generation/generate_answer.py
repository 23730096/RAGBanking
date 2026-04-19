from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import requests

from core.setup_logging import setup_logging
from core.rag_runtime import load_app_settings, hybrid_retrieve

LOGGER = logging.getLogger("generation.rag")


def build_context_block(results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []

    for idx, item in enumerate(results, start=1):
        metadata = item.get("metadata", {})
        source_file = metadata.get("source_filename") or metadata.get("source_file") or "unknown_source"

        block = (
            f"[Nguồn {idx}]\n"
            f"chunk_id: {item.get('chunk_id')}\n"
            f"score: {item.get('score'):.4f}\n"
            f"source: {source_file}\n"
            f"nội dung: {item.get('text')}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)


def build_prompt(question: str, results: List[Dict[str, Any]]) -> str:
    context = build_context_block(results)

    return f"""
Bạn là trợ lý hỏi đáp nghiệp vụ ngân hàng theo hướng RAG (Hybrid).

Nhiệm vụ:
- Chỉ được trả lời dựa trên phần NGỮ CẢNH được cung cấp.
- Không tự bịa thêm thông tin.
- Nếu ngữ cảnh không đủ để kết luận, phải trả lời rõ là: "Không đủ thông tin trong dữ liệu để trả lời."
- Trả lời ngắn gọn, đúng trọng tâm.
- Cuối câu trả lời phải liệt kê nguồn.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}

ĐỊNH DẠNG:
Trả lời:
...

Nguồn:
- ...
""".strip()


def call_ollama(prompt: str, settings: Dict[str, Any]) -> str:
    llm_cfg = settings["llm"]

    url = llm_cfg["base_url"].rstrip("/") + "/api/generate"
    payload = {
        "model": llm_cfg["model_name"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": llm_cfg.get("temperature", 0.2),
            "num_predict": llm_cfg.get("max_tokens", 512),
        },
    }

    timeout = int(llm_cfg.get("timeout", 60))
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    return response.json().get("response", "").strip()


def generate_answer(question: str) -> Dict[str, Any]:
    settings = load_app_settings()

    retrieved_results = hybrid_retrieve(question)

    if not retrieved_results:
        return {
            "question": question,
            "answer": "Không đủ thông tin trong dữ liệu để trả lời.",
            "sources": [],
        }

    prompt = build_prompt(question, retrieved_results)
    answer = call_ollama(prompt, settings)

    sources = []
    for item in retrieved_results:
        metadata = item.get("metadata", {})
        source_file = metadata.get("source_filename") or metadata.get("source_file") or "unknown_source"
        sources.append({
            "chunk_id": item.get("chunk_id"),
            "score": item.get("score"),
            "source": source_file,
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }


def pretty_print(result: Dict[str, Any]) -> None:
    print("\n==================== KẾT QUẢ ====================")
    print(f"Câu hỏi: {result['question']}\n")
    print("Trả lời:")
    print(result["answer"])

    print("\nNguồn:")
    for idx, source in enumerate(result.get("sources", []), start=1):
        print(f"- [{idx}] {source}")


def save_result_json(result: Dict[str, Any], output_path: str | Path = "outputs/last_answer.json") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    setup_logging()

    question = input("Nhập câu hỏi: ").strip()
    if not question:
        print("Câu hỏi rỗng.")
        return

    result = generate_answer(question)
    pretty_print(result)
    save_result_json(result)


if __name__ == "__main__":
    main()