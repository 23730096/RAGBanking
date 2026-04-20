from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List

import requests

from core.setup_logging import setup_logging
from core.rag_runtime import load_app_settings, hybrid_retrieve
from generation.prompt_builder import (
    build_prompt,
    select_context_chunks,
    build_sources,
)

LOGGER = logging.getLogger("generation.rag")


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
    LOGGER.info("Calling Ollama | model=%s | timeout=%s", llm_cfg["model_name"], timeout)
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    return response.json().get("response", "").strip()


BANKING_HINTS = [
    "tài khoản", "tai khoan", "mở tài khoản", "mo tai khoan", "thẻ", "the",
    "thẻ tín dụng", "the tin dung", "thẻ ghi nợ", "vay", "tiết kiệm", "tiet kiem",
    "lãi suất", "lai suat", "chuyển khoản", "chuyen khoan", "napas", "sms banking",
    "internet banking", "mobile banking", "ibanking", "ebanking", "otp", "phí",
    "phi", "biểu phí", "bieu phi", "hạn mức", "han muc", "atm", "swift",
    "cif", "sổ tiết kiệm", "so tiet kiem", "ngân hàng", "ngan hang"
]

SMALLTALK_PATTERNS = [
    "xin chào", "chào", "hello", "hi", "bạn là ai", "ban la ai", "cảm ơn", "cam on", "tạm biệt", "tam biet"
]


def _normalize_router_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_smalltalk(question: str) -> bool:
    q = _normalize_router_text(question)
    return any(p in q for p in SMALLTALK_PATTERNS)


def is_banking_related(question: str) -> bool:
    q = _normalize_router_text(question)
    return any(k in q for k in BANKING_HINTS)


def should_abstain(results: List[Dict], min_items: int, min_score: float) -> bool:
    if not results:
        return True

    if len(results) >= min_items:
        strong = [r for r in results if r.get("score", 0) >= min_score]
        if strong:
            return False

    top_score = float(results[0].get("score", 0))
    if top_score >= min_score:
        return False

    return True


def generate_answer(question: str) -> Dict[str, Any]:
    settings = load_app_settings()

    LOGGER.info("Question: %s", question)
    if is_smalltalk(question) and not is_banking_related(question):
        return {
            "question": question,
            "answer": "Xin chào! Tôi là trợ lý hỏi đáp nghiệp vụ ngân hàng. Bạn hãy nhập câu hỏi về sản phẩm, dịch vụ, biểu phí, thủ tục hoặc quy định ngân hàng.",
            "sources": [],
            "dropped_context": 0,
        }

    retrieved_results = hybrid_retrieve(question)

    retrieval_cfg = settings.get("retrieval", {})
    min_items = int(retrieval_cfg.get("min_context_items", 1))
    min_score = float(retrieval_cfg.get("min_answer_score", 0.4))
    max_chars = int(retrieval_cfg.get("max_context_chars", 4000))

    LOGGER.info(
        "Decision config | min_items=%s | min_score=%s | max_chars=%s | retrieved=%s",
        min_items,
        min_score,
        max_chars,
        len(retrieved_results),
    )

    if should_abstain(retrieved_results, min_items, min_score):
        LOGGER.warning("Abstain before context selection | retrieved=%s", len(retrieved_results))
        return {
            "question": question,
            "answer": "Không tìm thấy đủ bằng chứng phù hợp trong dữ liệu để trả lời chắc chắn. Bạn có thể hỏi rõ hơn về sản phẩm, dịch vụ, biểu phí hoặc thủ tục ngân hàng cần tra cứu.",
            "sources": [],
            "dropped_context": 0,
        }

    selected, dropped = select_context_chunks(retrieved_results, max_chars)
    LOGGER.info("Context selection | selected=%s | dropped=%s", len(selected), len(dropped))

    if should_abstain(selected, min_items, min_score):
        LOGGER.warning("Abstain after context selection | selected=%s", len(selected))
        return {
            "question": question,
            "answer": "Không tìm thấy đủ bằng chứng phù hợp trong dữ liệu để trả lời chắc chắn. Bạn có thể hỏi rõ hơn về sản phẩm, dịch vụ, biểu phí hoặc thủ tục ngân hàng cần tra cứu.",
            "sources": [],
            "dropped_context": len(dropped),
        }

    prompt = build_prompt(question, selected)
    answer = call_ollama(prompt, settings)

    sources = build_sources(selected)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "dropped_context": len(dropped),
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
