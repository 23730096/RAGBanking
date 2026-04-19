from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.setup_logging import setup_logging
from core.rag_runtime import hybrid_retrieve

LOGGER = logging.getLogger("retrieval")


def retrieve(question: str) -> List[Dict[str, Any]]:
    if not question or not question.strip():
        LOGGER.warning("Received empty query for retrieval.")
        return []

    results = hybrid_retrieve(question)
    LOGGER.info("Retrieved %s documents (hybrid)", len(results))
    return results


def pretty_print(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("Không tìm thấy chunk phù hợp.")
        return

    for idx, item in enumerate(results, start=1):
        print(f"\n===== RESULT {idx} =====")
        print(f"score      : {item['score']:.4f}")
        print(f"dense      : {item.get('dense_score', 0):.4f}")
        print(f"lexical    : {item.get('lexical_score', 0):.4f}")
        print(f"chunk_id   : {item.get('chunk_id')}")
        print(f"metadata   : {item.get('metadata')}")
        print(f"text       : {item.get('text')}")


def main() -> None:
    setup_logging()

    question = input("Nhập câu hỏi: ").strip()
    if not question:
        print("Câu hỏi rỗng.")
        return

    results = retrieve(question)
    pretty_print(results)


if __name__ == "__main__":
    main()