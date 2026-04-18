from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import requests
import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.setup_logging import setup_logging
from core.load_settings import load_settings

LOGGER = logging.getLogger("generation.rag")


def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_settings(settings_path: str | Path = "config/settings.yaml") -> Dict[str, Any]:
    settings = read_yaml(settings_path)
    return load_settings(settings)


def create_qdrant_client(settings: Dict[str, Any]) -> QdrantClient:
    cfg = settings["vector_database"]
    return QdrantClient(
        url=cfg["url"],
        api_key=cfg.get("api_key"),
        timeout=int(cfg.get("timeout", 30)),
    )


def load_embedding_model(settings: Dict[str, Any]) -> SentenceTransformer:
    cfg = settings["embedding"]
    return SentenceTransformer(
        cfg["model"],
        device=cfg.get("device", "cpu"),
    )


def build_query_text(question: str) -> str:
    question = " ".join(question.split()).strip()
    return f"query: {question}"


def embed_query(model: SentenceTransformer, question: str) -> List[float]:
    query_text = build_query_text(question)
    vector = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int,
):
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return response.points


def filter_results(results, score_threshold: float) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for item in results:
        score = float(item.score)
        if score < score_threshold:
            continue

        payload = item.payload or {}
        filtered.append(
            {
                "id": item.id,
                "score": score,
                "chunk_id": payload.get("chunk_id"),
                "text": payload.get("text"),
                "modality": payload.get("modality"),
                "embedding_model": payload.get("embedding_model"),
                "metadata": payload.get("metadata", {}),
            }
        )

    return filtered


def retrieve_context(question: str, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    client = create_qdrant_client(settings)
    model = load_embedding_model(settings)

    top_k = int(settings["retrieval"].get("top_k", 5))
    score_threshold = float(settings["retrieval"].get("score_threshold", 0.0))
    collection_name = settings["vector_database"]["collection_name"]

    LOGGER.info("Question: %s", question)
    LOGGER.info(
        "Collection=%s | top_k=%s | score_threshold=%s",
        collection_name,
        top_k,
        score_threshold,
    )

    query_vector = embed_query(model, question)

    raw_results = search_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
    )

    return filter_results(raw_results, score_threshold=score_threshold)


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
Bạn là trợ lý hỏi đáp nghiệp vụ ngân hàng theo hướng RAG.

Nhiệm vụ:
- Chỉ được trả lời dựa trên phần NGỮ CẢNH được cung cấp.
- Không tự bịa thêm thông tin.
- Nếu ngữ cảnh không đủ để kết luận, phải trả lời rõ là: "Không đủ thông tin trong dữ liệu để trả lời."
- Cố gắng trả lời ngắn gọn, đúng trọng tâm, dễ hiểu.
- Sau câu trả lời, phải liệt kê nguồn tham chiếu đã dùng.

CÂU HỎI:
{question}

NGỮ CẢNH:
{context}

ĐỊNH DẠNG TRẢ LỜI:
Trả lời:
<ghi câu trả lời ở đây>

Nguồn sử dụng:
- <nguồn 1>
- <nguồn 2>
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

    data = response.json()
    return data.get("response", "").strip()


def generate_answer(question: str) -> Dict[str, Any]:
    settings = load_app_settings("config/settings.yaml")

    retrieved_results = retrieve_context(question, settings)

    if not retrieved_results:
        return {
            "question": question,
            "answer": "Không đủ thông tin trong dữ liệu để trả lời.",
            "sources": [],
            "retrieved_chunks": [],
        }

    prompt = build_prompt(question, retrieved_results)
    answer = call_ollama(prompt, settings)

    sources = []
    for item in retrieved_results:
        metadata = item.get("metadata", {})
        source_file = metadata.get("source_filename") or metadata.get("source_file") or "unknown_source"
        sources.append(
            {
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "source": source_file,
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_results,
    }


def pretty_print(result: Dict[str, Any]) -> None:
    print("\n==================== KẾT QUẢ ====================")
    print(f"Câu hỏi: {result['question']}\n")
    print("Trả lời:")
    print(result["answer"])

    print("\nNguồn truy xuất:")
    if not result["sources"]:
        print("- Không có")
    else:
        for idx, source in enumerate(result["sources"], start=1):
            print(
                f"- [{idx}] chunk_id={source['chunk_id']} | "
                f"score={source['score']:.4f} | source={source['source']}"
            )


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