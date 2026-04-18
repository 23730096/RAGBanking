from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.setup_logging import setup_logging
from core.load_settings import load_settings

LOGGER = logging.getLogger("retrieval.qdrant")


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
    query_vector: list[float],
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


def retrieve(question: str) -> List[Dict[str, Any]]:
    settings = load_app_settings("config/settings.yaml")

    client = create_qdrant_client(settings)
    model = load_embedding_model(settings)

    top_k = int(settings["retrieval"].get("top_k", 5))
    score_threshold = float(settings["retrieval"].get("score_threshold", 0.0))
    collection_name = settings["vector_database"]["collection_name"]

    LOGGER.info("Question: %s", question)
    LOGGER.info("Collection: %s | top_k=%s | score_threshold=%s", collection_name, top_k, score_threshold)

    query_vector = embed_query(model, question)

    raw_results = search_qdrant(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        top_k=top_k,
    )

    final_results = filter_results(raw_results, score_threshold=score_threshold)
    return final_results


def pretty_print(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("Không tìm thấy chunk phù hợp.")
        return

    for idx, item in enumerate(results, start=1):
        print(f"\n===== RESULT {idx} =====")
        print(f"score      : {item['score']:.4f}")
        print(f"chunk_id   : {item.get('chunk_id')}")
        print(f"modality   : {item.get('modality')}")
        print(f"model      : {item.get('embedding_model')}")
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