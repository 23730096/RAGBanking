from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.load_settings import load_settings

LOGGER = logging.getLogger("rag.runtime")
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_app_settings(settings_path: str | Path = "config/settings.yaml") -> Dict[str, Any]:
    settings = read_yaml(settings_path)
    return load_settings(settings)


@lru_cache(maxsize=1)
def create_qdrant_client() -> QdrantClient:
    settings = load_app_settings()
    cfg = settings["vector_database"]
    return QdrantClient(
        url=cfg["url"],
        api_key=cfg.get("api_key"),
        timeout=int(cfg.get("timeout", 30)),
    )


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    settings = load_app_settings()
    cfg = settings["embedding"]
    return SentenceTransformer(
        cfg["model"],
        device=cfg.get("device", "cpu"),
    )


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def build_query_text(question: str) -> str:
    question = normalize_text(question)
    return f"query: {question}"


def embed_query(question: str) -> List[float]:
    model = load_embedding_model()
    vector = model.encode(
        build_query_text(question),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def search_qdrant_dense(question: str, top_k: int, collection_name: str):
    client = create_qdrant_client()
    query_vector = embed_query(question)

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


def tokenize_for_lexical_score(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(normalize_text(text).lower())


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize_for_lexical_score(query)
    text_tokens = tokenize_for_lexical_score(text)

    if not query_tokens or not text_tokens:
        return 0.0

    query_tf: Dict[str, int] = {}
    for token in query_tokens:
        query_tf[token] = query_tf.get(token, 0) + 1

    text_tf: Dict[str, int] = {}
    for token in text_tokens:
        text_tf[token] = text_tf.get(token, 0) + 1

    overlap = 0.0
    for token, q_count in query_tf.items():
        if token in text_tf:
            overlap += min(q_count, text_tf[token])

    coverage = overlap / max(len(query_tokens), 1)
    density = overlap / math.sqrt(max(len(text_tokens), 1))
    return float((0.7 * coverage) + (0.3 * density))


def hybrid_retrieve(question: str) -> List[Dict[str, Any]]:
    settings = load_app_settings()
    retrieval_cfg = settings["retrieval"]
    collection_name = settings["vector_database"]["collection_name"]

    top_k = int(retrieval_cfg.get("top_k", 5))
    score_threshold = float(retrieval_cfg.get("score_threshold", 0.0))
    dense_weight = float(retrieval_cfg.get("dense_weight", 0.7))
    lexical_weight = float(retrieval_cfg.get("lexical_weight", 0.3))
    candidate_multiplier = int(retrieval_cfg.get("dense_candidate_multiplier", 4))
    dense_limit = max(top_k, top_k * candidate_multiplier)

    raw_results = search_qdrant_dense(
        question=question,
        top_k=dense_limit,
        collection_name=collection_name,
    )

    final_results: List[Dict[str, Any]] = []
    for item in raw_results:
        dense_score = float(item.score)
        payload = item.payload or {}
        text = payload.get("text", "")
        if not text:
            continue

        keyword_score = lexical_score(question, text)
        hybrid_score = (dense_weight * dense_score) + (lexical_weight * keyword_score)
        if hybrid_score < score_threshold:
            continue

        final_results.append(
            {
                "id": item.id,
                "score": hybrid_score,
                "dense_score": dense_score,
                "lexical_score": keyword_score,
                "chunk_id": payload.get("chunk_id"),
                "text": text,
                "modality": payload.get("modality"),
                "embedding_model": payload.get("embedding_model"),
                "metadata": payload.get("metadata", {}),
            }
        )

    final_results.sort(key=lambda x: x["score"], reverse=True)
    LOGGER.info(
        "Hybrid retrieval done | question=%s | dense_limit=%s | final=%s",
        question,
        dense_limit,
        len(final_results),
    )
    return final_results[:top_k]
