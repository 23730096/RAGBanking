from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from core.load_settings import load_settings


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
    return SentenceTransformer(cfg["model"], device=cfg.get("device", "cpu"))


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize_for_lexical(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
