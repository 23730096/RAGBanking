from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from core.setup_logging import setup_logging
from core.load_settings import load_settings

LOGGER = logging.getLogger("vectorstore.qdrant")


def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_settings(settings_path: str | Path = "config/settings.yaml") -> Dict[str, Any]:
    settings = read_yaml(settings_path)
    return load_settings(settings)


def get_qdrant_distance(distance_name: str) -> Distance:
    value = str(distance_name).lower().strip()

    if value == "cosine":
        return Distance.COSINE
    if value == "euclid":
        return Distance.EUCLID
    if value == "dot":
        return Distance.DOT

    raise ValueError(f"Unsupported Qdrant distance: {distance_name}")


def create_qdrant_client(settings: Dict[str, Any]) -> QdrantClient:
    cfg = settings["vector_database"]

    return QdrantClient(
        url=cfg["url"],
        api_key=cfg.get("api_key"),
        timeout=int(cfg.get("timeout", 30)),
    )


def get_embedding_files(settings: Dict[str, Any]) -> List[Path]:
    processed_dir = Path(settings["data"]["processed_dir"])
    embedding_root = processed_dir / "embeddings"

    if not embedding_root.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_root}")

    files = sorted(embedding_root.glob("*/*_embeddings.json"))

    if not files:
        raise FileNotFoundError(f"No embedding files found under: {embedding_root}")

    return files


def load_embedding_items(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported embedding file structure: {path}")


def collect_all_embedding_items(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = get_embedding_files(settings)
    all_items: List[Dict[str, Any]] = []

    LOGGER.info("Found %s embedding files", len(files))

    for file_path in files:
        items = load_embedding_items(file_path)
        LOGGER.info("Loaded %s items from %s", len(items), file_path)
        all_items.extend(items)

    if not all_items:
        raise ValueError("No embedding items loaded from embedding files")

    return all_items


def ensure_collection(client: QdrantClient, settings: Dict[str, Any], sample_vector_size: int) -> None:
    cfg = settings["vector_database"]
    collection_name = cfg["collection_name"]
    configured_vector_size = int(cfg["vector_size"])
    configured_distance = cfg["distance"]

    if configured_vector_size != sample_vector_size:
        raise ValueError(
            f"Vector size mismatch: settings.yaml={configured_vector_size}, "
            f"embedding_file={sample_vector_size}"
        )

    existing_collections = client.get_collections().collections
    existing_names = [c.name for c in existing_collections]

    if collection_name in existing_names:
        LOGGER.info("Collection already exists: %s", collection_name)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=configured_vector_size,
            distance=get_qdrant_distance(configured_distance),
        ),
    )

    LOGGER.info(
        "Created Qdrant collection | name=%s | vector_size=%s | distance=%s",
        collection_name,
        configured_vector_size,
        configured_distance,
    )


def validate_embedding_items(items: List[Dict[str, Any]]) -> int:
    first_vector_size: int | None = None

    for idx, item in enumerate(items):
        if "embedding" not in item:
            raise ValueError(f"Missing 'embedding' in item index={idx}")

        vector = item["embedding"]
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Invalid embedding vector in item index={idx}")

        if first_vector_size is None:
            first_vector_size = len(vector)
        elif len(vector) != first_vector_size:
            raise ValueError(
                f"Inconsistent vector size at item index={idx}: "
                f"expected {first_vector_size}, got {len(vector)}"
            )

    if first_vector_size is None:
        raise ValueError("No valid embeddings found")

    return first_vector_size


def build_points(items: List[Dict[str, Any]]) -> List[PointStruct]:
    points: List[PointStruct] = []

    for idx, item in enumerate(items, start=1):
        payload = {
            "chunk_id": item.get("chunk_id"),
            "text": item.get("text"),
            "modality": item.get("modality", "text"),
            "embedding_model": item.get("embedding_model"),
            "metadata": item.get("metadata", {}),
        }

        points.append(
            PointStruct(
                id=idx,
                vector=item["embedding"],
                payload=payload,
            )
        )

    return points


def upsert_in_batches(
    client: QdrantClient,
    collection_name: str,
    points: List[PointStruct],
    batch_size: int = 128,
) -> None:
    total = len(points)

    for start in range(0, total, batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        LOGGER.info("Upserted %s/%s points", min(start + batch_size, total), total)


def print_collection_summary(client: QdrantClient, collection_name: str) -> None:
    info = client.get_collection(collection_name)
    LOGGER.info(
        "Collection summary | name=%s | points_count=%s",
        collection_name,
        info.points_count,
    )


def main() -> None:
    setup_logging()

    LOGGER.info("===== START INDEX TO QDRANT =====")

    settings = load_app_settings("config/settings.yaml")
    client = create_qdrant_client(settings)

    items = collect_all_embedding_items(settings)
    LOGGER.info("Total embedding items loaded: %s", len(items))

    vector_size = validate_embedding_items(items)
    LOGGER.info("Detected vector size from embeddings: %s", vector_size)

    ensure_collection(client, settings, sample_vector_size=vector_size)

    points = build_points(items)
    LOGGER.info("Built %s Qdrant points", len(points))

    collection_name = settings["vector_database"]["collection_name"]
    upsert_in_batches(
        client=client,
        collection_name=collection_name,
        points=points,
        batch_size=128,
    )

    print_collection_summary(client, collection_name)

    LOGGER.info("===== DONE INDEX TO QDRANT =====")


if __name__ == "__main__":
    main()