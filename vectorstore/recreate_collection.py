from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.setup_logging import setup_logging
from core.load_settings import load_settings

LOGGER = logging.getLogger("vectorstore.recreate_collection")


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
        check_compatibility=False,
    )


def recreate_collection() -> None:
    settings = load_app_settings("config/settings.yaml")
    client = create_qdrant_client(settings)

    cfg = settings["vector_database"]
    collection_name = cfg["collection_name"]
    vector_size = int(cfg["vector_size"])
    distance = get_qdrant_distance(cfg["distance"])

    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if collection_name in existing_names:
        LOGGER.warning("Deleting existing collection: %s", collection_name)
        client.delete_collection(collection_name=collection_name)

    LOGGER.info(
        "Creating collection | name=%s | vector_size=%s | distance=%s",
        collection_name,
        vector_size,
        cfg["distance"],
    )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    info = client.get_collection(collection_name)
    LOGGER.info("Done | points_count=%s", info.points_count)
    print(f"Recreated collection: {collection_name}")


if __name__ == "__main__":
    setup_logging()
    recreate_collection()