from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from sentence_transformers import SentenceTransformer

from core.setup_logging import setup_logging
from core.load_settings import load_settings

LOGGER = logging.getLogger("embedding.text")


def read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_app_settings(settings_path: str | Path = "config/settings.yaml") -> Dict[str, Any]:
    settings = read_yaml(settings_path)
    return load_settings(settings)


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_chunk_file(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "chunks" in data and isinstance(data["chunks"], list):
            return data["chunks"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]

    raise ValueError(f"Unsupported chunk structure in file: {path}")


def build_chunk_id(chunk: Dict[str, Any], index: int, source_file: Path) -> str:
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])

    metadata = chunk.get("metadata", {}) or {}
    filename_stem = source_file.stem
    chunk_index = metadata.get("chunk_index", index)
    text_hash = sha1_text(str(chunk.get("text", "")))[:10]
    return f"{filename_stem}_{chunk_index}_{text_hash}"


def prepare_text_for_e5(text: str) -> str:
    # Với E5, passage phải có prefix passage:
    return f"passage: {normalize_text(text)}"


def extract_records_from_chunk_file(chunk_file: Path, model_name: str) -> List[Dict[str, Any]]:
    chunks = load_chunk_file(chunk_file)
    records: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = normalize_text(chunk.get("text", ""))
        if not text:
            LOGGER.warning("Skip empty chunk | file=%s | index=%s", chunk_file.name, idx)
            continue

        metadata = chunk.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}

        # thêm source info cho dễ trace
        metadata.setdefault("source_file", str(chunk_file))
        metadata.setdefault("source_filename", chunk_file.name)
        metadata.setdefault("source_date_dir", chunk_file.parent.name)

        chunk_id = build_chunk_id(chunk, idx, chunk_file)

        records.append(
            {
                "embedding_id": f"emb_{chunk_id}",
                "chunk_id": chunk_id,
                "modality": "text",
                "embedding_model": model_name,
                "text": text,
                "text_for_embedding": prepare_text_for_e5(text),
                "metadata": metadata,
            }
        )

    return records


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def save_embedding_file(output_path: Path, items: List[Dict[str, Any]]) -> None:
    ensure_parent_dir(output_path)
    payload = {
        "modality": "text",
        "count": len(items),
        "items": items,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_chunk_files(
    processed_dir: Path,
    input_file: Optional[str] = None,
) -> List[Path]:
    if input_file:
        file_path = Path(input_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Input chunk file not found: {file_path}")
        return [file_path]

    # Quét toàn bộ json trong data/processed/[date]/*.json
    files = sorted(
        [
            p for p in processed_dir.glob("*/*.json")
            if p.is_file() and "embeddings" not in p.parts
        ]
    )

    if not files:
        raise FileNotFoundError(f"No chunk json files found under: {processed_dir}")

    return files


def build_output_path(processed_dir: Path, chunk_file: Path) -> Path:
    date_dir = chunk_file.parent.name
    filename = chunk_file.stem
    return processed_dir / "embeddings" / date_dir / f"{filename}_embeddings.json"


def run_embedding(
    settings: Dict[str, Any],
    input_file: Optional[str] = None,
) -> None:
    processed_dir = Path(settings["data"]["processed_dir"])
    model_name = settings["embedding"]["model"]
    batch_size = int(settings["embedding"].get("batch_size", 32))
    device = settings["embedding"].get("device", "cpu")

    chunk_files = resolve_chunk_files(processed_dir, input_file=input_file)

    LOGGER.info("Embedding model : %s", model_name)
    LOGGER.info("Device          : %s", device)
    LOGGER.info("Batch size      : %s", batch_size)
    LOGGER.info("Found %s chunk files", len(chunk_files))

    model = SentenceTransformer(model_name, device=device)

    for chunk_file in chunk_files:
        LOGGER.info("Processing chunk file: %s", chunk_file)

        records = extract_records_from_chunk_file(chunk_file, model_name=model_name)

        if not records:
            LOGGER.warning("No valid chunks in file: %s", chunk_file)
            continue

        texts = [item["text_for_embedding"] for item in records]
        vectors = embed_texts(model=model, texts=texts, batch_size=batch_size)

        for record, vector in zip(records, vectors):
            record["embedding"] = vector
            record["vector_size"] = len(vector)
            del record["text_for_embedding"]

        output_path = build_output_path(processed_dir, chunk_file)
        save_embedding_file(output_path, records)

        LOGGER.info(
            "Saved embeddings | file=%s | output=%s | count=%s",
            chunk_file.name,
            output_path,
            len(records),
        )


def main() -> None:
    setup_logging()
    settings = load_app_settings("config/settings.yaml")

    # Nếu muốn cố định 1 file thì sửa ở đây
    input_file = None

    run_embedding(settings=settings, input_file=input_file)


if __name__ == "__main__":
    main()