from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    settings = load_settings(settings)
    return settings


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no} in {path}: {exc}") from exc
    return records


def load_chunks(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    if input_path.suffix.lower() == ".jsonl":
        raw = load_jsonl(input_path)
    elif input_path.suffix.lower() == ".json":
        raw = load_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Hỗ trợ nhiều kiểu structure
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        if "chunks" in raw and isinstance(raw["chunks"], list):
            return raw["chunks"]
        if "data" in raw and isinstance(raw["data"], list):
            return raw["data"]

    raise ValueError("Unsupported chunk structure. Expect list or dict with 'chunks'/'data' list.")


def build_chunk_id(chunk: Dict[str, Any], index: int) -> str:
    if chunk.get("chunk_id"):
        return str(chunk["chunk_id"])

    metadata = chunk.get("metadata", {}) or {}
    doc_id = metadata.get("doc_id") or metadata.get("source") or "unknown_doc"
    chunk_index = metadata.get("chunk_index", index)
    text_hash = sha1_text(str(chunk.get("text", "")))[:10]
    return f"{doc_id}_{chunk_index}_{text_hash}"


def prepare_text_for_e5(text: str) -> str:
    """
    Với E5, văn bản passage nên có prefix 'passage: '
    Query sau này sẽ nên dùng prefix 'query: '
    """
    text = normalize_text(text)
    return f"passage: {text}"


def extract_chunk_payload(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        text = normalize_text(text)

        if not text:
            LOGGER.warning("Skip empty chunk at index=%s", idx)
            continue

        metadata = chunk.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"value": metadata}

        chunk_id = build_chunk_id(chunk, idx)

        prepared.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "text_for_embedding": prepare_text_for_e5(text),
                "metadata": metadata,
            }
        )

    return prepared


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> List[List[float]]:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # phù hợp cosine search
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def build_output_records(
    prepared_chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    model_name: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for chunk, vector in zip(prepared_chunks, vectors):
        records.append(
            {
                "embedding_id": f"emb_{chunk['chunk_id']}",
                "chunk_id": chunk["chunk_id"],
                "modality": "text",
                "embedding_model": model_name,
                "vector_size": len(vector),
                "text": chunk["text"],
                "embedding": vector,
                "metadata": chunk["metadata"],
            }
        )

    return records


def save_embeddings_json(output_path: Path, records: List[Dict[str, Any]]) -> None:
    ensure_dir(output_path)
    payload = {
        "modality": "text",
        "count": len(records),
        "items": records,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_paths(settings: Dict[str, Any]) -> Tuple[Path, Path]:
    processed_dir = Path(settings["data"]["processed_dir"])
    input_path = processed_dir / "chunks" / "chunks.json"
    output_path = processed_dir / "embeddings" / "text_embeddings.json"
    return input_path, output_path


def main() -> None:
    setup_logging()

    settings = load_app_settings("config/settings.yaml")

    input_path, output_path = resolve_paths(settings)

    model_name = settings["embedding"]["model"]
    batch_size = int(settings["embedding"].get("batch_size", 32))
    device = settings["embedding"].get("device", "cpu")

    LOGGER.info("===== START TEXT EMBEDDING =====")
    LOGGER.info("Model       : %s", model_name)
    LOGGER.info("Device      : %s", device)
    LOGGER.info("Batch size  : %s", batch_size)
    LOGGER.info("Input path  : %s", input_path)
    LOGGER.info("Output path : %s", output_path)

    chunks = load_chunks(input_path)
    LOGGER.info("Loaded raw chunks: %s", len(chunks))

    prepared_chunks = extract_chunk_payload(chunks)
    LOGGER.info("Prepared valid chunks: %s", len(prepared_chunks))

    if not prepared_chunks:
        LOGGER.warning("No valid chunks found. Stop embedding.")
        return

    model = SentenceTransformer(model_name, device=device)

    texts = [item["text_for_embedding"] for item in prepared_chunks]
    vectors = embed_texts(
        model=model,
        texts=texts,
        batch_size=batch_size,
    )

    records = build_output_records(
        prepared_chunks=prepared_chunks,
        vectors=vectors,
        model_name=model_name,
    )

    save_embeddings_json(output_path, records)

    LOGGER.info("Saved %s text embeddings", len(records))
    LOGGER.info("Vector size = %s", len(records[0]['embedding']) if records else 0)
    LOGGER.info("===== DONE TEXT EMBEDDING =====")


if __name__ == "__main__":
    main()