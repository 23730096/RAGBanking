import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("ingestion")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def infer_metadata(file_path: str) -> Dict[str, Any]:
    file_name = file_path.lower()
    metadata = {"source_type": "unstructured"}

    if "vcb" in file_name:
        metadata["bank"] = "vietcombank"

    if "faq" in file_name:
        metadata["doc_type"] = "faq"
    elif "phi" in file_name or "fee" in file_name:
        metadata["doc_type"] = "fee"
    elif "huongdan" in file_name:
        metadata["doc_type"] = "guide"
    else:
        metadata["doc_type"] = "general"

    if "sms" in file_name:
        metadata["topic"] = "sms_banking"
    elif "tai-khoan" in file_name or "tai_khoan" in file_name:
        metadata["topic"] = "account"

    return metadata


def chunk_unstructured(
    text: str,
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    text = clean_text(text)
    metadata_base = infer_metadata(file_path)

    chunks = []
    start = 0
    idx = 1

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "id": f"{file_path}::chunk_{idx}",
                "text": chunk_text,
                "metadata": {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    **metadata_base
                }
            })

        if end >= len(text):
            break

        start = end - chunk_overlap
        idx += 1

    logger.info("Chunked unstructured | file=%s | chunks=%s", file_path, len(chunks))
    return chunks


def chunk_structured(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks = []

    for idx, record in enumerate(records, start=1):
        text_parts = []

        if "question" in record:
            text_parts.append(f"Câu hỏi: {record['question']}")
        if "answer" in record:
            text_parts.append(f"Trả lời: {record['answer']}")

        text = "\n".join(text_parts).strip()
        if not text:
            continue

        chunks.append({
            "id": f"json::chunk_{idx}",
            "text": text,
            "metadata": {
                "source_type": "json",
                "doc_type": record.get("category", "faq"),
                "product": record.get("product"),
                "topic": record.get("topic"),
                "chunk_index": idx
            }
        })

    logger.info("Chunked structured JSON | total=%s", len(chunks))
    return chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict[str, Any]]:
    all_chunks = []

    for doc in documents:
        file_path = doc.get("file_path", "")
        content = doc.get("content")

        if isinstance(content, list):
            logger.info("Detected structured JSON: %s", file_path)
            chunks = chunk_structured(content)
        else:
            logger.info("Detected unstructured file: %s", file_path)
            chunks = chunk_unstructured(
                text=content or "",
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        all_chunks.extend(chunks)

    logger.info("Total chunks created: %s", len(all_chunks))
    return all_chunks


def slugify_filename(file_name: str) -> str:
    name, _ = os.path.splitext(file_name)
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[-\s]+", "_", name)
    return name


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def current_timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def enrich_chunks_metadata(
    chunks: List[Dict[str, Any]],
    version: str = "v1",
    run_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not run_id:
        run_id = generate_run_id()

    created_at = current_timestamp()
    enriched = []

    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        file_name = metadata.get("file_name", "unknown")
        safe_name = slugify_filename(file_name)

        chunk["id"] = f"{safe_name}_{version}_chunk_{idx:04d}"
        metadata["chunk_index"] = idx
        metadata["version"] = version
        metadata["run_id"] = run_id
        metadata["created_at"] = created_at
        chunk["metadata"] = metadata
        enriched.append(chunk)

    logger.info("Chunks metadata enriched | total=%s | run_id=%s", len(enriched), run_id)
    return enriched


def build_output_path(
    source_file_path: str,
    processed_dir: str = "data/processed",
    version: str = "v1",
    run_id: Optional[str] = None,
    ext: str = "json"
) -> str:
    if not run_id:
        run_id = generate_run_id()

    file_name = os.path.basename(source_file_path)
    safe_name = slugify_filename(file_name)

    output_dir = Path(processed_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir / f"{safe_name}_chunks_{version}.{ext}")


def save_chunks(chunks: List[Dict[str, Any]], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Chunks saved | output=%s | total=%s", output_path, len(chunks))