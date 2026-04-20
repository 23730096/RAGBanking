from pathlib import Path

from core.setup_logging import setup_logging
from ingestion.load_data import load_data
from ingestion.chunking import (
    chunk_documents,
    enrich_chunks_metadata,
    build_output_path,
    save_chunks
)

setup_logging()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = "data/processed"

# ===== Tạo run_id chung cho cả batch =====
run_id = None  # để None cho auto, hoặc fix nếu muốn

all_files = list(RAW_DIR.glob("**/*"))

total_chunks = 0

for file_path in all_files:
    if not file_path.is_file():
        continue

    print(f"\n📄 Processing: {file_path}")

    try:
        # ===== LOAD =====
        docs = load_data(file_path=str(file_path))

        # ===== CHUNK =====
        chunks = chunk_documents(
            docs,
            chunk_size=1000,
            chunk_overlap=200
        )

        if not chunks:
            print("⚠️ No chunks created, skip")
            continue

        # ===== ENRICH =====
        chunks = enrich_chunks_metadata(
            chunks,
            version="v1",
            run_id=run_id
        )

        # ===== BUILD OUTPUT PATH =====
        output_path = build_output_path(
            source_file_path=str(file_path),
            processed_dir=PROCESSED_DIR,
            version="v1",
            run_id=chunks[0]["metadata"]["run_id"],  # lấy run_id thực tế
            ext="json"
        )

        # ===== SAVE =====
        save_chunks(chunks, output_path)

        print(f"✅ Saved {len(chunks)} chunks → {output_path}")
        total_chunks += len(chunks)

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

print("\n==============================")
print(f"🎯 DONE - Total chunks: {total_chunks}")