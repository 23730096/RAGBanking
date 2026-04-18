from core.setup_logging import setup_logging
from ingestion.load_data import load_data
from ingestion.chunking import (
    chunk_documents,
    enrich_chunks_metadata,
    build_output_path,
    save_chunks
)

setup_logging()

docs = load_data(file_path="data/raw/vcb-FAQ - SMS Banking.pdf")
chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

run_id = "20260418_150000"  # hoặc để None rồi auto generate
chunks = enrich_chunks_metadata(chunks, version="v1", run_id=run_id)

output_path = build_output_path(
    source_file_path="data/raw/vcb-FAQ - SMS Banking.pdf",
    processed_dir="data/processed",
    version="v1",
    run_id=run_id,
    ext="json"
)

save_chunks(chunks, output_path)

print(f"Saved {len(chunks)} chunks to {output_path}")