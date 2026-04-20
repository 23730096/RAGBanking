from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generation.generate_answer import generate_answer
from retrieval.retrieve import retrieve
from core.rag_runtime import load_app_settings


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"

app = FastAPI(title="RAG Banking API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list
    dropped_context: Optional[int] = 0


@app.get("/")
def home():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/retrieve")
def api_retrieve(payload: AskRequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return {"question": question, "results": retrieve(question)}


def _handle_ask(payload: AskRequest):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    return generate_answer(question)


@app.post("/api/ask", response_model=AskResponse)
def api_ask(payload: AskRequest):
    return _handle_ask(payload)


@app.post("/ask", response_model=AskResponse)
def ask_alias(payload: AskRequest):
    return _handle_ask(payload)


@app.post("/chat", response_model=AskResponse)
def chat_alias(payload: AskRequest):
    return _handle_ask(payload)


@app.post("/query", response_model=AskResponse)
def query_alias(payload: AskRequest):
    return _handle_ask(payload)


# ================= DOCUMENT DOWNLOAD =================


def _find_file_by_name(file_name: str) -> Optional[Path]:
    settings = load_app_settings()
    data_cfg = settings.get("data", {})

    raw_dir = BASE_DIR / data_cfg.get("raw_dir", "data/raw")
    processed_dir = BASE_DIR / data_cfg.get("processed_dir", "data/processed")

    file_name = file_name.lower()

    for base_dir in [raw_dir, processed_dir]:
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*"):
            if path.is_file() and path.name.lower() == file_name:
                return path

    return None


@app.get("/api/documents/{file_name}")
def download_document(file_name: str):
    safe_name = Path(unquote(file_name)).name
    file_path = _find_file_by_name(safe_name)

    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        file_path,
        filename=file_path.name,
        media_type="application/octet-stream",
    )
