from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.rag_runtime import hybrid_retrieve

app = FastAPI(
    title="RAG Banking API",
    description="API cho AI agent nghiệp vụ ngân hàng dựa trên RAG/Hybrid RAG",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

class DocumentItem(BaseModel):
    file_name: str
    open_url: str
    download_url: str
    mime_type: Optional[str] = None


class DocumentsResponse(BaseModel):
    items: List[DocumentItem] = []

class AskRequest(BaseModel):
    question: str


class ReferenceItem(BaseModel):
    title: Optional[str] = None
    file_name: Optional[str] = None
    download_url: Optional[str] = None
    page: Optional[int] = None
    snippet: Optional[str] = None
    score: Optional[float] = None


class SourceItem(BaseModel):
    source_file: Optional[str] = None
    score: Optional[float] = None
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AskResponse(BaseModel):
    status: str
    answer: str
    references: List[ReferenceItem] = []
    sources: List[SourceItem] = []
    suggestions: List[str] = []
    meta: Dict[str, Any] = {}


def _encode_rel_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    try:
        raw_bytes = path_str.encode("utf-8")
        return base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")
    except Exception:
        return None


def _decode_rel_path(token: str) -> str:
    padding = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode((token + padding).encode("ascii")).decode("utf-8")


def _safe_join_raw(rel_path: str) -> Optional[Path]:
    if not rel_path:
        return None
    rel_path = rel_path.replace("\\", "/").lstrip("/")
    if rel_path.startswith("data/raw/"):
        rel_path = rel_path[len("data/raw/"):]
    candidate = (RAW_DIR / rel_path).resolve()
    try:
        candidate.relative_to(RAW_DIR.resolve())
    except Exception:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    # fallback by filename only
    name = Path(rel_path).name
    for p in RAW_DIR.rglob(name):
        if p.is_file():
            return p
    return None


def _normalize_reference(item: Dict[str, Any]) -> Dict[str, Any]:
    metadata = item.get("metadata") or {}
    text = (item.get("text") or item.get("snippet") or "").strip()
    snippet = text[:320] + ("..." if len(text) > 320 else "") if text else None
    raw_path = item.get("raw_path") or metadata.get("file_path") or metadata.get("raw_path") or ""
    token = _encode_rel_path(str(raw_path)) if raw_path else None
    file_name = item.get("file_name") or metadata.get("file_name") or Path(str(raw_path)).name or None
    return {
        "title": item.get("title") or (Path(file_name).stem if file_name else None),
        "file_name": file_name,
        "page": item.get("page") or metadata.get("page"),
        "snippet": item.get("snippet") or snippet,
        "score": item.get("score"),
        "download_url": f"/api/download/{token}" if token else None,
    }


def _normalize_sources(raw_sources: Any) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    if not raw_sources:
        return norm

    if isinstance(raw_sources, dict):
        raw_sources = [raw_sources]

    if not isinstance(raw_sources, list):
        return norm

    for item in raw_sources:
        if isinstance(item, str):
            norm.append({
                "source_file": None,
                "score": None,
                "text": item,
                "metadata": {},
            })
            continue

        if not isinstance(item, dict):
            continue

        metadata = item.get("metadata") or {}
        text = item.get("text") or item.get("content") or item.get("page_content") or item.get("snippet")
        score = item.get("score")
        source_file = (
            item.get("source_file")
            or item.get("file_name")
            or metadata.get("source_file")
            or metadata.get("file_name")
            or metadata.get("filename")
        )
        download_url = item.get("download_url")
        norm.append({
            "source_file": source_file,
            "file_name": source_file,
            "download_url": download_url,
            "score": score,
            "text": text,
            "metadata": metadata,
        })

    return norm


def _build_references(raw_sources: Any) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()
    if isinstance(raw_sources, dict):
        raw_sources = [raw_sources]
    for item in raw_sources or []:
        if not isinstance(item, dict):
            continue
        ref = _normalize_reference(item)
        key = (ref.get("file_name"), ref.get("page"), ref.get("snippet"))
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)
    return refs


def _retrieve_only(question: str) -> List[Dict[str, Any]]:
    try:
        docs = hybrid_retrieve(question)
        return _normalize_sources(docs)
    except Exception:
        return []


def _call_generator(question: str) -> Dict[str, Any]:
    try:
        from generation.generate_answer import generate_answer  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Không import được generation.generate_answer.generate_answer: {e}")

    result = generate_answer(question)

    if isinstance(result, str):
        return {"answer": result, "sources": [], "references": []}

    if isinstance(result, dict):
        answer = result.get("answer") or result.get("response") or result.get("result") or ""
        raw_sources = result.get("sources") or result.get("evidence") or result.get("retrieved_docs") or result.get("contexts") or []
        references = result.get("references") or _build_references(raw_sources)
        return {
            "answer": answer,
            "sources": _normalize_sources(raw_sources),
            "references": references,
        }

    return {"answer": str(result), "sources": [], "references": []}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/api/documents", response_model=DocumentsResponse)
def list_documents() -> DocumentsResponse:
    items: List[DocumentItem] = []
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.rglob("*")):
            if not p.is_file():
                continue
            rel_path = p.relative_to(BASE_DIR).as_posix()
            token = _encode_rel_path(rel_path)
            if not token:
                continue
            mime_type, _ = mimetypes.guess_type(str(p))
            url = f"http://localhost:8000/api/download/{token}"
            items.append(DocumentItem(
                file_name=p.name,
                open_url=url,
                download_url=url,
                mime_type=mime_type or "application/octet-stream",
            ))
    return DocumentsResponse(items=items)

@app.get("/api/download/{encoded_path}")
def download_document(encoded_path: str):
    try:
        rel_path = _decode_rel_path(encoded_path)
    except Exception:
        raise HTTPException(status_code=400, detail="Đường dẫn tài liệu không hợp lệ.")

    file_path = _safe_join_raw(rel_path)
    if not file_path:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu gốc để tải.")

    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(path=file_path, filename=file_path.name, media_type=media_type or "application/octet-stream")


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        from core.decision_engine import decide  # type: ignore

        retrieved_docs = _retrieve_only(question) or []
        decision = decide(question, retrieved_docs)

        if decision.get("status") in {"reject", "clarify"} and not retrieved_docs:
            return AskResponse(
                status=decision["status"],
                answer=decision["answer"],
                references=[],
                sources=[],
                suggestions=decision.get("suggestions", []),
                meta={
                    "reason": decision.get("reason"),
                    "domain_score": decision.get("domain_score"),
                    "evidence": {
                        "has_evidence": False,
                        "top_score": 0,
                        "avg_score": 0,
                        "retrieved_count": 0,
                    },
                },
            )

        result = _call_generator(question) or {}
        answer = (result.get("answer") or "").strip()
        final_sources = result.get("sources") or retrieved_docs
        final_references = result.get("references") or _build_references([])

        if not answer:
            answer = "Tôi chưa tạo được câu trả lời phù hợp từ dữ liệu hiện có."

        scores = [float(doc.get("score", 0) or 0) for doc in retrieved_docs if isinstance(doc, dict)]

        evidence = {
            "has_evidence": len(retrieved_docs) > 0,
            "top_score": max(scores) if scores else 0,
            "avg_score": (sum(scores) / len(scores)) if scores else 0,
            "retrieved_count": len(retrieved_docs),
        }

        return AskResponse(
            status="answer",
            answer=answer,
            references=final_references,
            sources=final_sources,
            suggestions=[],
            meta={
                "reason": decision.get("reason"),
                "domain_score": decision.get("domain_score"),
                "evidence": evidence,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý câu hỏi: {e}")


@app.get("/api")
def root() -> Dict[str, str]:
    return {
        "message": "RAG Banking API is running.",
        "docs": "/api/docs",
        "health": "/api/health",
        "ask": "/api/ask"
    }
