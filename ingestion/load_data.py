import os
from typing import List, Optional
import logging

try:
    import fitz  # PyMuPDF
except:
    fitz = None

try:
    from docx import Document
except:
    Document = None

logger = logging.getLogger("ingestion")


def load_txt(file_path: str) -> str:
    logger.debug("Loading TXT file: %s", file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    if not fitz:
        raise ImportError("PyMuPDF not installed. pip install pymupdf")

    logger.debug("Loading PDF file: %s", file_path)
    text = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        text.append(page_text)

    logger.debug("PDF loaded: %s pages", len(doc))
    return "\n".join(text)


def load_docx(file_path: str) -> str:
    if not Document:
        raise ImportError("python-docx not installed. pip install python-docx")

    logger.debug("Loading DOCX file: %s", file_path)
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_file(file_path: str) -> str:
    """
    Load file based on extension
    """
    if not os.path.exists(file_path):
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(file_path)

    ext = os.path.splitext(file_path)[1].lower()

    logger.info("Loading file: %s", file_path)

    try:
        if ext == ".txt":
            return load_txt(file_path)
        elif ext == ".pdf":
            return load_pdf(file_path)
        elif ext == ".docx":
            return load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception:
        logger.exception("Failed to load file: %s", file_path)
        raise



def load_files(file_paths: List[str]) -> List[dict]:
    """
    Load multiple files and return structured data
    """
    results = []

    for path in file_paths:
        try:
            content = load_file(path)
            results.append({
                "file_path": path,
                "content": content
            })
        except Exception:
            logger.warning("Skipping file due to error: %s", path)

    logger.info("Loaded %s/%s files", len(results), len(file_paths))
    return results



def load_from_config(config: dict) -> List[dict]:
    """
    Load files based on config:
    config example:
    {
        "data": {
            "raw_dir": "data/raw",
            "files": ["a.pdf", "b.txt"]
        }
    }
    """

    raw_dir = config.get("data", {}).get("raw_dir", "")
    files = config.get("data", {}).get("files", [])

    logger.info("Loading files from config | raw_dir=%s", raw_dir)

    file_paths = []

    if files:
        # Load specific files
        file_paths = [os.path.join(raw_dir, f) for f in files]
    else:
        # Load all files in directory
        for root, _, filenames in os.walk(raw_dir):
            for name in filenames:
                file_paths.append(os.path.join(root, name))

    return load_files(file_paths)



def load_data(
    file_path: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    config: Optional[dict] = None
) -> List[dict]:
    """
    Flexible loader:
    - load_data(file_path="a.pdf")
    - load_data(file_paths=[...])
    - load_data(config=config)
    """

    if file_path:
        logger.info("Loading single file via input")
        return [{
            "file_path": file_path,
            "content": load_file(file_path)
        }]

    if file_paths:
        logger.info("Loading multiple files via input")
        return load_files(file_paths)

    if config:
        logger.info("Loading files via config")
        return load_from_config(config)

    raise ValueError("No input provided for loading data")

if __name__ == "__main__":
    load_data()