"""Upload / Paste router for ingesting documents into the knowledge store."""

from __future__ import annotations

import io
import logging
import uuid
import zipfile
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from openjarvis.connectors.store import KnowledgeStore
from openjarvis.core.config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/connectors/upload", tags=["upload"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_EXTENSIONS = {".txt", ".md", ".csv", ".pdf", ".docx"}
_ALLOWED_CONTENT_TYPES = {
    ".txt": {"text/plain"},
    ".md": {"text/markdown", "text/plain"},
    ".csv": {"text/csv", "application/csv", "text/plain"},
    ".pdf": {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    },
}
_MAX_PASTE_BYTES = 1 * 1024 * 1024
_MAX_UPLOAD_FILES = 8
_MAX_FILE_BYTES = 8 * 1024 * 1024
_MAX_TOTAL_UPLOAD_BYTES = 16 * 1024 * 1024
_MAX_EXTRACTED_CHARS = 2_000_000
_MAX_DOCUMENT_PAGES = 100
_MAX_ARCHIVE_ENTRIES = 1_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 16 * 1024 * 1024


def _chunk_text(text: str, max_chars: int = 1000) -> List[str]:
    """Split *text* into ~max_chars pieces at paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        chunks.append(current.strip())
    # Guard against very large paragraphs that exceed max_chars
    final: List[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            # Find last space within limit
            split_at = chunk.rfind(" ", 0, max_chars)
            if split_at == -1:
                split_at = max_chars
            final.append(chunk[:split_at].strip())
            chunk = chunk[split_at:].strip()
        if chunk:
            final.append(chunk)
    return final


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF using pdfplumber or PyPDF2."""
    # Try pdfplumber first
    try:
        import pdfplumber  # type: ignore[import-untyped]

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > _MAX_DOCUMENT_PAGES:
                raise HTTPException(
                    status_code=413,
                    detail="PDF exceeds the 100 page limit.",
                )
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n\n".join(pages)
        if len(text) > _MAX_EXTRACTED_CHARS:
            raise HTTPException(
                status_code=413,
                detail="Extracted document text is too large.",
            )
        return text
    except ImportError:
        pass

    # Fall back to PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore[import-untyped]

        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > _MAX_DOCUMENT_PAGES:
            raise HTTPException(
                status_code=413,
                detail="PDF exceeds the 100 page limit.",
            )
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(pages)
        if len(text) > _MAX_EXTRACTED_CHARS:
            raise HTTPException(
                status_code=413,
                detail="Extracted document text is too large.",
            )
        return text
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF parsing requires pdfplumber or PyPDF2. "
                "Install one with: pip install pdfplumber"
            ),
        )


def _extract_text_from_docx(data: bytes) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_ARCHIVE_ENTRIES:
                raise HTTPException(
                    status_code=413,
                    detail="DOCX archive contains too many entries.",
                )
            uncompressed = sum(entry.file_size for entry in entries)
            compressed = sum(max(entry.compress_size, 1) for entry in entries)
            if (
                uncompressed > _MAX_ARCHIVE_UNCOMPRESSED_BYTES
                or uncompressed > compressed * 100
            ):
                raise HTTPException(
                    status_code=413,
                    detail="DOCX archive expands beyond safe limits.",
                )
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid DOCX file.") from exc
    try:
        from docx import Document  # type: ignore[import-untyped]

        doc = Document(io.BytesIO(data))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if len(text) > _MAX_EXTRACTED_CHARS:
            raise HTTPException(
                status_code=413,
                detail="Extracted document text is too large.",
            )
        return text
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail=(
                "DOCX parsing requires python-docx. "
                "Install with: pip install python-docx"
            ),
        )


def _get_store() -> KnowledgeStore:
    """Return a KnowledgeStore pointing at the default knowledge DB."""
    db_path = DEFAULT_CONFIG_DIR / "knowledge.db"
    return KnowledgeStore(db_path=db_path)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PasteRequest(BaseModel):
    title: str = ""
    content: str


class IngestResponse(BaseModel):
    chunks_added: int
    source: str = "upload"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest_paste(body: PasteRequest) -> IngestResponse:
    """Ingest pasted text into the knowledge store."""
    text = body.content.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Content is empty")
    if len(body.content.encode("utf-8")) > _MAX_PASTE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Pasted content exceeds the 1 MiB limit.",
        )
    if len(body.title.encode("utf-8")) > 256:
        raise HTTPException(status_code=422, detail="Title is too long.")

    store = _get_store()
    doc_id = str(uuid.uuid4())
    chunks = _chunk_text(text)

    for idx, chunk in enumerate(chunks):
        store.store(
            chunk,
            source="upload",
            doc_type="paste",
            doc_id=doc_id,
            title=body.title or "Pasted text",
            chunk_index=idx,
        )

    logger.info("Ingested %d chunks from pasted text (doc_id=%s)", len(chunks), doc_id)
    return IngestResponse(chunks_added=len(chunks))


@router.post("/ingest/files", response_model=IngestResponse)
async def ingest_files(
    files: List[UploadFile] = File(...),
    title: Optional[str] = Form(None),
) -> IngestResponse:
    """Ingest uploaded files into the knowledge store."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(status_code=413, detail="Too many uploaded files.")
    if title is not None and len(title.encode("utf-8")) > 256:
        raise HTTPException(status_code=422, detail="Title is too long.")

    parsed_documents: List[tuple[str, str, str]] = []
    total_bytes = 0

    for upload in files:
        filename = upload.filename or "untitled"
        ext = ""
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[-1].lower()

        if ext not in _ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=415,
                detail=(f"Unsupported file type: {ext}. Allowed: {allowed}"),
            )
        content_type = (upload.content_type or "").lower()
        if content_type not in _ALLOWED_CONTENT_TYPES[ext]:
            raise HTTPException(
                status_code=415,
                detail="Uploaded file content type does not match its extension.",
            )

        chunks: List[bytes] = []
        file_bytes = 0
        while True:
            chunk = await upload.read(64 * 1024)
            if not chunk:
                break
            file_bytes += len(chunk)
            total_bytes += len(chunk)
            if (
                file_bytes > _MAX_FILE_BYTES
                or total_bytes > _MAX_TOTAL_UPLOAD_BYTES
            ):
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded files exceed the configured size limits.",
                )
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            continue

        # Parse content based on extension
        if ext in (".txt", ".md", ".csv"):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        elif ext == ".pdf":
            text = _extract_text_from_pdf(data)
        elif ext == ".docx":
            text = _extract_text_from_docx(data)
        else:
            continue

        text = text.strip()
        if not text:
            continue
        if len(text) > _MAX_EXTRACTED_CHARS:
            raise HTTPException(
                status_code=413,
                detail="Extracted document text is too large.",
            )
        parsed_documents.append((filename, ext, text))

    # Do not create or mutate the knowledge store until every upload has
    # passed type, size and parser limits.
    store = _get_store()
    total_chunks = 0
    for filename, ext, text in parsed_documents:
        doc_id = str(uuid.uuid4())
        doc_title = title or filename
        chunks = _chunk_text(text)

        for idx, chunk in enumerate(chunks):
            store.store(
                chunk,
                source="upload",
                doc_type=ext.lstrip("."),
                doc_id=doc_id,
                title=doc_title,
                chunk_index=idx,
            )

        total_chunks += len(chunks)
        logger.info(
            "Ingested %d chunks from file %s (doc_id=%s)",
            len(chunks),
            filename,
            doc_id,
        )

    return IngestResponse(chunks_added=total_chunks)
