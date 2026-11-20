"""
Vectorize all meeting files (HTML agendas, HTML minutes, and attached PDFs).
Stores chunked documents and normalized embeddings into ChromaDB with metadata
including file_name, file_path, meeting_date, meeting_title, and chunk_index.
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Prevent huggingface hub from trying to ping the internet if offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from db import get_chroma_client

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_MEETINGS_COLLECTION = "meeting_documents"

DATE_PATTERNS = [
    # E.g. "January 21, 2025", "May 06, 2025", "April 8, 2025"
    re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE),
    # E.g. "2025-01-21", "2024-11-19"
    re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
    # E.g. "11-19-24", "04-22-25"
    re.compile(r'\b(\d{1,2})-(\d{1,2})-(\d{2,4})\b'),
]


def extract_meeting_date(folder_name: str, file_name: str, default_date: str = "") -> str:
    """Extract meeting date in YYYY-MM-DD or standard readable string."""
    # Check folder name first (e.g. "April 22_ 2025 - ...")
    combined = f"{folder_name} {file_name}"
    
    # Check "Month Day, Year"
    m_month = DATE_PATTERNS[0].search(combined)
    if m_month:
        month_str, day_str, year_str = m_month.groups()
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year_str}", "%B %d %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return f"{month_str} {day_str}, {year_str}"
            
    # Check YYYY-MM-DD
    m_iso = DATE_PATTERNS[1].search(combined)
    if m_iso:
        return f"{m_iso.group(1)}-{m_iso.group(2)}-{m_iso.group(3)}"

    # Check MM-DD-YY
    m_us = DATE_PATTERNS[2].search(combined)
    if m_us:
        m, d, y = m_us.groups()
        if len(y) == 2:
            y = f"20{y}"
        return f"{y}-{int(m):02d}-{int(d):02d}"

    return default_date or "Unknown Date"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    text_parts: list[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()
                if page_text:
                    text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
            except Exception as pe:
                continue
    except Exception as e:
        print(f"    [Warning] Could not read PDF {pdf_path.name}: {e}")
    return "\n\n".join(text_parts).strip()


def extract_text_from_html(html_path: Path) -> str:
    """Extract clean readable text from HTML agenda/minutes."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        
        # Remove scripts and style tags
        for script_or_style in soup(["script", "style", "noscript", "svg"]):
            script_or_style.decompose()

        text = soup.get_text(separator="\n")
        # Collapse multiple blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines).strip()
    except Exception as e:
        print(f"    [Warning] Could not read HTML {html_path.name}: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping character chunks cleanly along paragraphs or sentences."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            # Look backwards for paragraph break, newline, or sentence end
            split_at = text.rfind("\n\n", start, end)
            if split_at == -1 or split_at < start + 500:
                split_at = text.rfind("\n", start, end)
            if split_at == -1 or split_at < start + 500:
                split_at = text.rfind(". ", start, end)
                if split_at != -1:
                    split_at += 1
            if split_at != -1 and split_at >= start + 500:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else text_len

    return chunks


def get_meetings_collection():
    """Retrieve or create the meeting_documents Chroma collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_MEETINGS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def vectorize_meeting_files(
    target_dir: Path,
    batch_size: int = 32,
    device: str | None = None,
    clean_collection: bool = False,
):
    """Scan directory for PDFs and HTMLs, chunk them, compute embeddings, and save to ChromaDB."""
    target_dir = Path(target_dir).resolve()
    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    # Determine PyTorch device
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"Vectorizing meeting files in: {target_dir}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME} | Device: {device}")
    print("=" * 70)

    collection = get_meetings_collection()
    if clean_collection:
        print(f"Resetting ChromaDB collection '{CHROMA_MEETINGS_COLLECTION}'...")
        client = get_chroma_client()
        client.delete_collection(CHROMA_MEETINGS_COLLECTION)
        collection = client.create_collection(
            name=CHROMA_MEETINGS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    # Find all supported files
    all_files: list[Path] = []
    for ext in ("*.pdf", "*.html", "*.htm"):
        all_files.extend(target_dir.rglob(ext))

    print(f"Found {len(all_files)} total files to process.")
    if not all_files:
        print("No files found.")
        return

    # Load SentenceTransformer model
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    existing_paths: set[str] = set()
    if not clean_collection:
        try:
            raw_meta = collection.get(include=["metadatas"])
            for m in raw_meta.get("metadatas", []):
                if m and "file_path" in m:
                    existing_paths.add(m["file_path"])
            print(f"Found {len(existing_paths)} already-indexed files in ChromaDB (will skip).")
        except Exception as e:
            print(f"Could not load existing paths: {e}")

    total_chunks_stored = 0
    files_processed = 0
    files_skipped = 0

    for file_path in all_files:
        rel_path_str = str(file_path.relative_to(target_dir))
        if rel_path_str in existing_paths:
            files_skipped += 1
            continue

        folder_name = file_path.parent.name
        file_name = file_path.name
        meeting_date = extract_meeting_date(folder_name, file_name)
        
        # Human-readable meeting title from parent folder
        meeting_title = folder_name.replace("_", " ").strip()
        
        # Check if chunks for this specific file already exist
        file_hash = hashlib.md5(str(file_path.relative_to(target_dir)).encode()).hexdigest()[:12]
        file_id_prefix = f"doc_{file_hash}"

        # Extract content
        if file_path.suffix.lower() == ".pdf":
            content = extract_text_from_pdf(file_path)
        else:
            content = extract_text_from_html(file_path)

        if not content:
            files_skipped += 1
            continue

        chunks = chunk_text(content)
        if not chunks:
            files_skipped += 1
            continue

        # Prepare metadata and IDs
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for idx, ch in enumerate(chunks):
            chunk_id = f"{file_id_prefix}_{idx}"
            ids.append(chunk_id)
            metadatas.append(
                {
                    "file_name": file_name,
                    "file_path": str(file_path.relative_to(target_dir)),
                    "meeting_date": meeting_date,
                    "meeting_title": meeting_title,
                    "file_type": file_path.suffix.lower().lstrip("."),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                }
            )

        # Compute embeddings in batches
        embeddings = embedder.encode(
            chunks,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()

        # Remove prior chunks for this file if re-running
        try:
            collection.delete(where={"file_path": str(file_path.relative_to(target_dir))})
        except Exception:
            pass

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        total_chunks_stored += len(chunks)
        files_processed += 1
        if files_processed % 10 == 0 or files_processed == len(all_files):
            print(f"[{files_processed}/{len(all_files)}] Processed: {file_name[:45]} -> {len(chunks)} chunks (Total chunks: {total_chunks_stored})")

    print("\n" + "=" * 70)
    print("Vectorization complete!")
    print(f"Files processed: {files_processed} | Files skipped/empty: {files_skipped}")
    print(f"Total chunks stored in '{CHROMA_MEETINGS_COLLECTION}': {total_chunks_stored}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize meeting files into ChromaDB.")
    parser.add_argument(
        "--dir",
        default="agendas",
        help="Root folder containing meeting documents (default: 'agendas')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SentenceTransformer (default: 32)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop and rebuild the meeting_documents collection",
    )
    args = parser.parse_args()

    vectorize_meeting_files(
        target_dir=Path(args.dir),
        batch_size=args.batch_size,
        clean_collection=args.clean,
    )
