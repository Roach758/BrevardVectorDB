"""
Vectorize Florida Statutes (Education, Unions, Employment, Criminal Law)
into ChromaDB (collection: 'florida_laws') and SQLite (database: 'laws.db').
"""

import argparse
import hashlib
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure local offline model resolution works
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from sentence_transformers import SentenceTransformer

from db import (
    BASE_DIR,
    CHROMA_LAWS_COLLECTION,
    CHROMA_STORAGE_PATH,
    SQLITE_LAWS_DB_PATH,
    get_chroma_client,
    get_laws_collection,
)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LAWS_DIR = BASE_DIR / "Laws"


def init_sqlite_laws() -> None:
    """Create local SQLite tables for Florida laws metadata."""
    with sqlite3.connect(SQLITE_LAWS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chapters (
                chapter INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                file_path TEXT NOT NULL,
                sections_count INTEGER NOT NULL,
                chunks_count INTEGER NOT NULL,
                processed_date TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sections (
                section TEXT PRIMARY KEY,
                chapter INTEGER NOT NULL,
                category TEXT NOT NULL,
                catchline TEXT NOT NULL,
                body TEXT NOT NULL,
                history TEXT,
                file_path TEXT NOT NULL
            )
            """
        )
        conn.commit()


def chunk_section(sec_num: str, catchline: str, body: str, max_chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Split section text into ~1200-1500 char chunks with context header on each chunk."""
    body = body.strip()
    prefix = f"[Florida Statute s. {sec_num} — {catchline}]\n\n"
    
    if len(prefix) + len(body) <= max_chunk_size:
        return [f"{prefix}{body}"]
        
    chunks: list[str] = []
    start = 0
    body_len = len(body)
    
    # Available body length per chunk after accounting for prefix
    effective_chunk_size = max_chunk_size - len(prefix)
    if effective_chunk_size < 400:
        effective_chunk_size = 1000
        
    while start < body_len:
        end = min(start + effective_chunk_size, body_len)
        if end < body_len:
            split_at = body.rfind("\n\n", start, end)
            if split_at == -1 or split_at < start + 300:
                split_at = body.rfind("\n", start, end)
            if split_at == -1 or split_at < start + 300:
                split_at = body.rfind(". ", start, end)
                if split_at != -1:
                    split_at += 1
            if split_at != -1 and split_at >= start + 300:
                end = split_at
                
        chunk_content = body[start:end].strip()
        if chunk_content:
            chunks.append(f"{prefix}{chunk_content}")
            
        start = end - overlap if end < body_len else body_len
        
    return chunks


def parse_markdown_chapter(file_path: Path, category: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a chapter markdown file into chapter metadata and a list of sections."""
    text = file_path.read_text(encoding="utf-8")
    
    # Extract Chapter number
    m_ch = re.search(r"Chapter_(\d+)\.md", file_path.name)
    ch_num = int(m_ch.group(1)) if m_ch else 0
    
    # Extract Title name from header
    m_title = re.search(r"##\s+CHAPTER\s+\d+\s+—\s+(.*?)(?=\n|$)", text, re.IGNORECASE)
    chapter_title = m_title.group(1).strip() if m_title else f"Chapter {ch_num}"
    
    # Extract sections: ### <sec_num> — <sec_body>
    sec_matches = list(re.finditer(r"\n###\s+([0-9]+\.[0-9]+[a-zA-Z0-9]*)\s+—\s+(.*?)(?=\n###|\Z)", text, re.DOTALL))
    
    sections: list[dict[str, Any]] = []
    total_chunks = 0
    
    for m in sec_matches:
        sec_num = m.group(1).strip()
        raw_body = m.group(2).strip()
        
        # Partition catchline from body
        if ".—" in raw_body:
            catchline, _, body = raw_body.partition(".—")
        elif "—" in raw_body:
            catchline, _, body = raw_body.partition("—")
        else:
            lines = raw_body.split("\n", 1)
            catchline = lines[0]
            body = lines[1] if len(lines) > 1 else ""
            
        catchline = re.sub(r"\s+", " ", catchline).strip()
        
        # Extract history if present
        history = ""
        if "*History: " in body:
            body_part, _, hist_part = body.partition("*History: ")
            body = body_part.strip()
            history = hist_part.rstrip("*").strip()
            
        chunks = chunk_section(sec_num, catchline, body)
        total_chunks += len(chunks)
        
        sections.append({
            "section": sec_num,
            "chapter": ch_num,
            "category": category,
            "catchline": catchline,
            "body": body,
            "history": history,
            "chunks": chunks,
            "file_path": str(file_path.relative_to(BASE_DIR)),
        })
        
    chapter_meta = {
        "chapter": ch_num,
        "category": category,
        "title": chapter_title,
        "file_path": str(file_path.relative_to(BASE_DIR)),
        "sections_count": len(sections),
        "chunks_count": total_chunks,
    }
    
    return chapter_meta, sections


def vectorize_laws(
    clean: bool = False,
    batch_size: int = 64,
    device: str | None = None,
) -> None:
    """Vectorize all Florida law markdown files and store them in ChromaDB and SQLite."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print("=" * 70)
    print("Florida Statutes Vectorization & Database Pipeline")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME} | Device: {device}")
    print(f"SQLite DB: {SQLITE_LAWS_DB_PATH}")
    print(f"ChromaDB Storage: {CHROMA_STORAGE_PATH} (Collection: {CHROMA_LAWS_COLLECTION})")
    print("=" * 70)
    
    init_sqlite_laws()
    chroma_client = get_chroma_client()
    
    if clean:
        print(f"Resetting ChromaDB collection '{CHROMA_LAWS_COLLECTION}' and SQLite tables...")
        try:
            chroma_client.delete_collection(CHROMA_LAWS_COLLECTION)
        except Exception:
            pass
        with sqlite3.connect(SQLITE_LAWS_DB_PATH) as conn:
            conn.execute("DELETE FROM chapters")
            conn.execute("DELETE FROM sections")
            conn.commit()
            
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_LAWS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    
    # Find all chapter files across the 4 categories
    categories = ["Education", "Unions", "Employment", "Criminal"]
    all_chapter_files: list[tuple[str, Path]] = []
    for cat in categories:
        cat_dir = LAWS_DIR / cat
        if cat_dir.exists():
            for f in sorted(cat_dir.glob("Chapter_*.md")):
                all_chapter_files.append((cat, f))
                
    print(f"Found {len(all_chapter_files)} chapter files to vectorize.")
    if not all_chapter_files:
        print("No chapter files found.")
        return
        
    print(f"Loading SentenceTransformer '{EMBEDDING_MODEL_NAME}' on {device}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    
    total_sections_count = 0
    total_chunks_count = 0
    now_utc = datetime.now(timezone.utc).isoformat()
    
    # Process chapter by chapter
    for idx, (cat, file_path) in enumerate(all_chapter_files, 1):
        ch_meta, sections = parse_markdown_chapter(file_path, cat)
        
        # Insert / Update SQLite
        with sqlite3.connect(SQLITE_LAWS_DB_PATH) as conn:
            conn.execute(
                """
                INSERT INTO chapters (chapter, category, title, file_path, sections_count, chunks_count, processed_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chapter) DO UPDATE SET
                    category = excluded.category,
                    title = excluded.title,
                    file_path = excluded.file_path,
                    sections_count = excluded.sections_count,
                    chunks_count = excluded.chunks_count,
                    processed_date = excluded.processed_date
                """,
                (
                    ch_meta["chapter"],
                    ch_meta["category"],
                    ch_meta["title"],
                    ch_meta["file_path"],
                    ch_meta["sections_count"],
                    ch_meta["chunks_count"],
                    now_utc,
                ),
            )
            for s in sections:
                conn.execute(
                    """
                    INSERT INTO sections (section, chapter, category, catchline, body, history, file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(section) DO UPDATE SET
                        chapter = excluded.chapter,
                        category = excluded.category,
                        catchline = excluded.catchline,
                        body = excluded.body,
                        history = excluded.history,
                        file_path = excluded.file_path
                    """,
                    (
                        s["section"],
                        s["chapter"],
                        s["category"],
                        s["catchline"],
                        s["body"],
                        s["history"],
                        s["file_path"],
                    ),
                )
            conn.commit()
            
        # Collect chunks for ChromaDB
        chapter_chunks: list[str] = []
        chapter_ids: list[str] = []
        chapter_metadatas: list[dict[str, Any]] = []
        
        for s in sections:
            for c_idx, chunk_text in enumerate(s["chunks"]):
                cid = f"fs_{s['section'].replace('.', '_')}_{c_idx}"
                chapter_ids.append(cid)
                chapter_chunks.append(chunk_text)
                chapter_metadatas.append({
                    "category": s["category"],
                    "chapter": s["chapter"],
                    "section": s["section"],
                    "catchline": s["catchline"],
                    "file_path": s["file_path"],
                    "chunk_index": c_idx,
                    "total_chunks": len(s["chunks"]),
                    "source": "Florida Statutes (2026 Edition)",
                })
                
        if chapter_chunks:
            # Delete existing chunks for this chapter
            try:
                collection.delete(where={"chapter": ch_meta["chapter"]})
            except Exception:
                pass
                
            embeddings = embedder.encode(
                chapter_chunks,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ).tolist()
            
            collection.add(
                ids=chapter_ids,
                documents=chapter_chunks,
                embeddings=embeddings,
                metadatas=chapter_metadatas,
            )
            
        total_sections_count += len(sections)
        total_chunks_count += len(chapter_chunks)
        
        if idx % 10 == 0 or idx == len(all_chapter_files):
            print(f"[{idx}/{len(all_chapter_files)}] Processed Chapter {ch_meta['chapter']} ({cat}) -> {len(sections)} sections, {len(chapter_chunks)} chunks (Total chunks: {total_chunks_count})")
            
    print("\n" + "=" * 70)
    print("Vectorization & Database Complete!")
    print(f"Total Chapters: {len(all_chapter_files)}")
    print(f"Total Sections Indexed: {total_sections_count}")
    print(f"Total Vector Chunks in ChromaDB: {total_chunks_count}")
    print(f"ChromaDB Collection Count: {collection.count()}")
    print("=" * 70)


def query_laws(query: str, n_results: int = 3) -> None:
    """Test query against the vectorized florida_laws collection."""
    print(f"\n--- Testing Query: '{query}' ---")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    collection = get_laws_collection()
    
    query_vec = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        similarity = 1.0 - dist
        sec = meta.get("section", "N/A")
        catch = meta.get("catchline", "N/A")
        cat = meta.get("category", "N/A")
        print(f"Match #{i} (Score: {similarity:.4f}) | Category: {cat} | F.S. {sec} — {catch}")
        print(f"Snippet: {doc[:220]}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize Florida laws into ChromaDB and SQLite.")
    parser.add_argument("--clean", action="store_true", help="Clean existing collection before indexing")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")
    parser.add_argument("--test-query", type=str, default="", help="Run a test query after indexing")
    args = parser.parse_args()
    
    vectorize_laws(clean=args.clean, batch_size=args.batch_size)
    
    if args.test_query:
        query_laws(args.test_query)
