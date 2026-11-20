"""
BrevardVectorDB Unified Search Tool (for Google Antigravity, Grokbot, and CLI users).

Searches across all three vector collections and two relational SQLite databases:
  1. `transcripts`       (5,454 chunks from 80 Brevard School Board video transcripts)
  2. `meeting_documents` (39,165 chunks from 1,663 agendas, minutes, and attachment PDFs)
  3. `florida_laws`      (9,435 chunks from 69 chapters / 2,422 sections of Florida Statutes)

Supports two search modes:
  - Semantic Search (default if sentence-transformers is installed): conceptual AI matching
  - Keyword Search (`--keyword`): instant exact-phrase scan (works without a GPU or AI models)

Usage examples:
  python search_db.py "cell phone policy"
  python search_db.py "public comment time limit" --collection all --top-k 5
  python search_db.py "1006.07" --collection laws --keyword
  python search_db.py --stats
"""

import argparse
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any

from unpack_db import ensure_database_unpacked

BASE_DIR = Path(__file__).resolve().parent
CHROMA_STORAGE_PATH = BASE_DIR / "chroma_storage"
VIDEOS_DB_PATH = BASE_DIR / "videos.db"
LAWS_DB_PATH = BASE_DIR / "laws.db"

BGE_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

COLLECTIONS_MAP = {
    "transcripts": "transcripts",
    "meetings": "meeting_documents",
    "laws": "florida_laws",
}


def format_timestamp(seconds: float | int | None) -> str:
    """Convert seconds into HH:MM:SS or MM:SS."""
    if seconds is None:
        return "00:00"
    total = max(0, int(round(float(seconds))))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def print_stats() -> None:
    """Print dataset counts and summary statistics."""
    ensure_database_unpacked()
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_STORAGE_PATH))
    print("=" * 76)
    print(" BREVARD VECTOR DB — DATASET OVERVIEW")
    print("=" * 76)
    total_vectors = 0
    for label, col_name in COLLECTIONS_MAP.items():
        try:
            col = client.get_collection(col_name)
            cnt = col.count()
            total_vectors += cnt
            print(f"  [ChromaDB] {col_name:22s} ({label:11s}): {cnt:8,d} vectors")
        except Exception as e:
            print(f"  [ChromaDB] {col_name:22s}: unavailable ({e})")
    print(f"  {'':38s}TOTAL     : {total_vectors:8,d} vectors")
    print("-" * 76)

    if VIDEOS_DB_PATH.exists():
        with sqlite3.connect(VIDEOS_DB_PATH) as conn:
            v_cnt = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
            print(f"  [SQLite]   videos.db                            : {v_cnt:8,d} meetings")

    if LAWS_DB_PATH.exists():
        with sqlite3.connect(LAWS_DB_PATH) as conn:
            c_cnt = conn.execute("SELECT COUNT(*) FROM chapters").fetchone()[0]
            s_cnt = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            print(
                f"  [SQLite]   laws.db                              : {c_cnt:8,d} chapters ({s_cnt:,d} sections)"
            )
    print("=" * 76)


def keyword_search_collection(
    collection: Any, query: str, top_k: int = 5
) -> list[dict[str, Any]]:
    """Perform an exact case-insensitive substring search over a Chroma collection."""
    needle = query.strip().lower()
    total = collection.count()
    batch_size = 2000
    matches: list[dict[str, Any]] = []

    for offset in range(0, total, batch_size):
        batch = collection.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc_id, doc, meta in zip(
            batch.get("ids", []),
            batch.get("documents", []),
            batch.get("metadatas", []),
        ):
            if not doc:
                continue
            occ = doc.lower().count(needle)
            if occ > 0:
                matches.append(
                    {
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta or {},
                        "score": float(occ),
                        "match_type": f"keyword ({occ}x)",
                    }
                )

    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:top_k]


def semantic_search_collection(
    collection: Any, query_embedding: list[float], top_k: int = 5
) -> list[dict[str, Any]]:
    """Perform vector similarity search on a Chroma collection."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = max(0.0, 1.0 - float(dist))
        results.append(
            {
                "id": doc_id,
                "document": doc or "",
                "metadata": meta or {},
                "score": round(similarity, 4),
                "match_type": "semantic",
            }
        )
    return results


def format_result(col_key: str, rank: int, item: dict[str, Any]) -> str:
    """Format a search hit into a clean, readable citation block."""
    meta = item["metadata"]
    doc = item["document"].strip()
    score = item["score"]
    lines = [f"\n--- [{col_key.upper()} #{rank}] (Score: {score} | {item['match_type']}) ---"]

    if col_key == "transcripts":
        title = meta.get("title", "Unknown Video")
        vid = meta.get("video_id", "")
        date = meta.get("upload_date", "Unknown Date")
        start_s = int(float(meta.get("start_time", 0)))
        end_s = int(float(meta.get("end_time", 0)))
        speaker = meta.get("primary_speaker", "Unknown Speaker")
        url = f"https://www.youtube.com/watch?v={vid}&t={start_s}s" if vid else ""
        lines.append(f"Meeting : {title} ({date})")
        lines.append(
            f"Time    : {format_timestamp(start_s)} - {format_timestamp(end_s)} | Speaker: {speaker}"
        )
        if url:
            lines.append(f"Watch   : {url}")
    elif col_key == "meetings":
        title = meta.get("meeting_title", meta.get("folder_name", "Meeting Document"))
        date = meta.get("meeting_date", "Unknown Date")
        fname = meta.get("file_name", "Unknown File")
        ftype = meta.get("file_type", "")
        lines.append(f"Meeting : {title} ({date})")
        lines.append(f"File    : {fname} ({ftype})")
    elif col_key == "laws":
        sec = meta.get("section", "")
        chap = meta.get("chapter", "")
        cat = meta.get("category", "")
        catchline = meta.get("catchline", "")
        lines.append(f"Statute : Florida Statute s. {sec} (Chapter {chap} — {cat})")
        lines.append(f"Title   : {catchline}")

    snippet = doc if len(doc) <= 900 else doc[:900] + " ..."
    lines.append(f"Excerpt :\n{snippet}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search BrevardVectorDB across Video Transcripts, Meeting Agendas/PDFs, and Florida Laws."
    )
    parser.add_argument("query", nargs="?", default="", help="Search question or keyword phrase")
    parser.add_argument(
        "--collection",
        "-c",
        choices=["all", "transcripts", "meetings", "laws"],
        default="all",
        help="Which collection to search (default: all)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of top results per collection (default: 5)",
    )
    parser.add_argument(
        "--keyword",
        action="store_true",
        help="Force exact keyword/phrase search instead of semantic AI embedding search",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database statistics and exit",
    )
    args = parser.parse_args()

    ensure_database_unpacked()

    if args.stats or not args.query.strip():
        print_stats()
        if not args.query.strip():
            print("\nTip: Pass a search query, e.g.:")
            print('  python search_db.py "cell phone policy"')
        return

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_STORAGE_PATH))
    target_keys = (
        list(COLLECTIONS_MAP.keys()) if args.collection == "all" else [args.collection]
    )

    use_keyword = args.keyword
    query_embedding: list[float] | None = None

    if not use_keyword:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(BGE_MODEL_NAME)
            prefixed = f"{BGE_QUERY_PREFIX}{args.query.strip()}"
            query_embedding = model.encode(
                [prefixed], normalize_embeddings=True
            )[0].tolist()
        except Exception as e:
            print(
                f"[Info] Semantic model unavailable ({e}); falling back to fast keyword search."
            )
            use_keyword = True

    print("=" * 76)
    print(f" SEARCH QUERY: {args.query!r} | Mode: {'KEYWORD' if use_keyword else 'SEMANTIC AI'}")
    print("=" * 76)

    for key in target_keys:
        col_name = COLLECTIONS_MAP[key]
        try:
            col = client.get_collection(col_name)
        except Exception as e:
            print(f"\n[!] Collection {col_name!r} not found: {e}")
            continue

        if use_keyword or query_embedding is None:
            hits = keyword_search_collection(col, args.query, top_k=args.top_k)
        else:
            hits = semantic_search_collection(col, query_embedding, top_k=args.top_k)

        print(f"\n>>> Collection: {col_name} ({len(hits)} top matches)")
        if not hits:
            print("    No matching chunks found.")
        for idx, hit in enumerate(hits, 1):
            print(format_result(key, idx, hit))


if __name__ == "__main__":
    main()
