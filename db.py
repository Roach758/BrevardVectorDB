"""
Local database layer for SQLite (video metadata) and ChromaDB (transcript vectors).
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

BASE_DIR = Path(__file__).resolve().parent
SQLITE_DB_PATH = BASE_DIR / "videos.db"
CHROMA_STORAGE_PATH = BASE_DIR / "chroma_storage"
CHROMA_COLLECTION_NAME = "transcripts"
CHROMA_LAWS_COLLECTION = "florida_laws"
SQLITE_LAWS_DB_PATH = BASE_DIR / "laws.db"


def get_sqlite_connection() -> sqlite3.Connection:
    """Open a SQLite connection with row dictionary access enabled."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_sqlite() -> None:
    """Create or migrate the local SQLite `videos` table."""
    with get_sqlite_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                url TEXT NOT NULL,
                channel TEXT NOT NULL DEFAULT '',
                upload_date TEXT NOT NULL DEFAULT '',
                duration INTEGER NOT NULL DEFAULT 0,
                processed_date TEXT NOT NULL
            )
            """
        )
        # Safely migrate older videos.db schemas that predate channel/upload_date/duration.
        existing_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(videos)").fetchall()
        }
        migrations = {
            "channel": "TEXT NOT NULL DEFAULT ''",
            "upload_date": "TEXT NOT NULL DEFAULT ''",
            "duration": "INTEGER NOT NULL DEFAULT 0",
        }
        for col_name, col_def in migrations.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col_name} {col_def}")
        conn.commit()


def get_chroma_client() -> Any:
    """Return a persistent local ChromaDB client."""
    CHROMA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_STORAGE_PATH))


def get_transcripts_collection(
    client: Any = None,
) -> chromadb.Collection:
    """Get or create the local ChromaDB collection for video transcripts."""
    chroma_client = client or get_chroma_client()
    return chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_laws_collection(
    client: Any = None,
) -> chromadb.Collection:
    """Get or create the local ChromaDB collection for Florida laws."""
    chroma_client = client or get_chroma_client()
    return chroma_client.get_or_create_collection(
        name=CHROMA_LAWS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def init_databases() -> None:
    """Initialize both SQLite and ChromaDB local storage."""
    init_sqlite()
    get_transcripts_collection()


def upsert_video_metadata(
    video_id: str,
    title: str,
    url: str,
    channel: str = "",
    upload_date: str = "",
    duration: int = 0,
) -> str:
    """Insert or update video metadata in SQLite and return the UTC timestamp."""
    processed_date = datetime.now(timezone.utc).isoformat()
    with get_sqlite_connection() as conn:
        conn.execute(
            """
            INSERT INTO videos (video_id, title, url, channel, upload_date, duration, processed_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                title = excluded.title,
                url = excluded.url,
                channel = excluded.channel,
                upload_date = excluded.upload_date,
                duration = excluded.duration,
                processed_date = excluded.processed_date
            """,
            (video_id, title, url, channel, upload_date, int(duration or 0), processed_date),
        )
        conn.commit()
    return processed_date


def get_videos_by_ids(video_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch video metadata rows keyed by `video_id`."""
    if not video_ids:
        return {}
    placeholders = ",".join("?" for _ in video_ids)
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT video_id, title, url, channel, upload_date, duration, processed_date
            FROM videos
            WHERE video_id IN ({placeholders})
            """,
            video_ids,
        ).fetchall()
    return {row["video_id"]: dict(row) for row in rows}


def list_all_videos() -> list[dict[str, Any]]:
    """Return all ingested videos ordered by most recently processed."""
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            """
            SELECT video_id, title, url, channel, upload_date, duration, processed_date
            FROM videos
            ORDER BY processed_date DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def replace_video_chunks(
    video_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """Remove any prior chunks for `video_id` and store the new chunks in ChromaDB."""
    collection = get_transcripts_collection()
    # Delete existing chunks for this video so re-ingestion stays idempotent.
    collection.delete(where={"video_id": video_id})

    if not chunks:
        return

    ids = [f"{video_id}_chunk_{idx}" for idx in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
