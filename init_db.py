"""
Standalone initialization script for local SQLite (`videos.db`) and ChromaDB (`./chroma_storage`).
"""

from db import (
    CHROMA_COLLECTION_NAME,
    CHROMA_STORAGE_PATH,
    SQLITE_DB_PATH,
    get_transcripts_collection,
    init_sqlite,
)


def main() -> None:
    init_sqlite()
    print(f"SQLite initialized at: {SQLITE_DB_PATH}")

    collection = get_transcripts_collection()
    print(
        f"ChromaDB initialized at: {CHROMA_STORAGE_PATH} "
        f"(collection='{CHROMA_COLLECTION_NAME}', count={collection.count()})"
    )


if __name__ == "__main__":
    main()
