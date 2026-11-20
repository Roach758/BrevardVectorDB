"""
Unpack & Verify Utility for BrevardVectorDB.

Because GitHub enforces a 100 MB file-size limit on standard repository files,
the 795 MB vector & relational database bundle is stored in `db_archive/` as
numbered split archive parts (`brevard_vector_db.zip.part001`, etc.).

Running this script (or running `search_db.py`, which calls this automatically)
reassembles and extracts:
  - `chroma_storage/` (54,054 embedded vectors across 3 collections)
  - `videos.db`       (80 school board video records & metadata)
  - `laws.db`         (69 Florida Statute chapters / 2,422 sections)
"""

from pathlib import Path
import shutil
import sys
import zipfile

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "db_archive"
CHROMA_DB_FILE = BASE_DIR / "chroma_storage" / "chroma.sqlite3"


def ensure_database_unpacked(force: bool = False) -> bool:
    """Ensure `chroma_storage/`, `videos.db`, and `laws.db` exist on disk."""
    if CHROMA_DB_FILE.exists() and not force:
        return True

    parts = sorted(ARCHIVE_DIR.glob("brevard_vector_db.zip.part*"))
    if not parts:
        print(
            "[!] Database archive parts not found in 'db_archive/' and 'chroma_storage/' is missing.",
            file=sys.stderr,
        )
        return False

    print("======================================================================")
    print(" First-Time Setup: Unpacking BrevardVectorDB from 'db_archive/'...")
    print(f" Found {len(parts)} archive parts. Reassembling & extracting...")
    print("======================================================================")

    temp_zip = BASE_DIR / "_temp_brevard_vector_db.zip"
    try:
        with open(temp_zip, "wb") as out_f:
            for part in parts:
                print(f"  -> Reading {part.name} ({part.stat().st_size / (1024 * 1024):.1f} MB)...")
                with open(part, "rb") as in_f:
                    shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)

        print("  -> Extracting databases to workspace root...")
        with zipfile.ZipFile(temp_zip, "r") as zf:
            zf.extractall(BASE_DIR)

        print("======================================================================")
        print(" [SUCCESS] Database unpacked and ready for queries!")
        print("======================================================================")
        return True
    finally:
        if temp_zip.exists():
            temp_zip.unlink()


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    ok = ensure_database_unpacked(force=force_flag)
    if ok:
        print("Database status: READY")
        print(f"  - ChromaDB : {CHROMA_DB_FILE}")
        print(f"  - Videos DB: {BASE_DIR / 'videos.db'}")
        print(f"  - Laws DB  : {BASE_DIR / 'laws.db'}")
    else:
        sys.exit(1)
