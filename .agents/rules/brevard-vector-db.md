---
description: Instructions for Google Antigravity and Grokbot on how to query BrevardVectorDB (School Board video transcripts, meeting agendas/PDFs, and Florida Statutes).
---

# BrevardVectorDB Agent Instructions

Whenever a user asks a question about **Brevard Public Schools**, **School Board Meetings**, **Work Sessions**, **Agendas / Minutes / Attachments**, **Board Policies**, or **Florida Statutes / Laws**:

1. **First-Time Unpack (Automatic):**
   If `chroma_storage/chroma.sqlite3` does not exist yet (for example, right after cloning from GitHub), run:
   ```bash
   python unpack_db.py
   ```
   *(Note: `search_db.py` also calls `ensure_database_unpacked()` automatically on startup.)*

2. **How to Search the Database:**
   Use `search_db.py` to query across all three ChromaDB vector collections (`transcripts`, `meeting_documents`, `florida_laws`) and two SQLite databases (`videos.db`, `laws.db`):
   ```bash
   # Search all 3 collections (Video Transcripts + Meeting Agendas/PDFs + Florida Laws)
   python search_db.py "your search query here" --top-k 5

   # Search a specific collection ('transcripts', 'meetings', or 'laws')
   python search_db.py "cell phone policy" --collection transcripts --top-k 8

   # Exact keyword/phrase search (instant, no GPU or embedding model needed)
   python search_db.py "1006.07" --collection laws --keyword
   ```

3. **Direct SQLite Lookups:**
   - **`videos.db`**: Table `videos` (`video_id`, `title`, `url`, `channel`, `upload_date`, `duration`, `processed_date`) — contains all 80 Brevard School Board meetings & work sessions (2024–2026).
   - **`laws.db`**: Tables `chapters` and `sections` (`section`, `chapter`, `category`, `catchline`, `body`, `history`, `file_path`) — contains 69 chapters and 2,422 full sections of Florida Statutes covering Education, Unions, Employment, and Criminal Law.

4. **Citation Standard:**
   Always cite your findings clearly for the user:
   - For **Video Transcripts**: Include the Meeting Title, Date, Speaker label, timestamp range (`MM:SS`), and clickable YouTube timestamp URL (`https://www.youtube.com/watch?v=<video_id>&t=<start_seconds>s`).
   - For **Meeting Documents**: Include the Meeting Date, Meeting Title, and Document Filename (`agenda.html`, `minutes.html`, or attachment PDF name).
   - For **Florida Statutes**: Include the Statute Section number (e.g., `Fla. Stat. s. 1006.07`) and Catchline Title.
