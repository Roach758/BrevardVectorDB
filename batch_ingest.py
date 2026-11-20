"""
Batch processor for Project Cicero video ingestion.
Reads URLs from Videos_Starting_in_2024.txt, checks if already ingested,
runs through main.ingest_video sequentially, and prints real-time status & elapsed timing.
"""

import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure local offline model resolution works without HF pings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sqlite3
from db import init_databases, SQLITE_DB_PATH
from main import IngestRequest, ingest_video


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def extract_video_id(url: str) -> str:
    """Extract YouTube 11-char video ID from URL."""
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    return match.group(1) if match else url.strip()


def get_already_processed_ids() -> set[str]:
    """Return set of video_ids already recorded in SQLite."""
    if not SQLITE_DB_PATH.exists():
        return set()
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            rows = conn.execute("SELECT video_id FROM videos").fetchall()
            return {row[0] for row in rows}
    except Exception:
        return set()


def main():
    list_file = Path("Videos_Starting_in_2024.txt")
    if not list_file.exists():
        print(f"Error: {list_file} not found!")
        sys.exit(1)

    with open(list_file, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    total_urls = len(urls)
    print("=" * 80)
    print(f"PROJECT CICERO BATCH INGESTION: {total_urls} VIDEOS QUEUED")
    print("=" * 80)

    init_databases()
    processed_ids = get_already_processed_ids()
    print(f"Already processed in videos.db: {len(processed_ids)} videos")

    overall_start_time = time.time()
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, url in enumerate(urls, 1):
        vid_id = extract_video_id(url)
        elapsed_overall = time.time() - overall_start_time

        print("\n" + "-" * 80)
        print(f"[{idx}/{total_urls}] Video ID: {vid_id} | URL: {url}")
        print(f"Total Session Elapsed Time: {format_duration(elapsed_overall)}")
        print("-" * 80)

        if vid_id in processed_ids:
            print(f"-> Skipping: {vid_id} is already processed in videos.db.")
            skipped_count += 1
            continue

        item_start_time = time.time()
        print(f"-> [{datetime.now().strftime('%H:%M:%S')}] Starting ingestion pipeline (Download -> Whisper -> WavLM -> Embed -> Store)...", flush=True)

        try:
            req = IngestRequest(url=url, diarization_mode="wavlm_local")
            result = ingest_video(req)

            item_duration = time.time() - item_start_time
            print(f"-> [SUCCESS] Title: '{result.title}'", flush=True)
            print(f"-> Segments: {result.segments_count} | Chunks: {result.chunks_count} | Speakers: {len(result.speakers_detected)}", flush=True)
            print(f"-> Time spent on this video: {format_duration(item_duration)}", flush=True)

            processed_ids.add(vid_id)
            success_count += 1

        except KeyboardInterrupt:
            print("\nBatch processing interrupted by user.")
            break
        except Exception as e:
            item_duration = time.time() - item_start_time
            print(f"-> [FAILED] Error after {format_duration(item_duration)}: {e}", flush=True)
            failed_count += 1

    total_duration = time.time() - overall_start_time
    print("\n" + "=" * 80)
    print("BATCH RUN COMPLETE")
    print(f"Processed: {success_count} | Skipped (already done): {skipped_count} | Failed: {failed_count}")
    print(f"Total time elapsed: {format_duration(total_duration)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
