"""
FastAPI backend for the local YouTube RAG application.

Designed for 6GB-8GB VRAM consumer GPUs (e.g., RTX 5050):
Whisper, WavLM diarization, and SentenceTransformer are never kept in VRAM at the same time.
"""

from bisect import bisect_right
from collections import deque
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Import vram_utils BEFORE faster_whisper so PyTorch's CUDA/cuBLAS DLLs are registered first.
from vram_utils import get_device, purge_vram

from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import yt_dlp

from db import (
    get_transcripts_collection,
    get_videos_by_ids,
    init_databases,
    list_all_videos,
    replace_video_chunks,
    upsert_video_metadata,
)
from diarization import assign_speakers, load_audio_16k_mono

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_LOG_BUFFER: deque[str] = deque(maxlen=250)


class _RingBufferLogHandler(logging.Handler):
    """Capture recent backend log lines in memory for the UI console."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _LOG_BUFFER.append(self.format(record))
        except Exception:
            pass


_ring_handler = _RingBufferLogHandler()
_ring_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(_ring_handler)

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Serialize GPU operations so concurrent requests never load two models into VRAM.
_gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure local databases and temp directory exist on startup."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    init_databases()
    logger.info("Local databases initialized. Device=%s", get_device())
    yield


app = FastAPI(
    title="Project Cicero: Become Your Own Statesman",
    description="VRAM-optimized local ingestion and semantic search for civic proceedings and YouTube videos.",
    version="1.2.0",
    lifespan=lifespan,
)


class IngestRequest(BaseModel):
    url: str = Field(..., min_length=5, description="YouTube video URL")
    diarization_mode: str = Field(
        default="wavlm_local",
        description="Speaker diarization mode: 'wavlm_local' (token-free GPU) or 'off'",
    )


class IngestResponse(BaseModel):
    video_id: str
    title: str
    url: str
    channel: str
    upload_date: str
    duration: int
    speakers_detected: list[str]
    segments_count: int
    chunks_count: int
    processed_date: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query or exact phrase")
    mode: str = Field(
        default="top_k",
        description="Search mode: 'top_k' (best N), 'semantic_all' (all chunks >= min_score), or 'keyword' (exhaustive exact phrase scan)",
    )
    top_k: int = Field(default=5, ge=1, le=200)
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    video_id: str | None = Field(default=None, description="Optional video_id filter")
    speaker: str | None = Field(default=None, description="Optional speaker filter (e.g., 'Speaker 1')")


class SearchResult(BaseModel):
    video_id: str
    title: str
    url: str
    timestamped_url: str
    channel: str
    upload_date: str
    duration: int
    start_time: float
    end_time: float
    speakers: str
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    mode: str
    total_matches: int
    results: list[SearchResult]


def _format_upload_date(raw_date: str | None) -> str:
    """Convert yt-dlp YYYYMMDD string to YYYY-MM-DD when possible."""
    if not raw_date:
        return ""
    cleaned = raw_date.strip()
    if len(cleaned) == 8 and cleaned.isdigit():
        return f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]}"
    return cleaned


def _download_audio(youtube_url: str) -> dict:
    """Download audio-only track to ./temp using yt-dlp and return video metadata + path."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(TEMP_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        audio_path = Path(ydl.prepare_filename(info))

    video_id = info["id"]
    return {
        "video_id": video_id,
        "title": info.get("title") or f"YouTube Video ({video_id})",
        "url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        "channel": info.get("channel") or info.get("uploader") or "",
        "upload_date": _format_upload_date(info.get("upload_date")),
        "duration": int(info.get("duration") or 0),
        "audio_path": audio_path,
    }


def _chunk_transcript_with_metadata(
    segments: list[dict],
    video_meta: dict,
) -> tuple[list[str], list[dict]]:
    """
    Split speaker-labeled transcript into ~300-500 token chunks on CPU while
    preserving exact start/end timestamps, speaker labels, and video metadata.
    """
    if not segments:
        return [], []

    joined_parts: list[str] = []
    char_starts: list[int] = []
    valid_segments: list[dict] = []
    cursor = 0
    prev_speaker = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker", "Speaker 1")
        # Prefix with [Speaker X]: whenever the active speaker changes.
        formatted = f"[{speaker}]: {text}" if speaker != prev_speaker else text
        prev_speaker = speaker

        char_starts.append(cursor)
        joined_parts.append(formatted)
        valid_segments.append(seg)
        cursor += len(formatted) + 1  # +1 for joining space

    full_transcript = " ".join(joined_parts)
    if not full_transcript:
        return [], []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        add_start_index=True,
    )
    docs = splitter.create_documents([full_transcript])

    chunk_texts: list[str] = []
    metadatas: list[dict] = []

    for chunk_idx, doc in enumerate(docs):
        start_char = doc.metadata.get("start_index", 0)
        end_char = start_char + len(doc.page_content)

        first_seg_idx = max(0, bisect_right(char_starts, start_char) - 1)
        last_seg_idx = max(first_seg_idx, bisect_right(char_starts, max(start_char, end_char - 1)) - 1)

        start_time = float(valid_segments[first_seg_idx]["start"])
        end_time = float(valid_segments[last_seg_idx]["end"])

        # Collect unique speakers appearing in this chunk in order of appearance.
        chunk_speakers: list[str] = []
        for s in valid_segments[first_seg_idx : last_seg_idx + 1]:
            sp = s.get("speaker", "Speaker 1")
            if sp not in chunk_speakers:
                chunk_speakers.append(sp)

        chunk_texts.append(doc.page_content)
        metadatas.append(
            {
                "video_id": video_meta["video_id"],
                "title": video_meta["title"],
                "channel": video_meta["channel"],
                "upload_date": video_meta["upload_date"],
                "duration": int(video_meta["duration"]),
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "speakers": ", ".join(chunk_speakers),
                "primary_speaker": chunk_speakers[0] if chunk_speakers else "Speaker 1",
                "chunk_index": chunk_idx,
            }
        )

    return chunk_texts, metadatas


def _build_timestamped_url(base_url: str, start_time: float) -> str:
    """Append `t=<seconds>s` to a YouTube URL cleanly."""
    seconds = max(0, int(start_time))
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)
    query_params["t"] = [f"{seconds}s"]
    new_query = urlencode(query_params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


@app.get("/health")
def health_check() -> dict:
    """Return service health and detected compute device."""
    return {"status": "ok", "device": get_device()}


@app.get("/logs")
def get_logs() -> dict:
    """Return recent backend pipeline and VRAM log entries."""
    return {"logs": list(_LOG_BUFFER)}


@app.get("/videos")
def get_videos() -> list[dict]:
    """List all ingested videos from SQLite."""
    return list_all_videos()


@app.post("/ingest", response_model=IngestResponse)
def ingest_video(request: IngestRequest) -> IngestResponse:
    """
    Sequentially download, transcribe, diarize speakers, chunk, vectorize, and store.
    Strictly purges VRAM between Whisper, WavLM diarizer, and SentenceTransformer.
    """
    audio_path: Path | None = None

    with _gpu_lock:
        try:
            # 1. Download audio-only track + video metadata to ./temp
            logger.info("Downloading audio for URL: %s", request.url)
            video_meta = _download_audio(request.url)
            audio_path = video_meta["audio_path"]
            video_id = video_meta["video_id"]

            device = get_device()
            # RTX 5050 (Blackwell) cuBLAS does not support ctranslate2 int8_float16; float16 runs cleanly.
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16") if device == "cuda" else "int8"

            # Decode audio once to 16 kHz mono float32 (avoids faster-whisper/av>=19 metadata_errors bug)
            waveform = load_audio_16k_mono(audio_path)

            # 2. Load faster-whisper onto GPU (int8_float16 for 6GB/8GB VRAM efficiency)
            # 3. Transcribe audio and retain segment start/end timestamps
            segments: list[dict] = []
            whisper_model = None
            try:
                logger.info(
                    "Loading faster-whisper (%s, device=%s, compute_type=%s)",
                    WHISPER_MODEL_NAME,
                    device,
                    compute_type,
                )
                whisper_model = WhisperModel(
                    WHISPER_MODEL_NAME,
                    device=device,
                    compute_type=compute_type,
                )
                raw_segments, _ = whisper_model.transcribe(
                    waveform,
                    vad_filter=True,
                )
                for seg in raw_segments:
                    text = seg.text.strip()
                    if text:
                        segments.append(
                            {
                                "start": float(seg.start),
                                "end": float(seg.end),
                                "text": text,
                            }
                        )
            finally:
                # 4. Purge VRAM: delete Whisper model, run gc.collect(), and flush CUDA cache
                # BEFORE loading the speaker diarization model.
                purge_vram(whisper_model, label="faster-whisper")
                whisper_model = None

            if not segments:
                raise HTTPException(
                    status_code=422,
                    detail="No speech segments were detected in the audio.",
                )

            # 5. Diarize speakers sequentially on GPU (token-free WavLM, purges its own VRAM)
            if request.diarization_mode != "off":
                segments = assign_speakers(audio_path, segments, device, waveform=waveform)
            else:
                for seg in segments:
                    seg["speaker"] = "Speaker 1"
            del waveform
            speakers_detected = list(dict.fromkeys(seg["speaker"] for seg in segments))

            # 6. Chunking (CPU): split speaker-labeled transcript into ~300-500 token chunks
            chunk_texts, metadatas = _chunk_transcript_with_metadata(segments, video_meta)
            logger.info("Created %d transcript chunks on CPU.", len(chunk_texts))

            # 7. Load SentenceTransformer onto GPU & vectorize all chunks
            embeddings: list[list[float]] = []
            embedder = None
            try:
                logger.info(
                    "Loading SentenceTransformer (%s) on %s",
                    EMBEDDING_MODEL_NAME,
                    device,
                )
                embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
                vectors = embedder.encode(
                    chunk_texts,
                    normalize_embeddings=True,
                    batch_size=32,
                    show_progress_bar=False,
                )
                embeddings = vectors.tolist()
            finally:
                # 8. Purge VRAM again: flush SentenceTransformer from GPU memory
                purge_vram(embedder, label="SentenceTransformer")
                embedder = None

            # Persist metadata in SQLite and vectors in ChromaDB
            processed_date = upsert_video_metadata(
                video_id=video_id,
                title=video_meta["title"],
                url=video_meta["url"],
                channel=video_meta["channel"],
                upload_date=video_meta["upload_date"],
                duration=video_meta["duration"],
            )
            replace_video_chunks(video_id, chunk_texts, embeddings, metadatas)

            return IngestResponse(
                video_id=video_id,
                title=video_meta["title"],
                url=video_meta["url"],
                channel=video_meta["channel"],
                upload_date=video_meta["upload_date"],
                duration=video_meta["duration"],
                speakers_detected=speakers_detected,
                segments_count=len(segments),
                chunks_count=len(chunk_texts),
                processed_date=processed_date,
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Ingestion failed for URL: %s", request.url)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if audio_path and audio_path.exists():
                try:
                    audio_path.unlink()
                except OSError:
                    logger.warning("Could not remove temp audio file: %s", audio_path)


def _build_search_results(
    documents: list[str],
    metadatas: list[dict],
    scores: list[float],
    speaker_filter: str | None = None,
) -> list[SearchResult]:
    """Join ChromaDB hits with SQLite metadata and build SearchResult items."""
    video_ids = list({meta["video_id"] for meta in metadatas if "video_id" in meta})
    videos_by_id = get_videos_by_ids(video_ids)
    speaker_needle = speaker_filter.strip().lower() if speaker_filter else ""

    results: list[SearchResult] = []
    for text, meta, score in zip(documents, metadatas, scores):
        chunk_speakers = meta.get("speakers", "")
        if speaker_needle and speaker_needle not in chunk_speakers.lower():
            continue

        vid = meta["video_id"]
        start_time = float(meta.get("start_time", 0.0))
        end_time = float(meta.get("end_time", start_time))
        video_info = videos_by_id.get(
            vid,
            {
                "title": meta.get("title") or f"YouTube Video ({vid})",
                "url": f"https://www.youtube.com/watch?v={vid}",
                "channel": meta.get("channel", ""),
                "upload_date": meta.get("upload_date", ""),
                "duration": int(meta.get("duration", 0)),
            },
        )
        base_url = video_info["url"]

        results.append(
            SearchResult(
                video_id=vid,
                title=video_info["title"],
                url=base_url,
                timestamped_url=_build_timestamped_url(base_url, start_time),
                channel=video_info.get("channel") or meta.get("channel", ""),
                upload_date=video_info.get("upload_date") or meta.get("upload_date", ""),
                duration=int(video_info.get("duration") or meta.get("duration", 0)),
                start_time=start_time,
                end_time=end_time,
                speakers=chunk_speakers,
                score=round(score, 4),
                text=text,
            )
        )
    return results


@app.post("/search", response_model=SearchResponse)
def search_transcripts(request: SearchRequest) -> SearchResponse:
    """
    Search transcripts using one of three modes:
    - 'top_k': top-N most semantically similar chunks (default).
    - 'semantic_all': EVERY chunk across the database meeting `min_score`.
    - 'keyword': exhaustive case-insensitive phrase scan across all chunks (no GPU needed).
    """
    collection = get_transcripts_collection()
    total_count = collection.count()
    if total_count == 0:
        return SearchResponse(query=request.query, mode=request.mode, total_matches=0, results=[])

    where_filter = {"video_id": request.video_id} if request.video_id else None

    # Mode 1: Exhaustive keyword/phrase scan (finds 100% of exact mentions across hours of video)
    if request.mode == "keyword":
        raw = collection.get(where=where_filter, include=["documents", "metadatas"])
        needle = request.query.strip().lower()
        matched_docs: list[str] = []
        matched_metas: list[dict] = []
        matched_scores: list[float] = []

        for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", [])):
            if needle in doc.lower():
                matched_docs.append(doc)
                matched_metas.append(meta)
                matched_scores.append(1.0)

        results = _build_search_results(
            matched_docs, matched_metas, matched_scores, speaker_filter=request.speaker
        )
        results.sort(key=lambda r: (r.upload_date, r.video_id, r.start_time))
        return SearchResponse(
            query=request.query,
            mode=request.mode,
            total_matches=len(results),
            results=results,
        )

    # Mode 2 & 3: Semantic vector search ('top_k' or 'semantic_all')
    with _gpu_lock:
        device = get_device()
        embedder = None
        try:
            embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            prefixed_query = f"{BGE_QUERY_INSTRUCTION}{request.query}"
            query_embedding = embedder.encode(
                prefixed_query,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        finally:
            purge_vram(embedder, label="SentenceTransformer (search)")
            embedder = None

    n_results = total_count if request.mode == "semantic_all" else min(request.top_k, total_count)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    scores = [max(0.0, 1.0 - float(d)) for d in distances]

    if request.mode == "semantic_all":
        filtered = [
            (doc, meta, sc)
            for doc, meta, sc in zip(documents, metadatas, scores)
            if sc >= request.min_score
        ]
        documents = [x[0] for x in filtered]
        metadatas = [x[1] for x in filtered]
        scores = [x[2] for x in filtered]

    results = _build_search_results(
        documents, metadatas, scores, speaker_filter=request.speaker
    )
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        total_matches=len(results),
        results=results,
    )
