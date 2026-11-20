"""
Project Cicero: Become Your Own Statesman
Streamlit frontend with a classical Italian & American civic aesthetic.
"""

from datetime import datetime
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

if "ui_logs" not in st.session_state:
    st.session_state["ui_logs"] = []


def log_ui_action(message: str) -> None:
    """Append a timestamped UI action entry to the session console log."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["ui_logs"].append(f"{stamp} [UI] {message}")


def fetch_backend_logs(api_url: str) -> list[str]:
    """Fetch recent backend pipeline logs from FastAPI /logs."""
    try:
        resp = requests.get(f"{api_url}/logs", timeout=5)
        resp.raise_for_status()
        return resp.json().get("logs", [])
    except requests.RequestException as exc:
        return [f"[UI] Could not fetch backend logs: {exc}"]


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total = max(0, int(seconds))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def fetch_indexed_videos(api_url: str) -> list[dict]:
    """Fetch the list of already-processed videos from the FastAPI backend."""
    try:
        resp = requests.get(f"{api_url}/videos", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return []


st.set_page_config(
    page_title="Project Cicero: Become Your Own Statesman",
    page_icon="🏛️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Classical Italian Marble & American Civic Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    /* Travertine marble & parchment canvas */
    .stApp {
        background: linear-gradient(180deg, #F8F4EC 0%, #EFE8D8 100%);
        color: #1B2432;
    }

    /* Sidebar in Federal Navy with Roman Gold trim */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16233B 0%, #1F3050 100%);
        border-right: 3px solid #C59B3B;
        color: #F8F4EC;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] a {
        color: #F5EFE2 !important;
    }
    [data-testid="stSidebar"] input {
        color: #1B2432 !important;
        background-color: #FAF6EE !important;
    }

    /* Classical Roman Pediment & American Civic Banner */
    .cicero-banner {
        background: linear-gradient(135deg, #16233B 0%, #24385E 55%, #7A1C24 100%);
        border: 2px solid #C59B3B;
        border-top: 6px solid #C59B3B;
        border-radius: 6px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 26px rgba(22, 35, 59, 0.18);
        color: #FAF6EE;
        text-align: center;
    }
    .cicero-crest {
        font-size: 1.05rem;
        letter-spacing: 0.28em;
        text-transform: uppercase;
        color: #E7C973;
        font-family: 'Cinzel', Georgia, serif;
        margin-bottom: 0.45rem;
    }
    .cicero-title {
        font-family: 'Cinzel', Georgia, serif;
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #FFFFFF;
        margin: 0.2rem 0;
        text-shadow: 0 2px 6px rgba(0, 0, 0, 0.35);
    }
    .cicero-subtitle {
        font-family: 'Lora', Georgia, serif;
        font-style: italic;
        font-size: 1.05rem;
        color: #EDE4D0;
        margin-top: 0.4rem;
    }
    .cicero-divider {
        height: 3px;
        width: 220px;
        margin: 0.9rem auto 0.6rem auto;
        background: linear-gradient(90deg, #B22234 0%, #FAF6EE 50%, #1B2A4A 100%);
        border-radius: 2px;
    }

    /* Section headings */
    h2, h3 {
        font-family: 'Cinzel', Georgia, serif !important;
        color: #16233B !important;
        letter-spacing: 0.03em;
    }

    /* Primary buttons in Roman Crimson */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"] {
        background: linear-gradient(180deg, #9B2226 0%, #7A181C 100%) !important;
        color: #FAF6EE !important;
        border: 1px solid #C59B3B !important;
        font-family: 'Cinzel', Georgia, serif !important;
        letter-spacing: 0.05em;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover {
        background: linear-gradient(180deg, #B22234 0%, #8B1E24 100%) !important;
    }

    /* Result cards styled like inscribed marble tablets */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFDF8 !important;
        border: 1px solid #D6C7A9 !important;
        border-left: 5px solid #16233B !important;
        border-radius: 6px !important;
        box-shadow: 0 3px 10px rgba(27, 36, 50, 0.06);
    }
    blockquote {
        font-family: 'Lora', Georgia, serif !important;
        border-left: 3px solid #9B2226 !important;
        background-color: #FAF5EA;
        padding: 0.65rem 1rem;
        border-radius: 0 4px 4px 0;
        color: #222936 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Classical Italian & American Civic Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cicero-banner">
        <div class="cicero-crest">★ &nbsp; E PLURIBUS UNUM &nbsp; • &nbsp; VERITAS IN FORO &nbsp; ★</div>
        <svg width="86" height="46" viewBox="0 0 120 64" xmlns="http://www.w3.org/2000/svg" aria-label="Classical American Civic Emblem">
            <!-- Classical Roman Pediment Roof -->
            <polygon points="60,4 114,22 6,22" fill="none" stroke="#E7C973" stroke-width="2.5"/>
            <circle cx="60" cy="14" r="3" fill="#E7C973"/>
            <!-- Stars & Stripes Frieze -->
            <rect x="14" y="24" width="92" height="5" fill="#B22234"/>
            <rect x="14" y="24" width="28" height="5" fill="#1B2A4A"/>
            <!-- Four Classical Columns -->
            <rect x="20" y="31" width="8" height="24" fill="#FAF6EE"/>
            <rect x="44" y="31" width="8" height="24" fill="#FAF6EE"/>
            <rect x="68" y="31" width="8" height="24" fill="#FAF6EE"/>
            <rect x="92" y="31" width="8" height="24" fill="#FAF6EE"/>
            <!-- Marble Plinth -->
            <rect x="10" y="56" width="100" height="4" fill="#E7C973"/>
        </svg>
        <div class="cicero-title">Project Cicero: Become Your Own Statesman</div>
        <div class="cicero-divider"></div>
        <div class="cicero-subtitle">
            Local Civic Archive &bull; Speaker-Aware Transcription, Semantic Inquiry &amp; Exhaustive Record Search
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: The Rostra (Ingestion Controls & Indexed Library)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🏛️ The Rostra: Ingest Record")
    api_url = st.text_input("FastAPI Backend URL", value=API_BASE_URL)

    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    diarization_choice = st.selectbox(
        "Speaker Identification",
        options=["Token-Free Local GPU (WavLM)", "Off (Faster)"],
        index=0,
        help="Uses microsoft/wavlm-base-plus-sv locally on your RTX 5050 GPU with no HuggingFace token required.",
    )
    diarization_mode = "wavlm_local" if diarization_choice.startswith("Token-Free") else "off"
    process_btn = st.button("Process Video", type="primary", use_container_width=True)

    if process_btn:
        if not youtube_url.strip():
            log_ui_action("Process Video clicked with empty URL.")
            st.warning("Please enter a valid YouTube URL.")
        else:
            log_ui_action(
                f"Started ingestion for '{youtube_url.strip()}' (diarization={diarization_mode})."
            )
            with st.spinner(
                "Running sequential VRAM pipeline (Download -> Whisper -> Diarize -> Chunk -> Embed)..."
            ):
                try:
                    response = requests.post(
                        f"{api_url.rstrip('/')}/ingest",
                        json={"url": youtube_url.strip(), "diarization_mode": diarization_mode},
                        timeout=900,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        speakers_str = ", ".join(data.get("speakers_detected", [])) or "N/A"
                        log_ui_action(
                            f"Ingestion succeeded: '{data['title']}' ({data['chunks_count']} chunks, "
                            f"{data['segments_count']} segments, speakers={speakers_str})."
                        )
                        st.success(
                            f"Indexed **{data['title']}** "
                            f"({data['chunks_count']} chunks, {data['segments_count']} segments, "
                            f"speakers: {speakers_str})."
                        )
                    else:
                        detail = response.json().get("detail", response.text)
                        log_ui_action(f"Ingestion failed ({response.status_code}): {detail}")
                        st.error(f"Ingestion failed: {detail}")
                except requests.RequestException as exc:
                    log_ui_action(f"Ingestion request error: {exc}")
                    st.error(f"Could not reach FastAPI backend: {exc}")

    st.divider()
    st.subheader("📜 Civic Archive")
    videos = fetch_indexed_videos(api_url.rstrip("/"))
    if not videos:
        st.info("No proceedings archived yet (or backend is offline).")
    else:
        for vid in videos:
            meta_bits = []
            if vid.get("channel"):
                meta_bits.append(vid["channel"])
            if vid.get("upload_date"):
                meta_bits.append(vid["upload_date"])
            if vid.get("duration"):
                meta_bits.append(format_timestamp(vid["duration"]))
            meta_suffix = f" ({' · '.join(meta_bits)})" if meta_bits else ""
            st.markdown(f"- [{vid['title']}]({vid['url']}){meta_suffix}")

# ---------------------------------------------------------------------------
# Main Area: The Forum (Natural Language & Exhaustive Record Search)
# ---------------------------------------------------------------------------
st.subheader("⚖️ The Forum: Search the Public Record")

with st.form("search_form", clear_on_submit=False):
    query = st.text_input(
        "Ask a question or enter an exact phrase",
        placeholder="e.g., school start times  OR  What did Speaker 1 say about the budget?",
    )

    col_mode, col_param, col_speaker, col_btn = st.columns([2.2, 1.3, 1.5, 1.0])
    with col_mode:
        search_mode_label = st.selectbox(
            "Search Mode",
            options=[
                "Top-K Semantic (Best N)",
                "All Semantic Matches (Threshold)",
                "Exhaustive Phrase Scan (Find Every Mention)",
            ],
            index=0,
        )
    with col_param:
        top_k = st.number_input("Top K (for Top-K)", min_value=1, max_value=200, value=10)
    with col_speaker:
        speaker_filter = st.text_input("Filter Speaker (optional)", placeholder="e.g. Speaker 1")
    with col_btn:
        st.write("")
        st.write("")
        search_submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    min_score = st.slider(
        "Minimum Similarity Threshold (for All Semantic Matches)",
        min_value=0.30,
        max_value=0.90,
        value=0.55,
        step=0.05,
    )

if search_submitted:
    if not query.strip():
        log_ui_action("Search submitted with empty query.")
        st.warning("Enter a search query first.")
    else:
        if search_mode_label.startswith("Top-K"):
            mode = "top_k"
        elif search_mode_label.startswith("All Semantic"):
            mode = "semantic_all"
        else:
            mode = "keyword"

        log_ui_action(
            f"Search submitted: query='{query.strip()}', mode={mode}, speaker='{speaker_filter.strip() or 'any'}'."
        )
        with st.spinner("Searching the local civic archive..."):
            try:
                resp = requests.post(
                    f"{api_url.rstrip('/')}/search",
                    json={
                        "query": query.strip(),
                        "mode": mode,
                        "top_k": int(top_k),
                        "min_score": float(min_score),
                        "speaker": speaker_filter.strip() or None,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                payload = resp.json()
                results = payload.get("results", [])
                log_ui_action(f"Search completed: {payload.get('total_matches', len(results))} match(es).")

                if not results:
                    st.info("No matching transcript segments found. Try lowering the threshold or ingesting a video first.")
                else:
                    st.markdown(f"Found **{payload.get('total_matches', len(results))}** matching segment(s):")
                    for idx, item in enumerate(results, start=1):
                        start_ts = format_timestamp(item["start_time"])
                        end_ts = format_timestamp(item.get("end_time", item["start_time"]))
                        meta_parts = [f"Time: `{start_ts}–{end_ts}`"]
                        if item.get("speakers"):
                            meta_parts.append(f"Speakers: `{item['speakers']}`")
                        if item.get("channel"):
                            meta_parts.append(f"Channel: `{item['channel']}`")
                        if item.get("upload_date"):
                            meta_parts.append(f"Uploaded: `{item['upload_date']}`")
                        meta_parts.append(f"Similarity: `{item['score']:.3f}`")

                        with st.container(border=True):
                            header_col, link_col = st.columns([4, 1])
                            with header_col:
                                st.markdown(f"**{idx}. {item['title']}**")
                                st.caption(" | ".join(meta_parts))
                            with link_col:
                                st.link_button(
                                    f"Watch @ {start_ts}",
                                    item["timestamped_url"],
                                    use_container_width=True,
                                )
                            st.markdown(f"> {item['text']}")
            except requests.RequestException as exc:
                log_ui_action(f"Search request error: {exc}")
                st.error(f"Search request failed: {exc}")

# ---------------------------------------------------------------------------
# Bottom Area: Civic Console Log (UI Actions + Backend Pipeline & VRAM Logs)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("📜 Civic Console Log")

log_col1, log_col2, _ = st.columns([1.2, 1.2, 5])
with log_col1:
    st.button("Refresh Logs", use_container_width=True)
with log_col2:
    if st.button("Clear UI Logs", use_container_width=True):
        st.session_state["ui_logs"] = []

backend_logs = fetch_backend_logs(api_url.rstrip("/"))
combined_logs = []
if st.session_state["ui_logs"]:
    combined_logs.append("=== FRONTEND ACTIONS ===")
    combined_logs.extend(st.session_state["ui_logs"])
if backend_logs:
    combined_logs.append("=== BACKEND PIPELINE & VRAM LOGS ===")
    combined_logs.extend(backend_logs)

if not combined_logs:
    combined_logs = ["No actions logged yet."]

st.code("\n".join(combined_logs[-120:]), language="log")
