"""
Local, token-free speaker diarization using WavLM X-Vectors and Agglomerative Clustering.

Runs sequentially in its own GPU stage and purges VRAM before the embedding stage.
"""

import logging
import os
from pathlib import Path

import av
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

from vram_utils import purge_vram

logger = logging.getLogger(__name__)

DIARIZATION_MODEL_NAME = os.getenv("DIARIZATION_MODEL", "microsoft/wavlm-base-plus-sv")
DIARIZATION_THRESHOLD = float(os.getenv("DIARIZATION_THRESHOLD", "0.14"))
SAMPLE_RATE = 16000
MIN_SEGMENT_SEC = 0.5
MAX_SEGMENT_SEC = 15.0


def load_audio_16k_mono(audio_path: Path) -> np.ndarray:
    """Decode an audio file into a 16 kHz mono float32 numpy array using PyAV."""
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
    chunks: list[np.ndarray] = []

    with av.open(str(audio_path)) as container:
        audio_stream = next(
            (s for s in container.streams if s.type == "audio"),
            None,
        )
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {audio_path}")

        for frame in container.decode(audio_stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()
                chunks.append(arr.reshape(-1))

        # Flush resampler buffer
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            chunks.append(arr.reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _assign_sequential_speaker_names(raw_labels: list[int]) -> list[str]:
    """Map arbitrary cluster IDs to 'Speaker 1', 'Speaker 2', ... in order of appearance."""
    mapping: dict[int, str] = {}
    named: list[str] = []
    for label in raw_labels:
        if label not in mapping:
            mapping[label] = f"Speaker {len(mapping) + 1}"
        named.append(mapping[label])
    return named


def assign_speakers(
    audio_path: Path,
    segments: list[dict],
    device: str,
    waveform: np.ndarray | None = None,
) -> list[dict]:
    """
    Annotate each Whisper segment dict with a `speaker` label (e.g., 'Speaker 1').

    Falls back gracefully if audio decoding or model loading fails.
    """
    if not segments:
        return segments

    if waveform is None:
        try:
            waveform = load_audio_16k_mono(audio_path)
        except Exception as exc:
            logger.warning("Audio decode for diarization failed (%s); skipping speaker labels.", exc)
            for seg in segments:
                seg["speaker"] = "Speaker 1"
            return segments

    total_samples = len(waveform)
    valid_indices: list[int] = []
    valid_slices: list[np.ndarray] = []

    for idx, seg in enumerate(segments):
        start_sample = max(0, int(seg["start"] * SAMPLE_RATE))
        end_sample = min(total_samples, int(seg["end"] * SAMPLE_RATE))
        if (end_sample - start_sample) >= int(MIN_SEGMENT_SEC * SAMPLE_RATE):
            # Cap slice length to MAX_SEGMENT_SEC to keep VRAM usage minimal.
            max_samples = int(MAX_SEGMENT_SEC * SAMPLE_RATE)
            if (end_sample - start_sample) > max_samples:
                end_sample = start_sample + max_samples
            valid_indices.append(idx)
            valid_slices.append(waveform[start_sample:end_sample])

    if len(valid_indices) < 2:
        for seg in segments:
            seg["speaker"] = "Speaker 1"
        return segments

    model = None
    feature_extractor = None
    embeddings: list[np.ndarray] = []

    try:
        logger.info("Loading speaker diarization model (%s) on %s", DIARIZATION_MODEL_NAME, device)
        feature_extractor = AutoFeatureExtractor.from_pretrained(DIARIZATION_MODEL_NAME)
        model = AutoModelForAudioXVector.from_pretrained(DIARIZATION_MODEL_NAME).to(device)
        model.eval()

        with torch.no_grad():
            for audio_slice in valid_slices:
                inputs = feature_extractor(
                    audio_slice,
                    sampling_rate=SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                emb = model(**inputs).embeddings
                emb = torch.nn.functional.normalize(emb, dim=-1)
                embeddings.append(emb.squeeze(0).cpu().numpy())
    except Exception as exc:
        logger.warning("Speaker diarization failed (%s); defaulting to Speaker 1.", exc)
        for seg in segments:
            seg["speaker"] = "Speaker 1"
        return segments
    finally:
        # Strictly purge diarization model from VRAM before SentenceTransformer loads.
        purge_vram(model, label="WavLM-XVector diarizer")
        model = None
        feature_extractor = None

    emb_matrix = np.vstack(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DIARIZATION_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    cluster_ids = clustering.fit_predict(emb_matrix).tolist()
    named_valid = _assign_sequential_speaker_names(cluster_ids)

    for idx, speaker_name in zip(valid_indices, named_valid):
        segments[idx]["speaker"] = speaker_name

    # Propagate speaker labels to ultra-short (<0.5s) segments from nearest labeled neighbor.
    for idx, seg in enumerate(segments):
        if "speaker" not in seg:
            nearest_pos = min(valid_indices, key=lambda v: abs(v - idx))
            seg["speaker"] = segments[nearest_pos]["speaker"]

    unique_count = len({seg["speaker"] for seg in segments})
    logger.info("Diarization complete: detected %d speaker(s) across %d segments.", unique_count, len(segments))
    return segments
