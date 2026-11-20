"""
VRAM management utilities for sequential GPU model execution on 6GB/8GB GPUs.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def configure_cuda_dll_paths() -> None:
    """
    Register PyTorch's bundled CUDA/cuBLAS/cuDNN DLL directory on Windows.

    ctranslate2 (used by faster-whisper) needs cublas64_12.dll and cudnn64_9.dll
    on the DLL search path before it loads; otherwise it can fail with
    CUBLAS_STATUS_NOT_SUPPORTED or missing-library errors.
    """
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(torch_lib))
        current_path = os.environ.get("PATH", "")
        if str(torch_lib) not in current_path:
            os.environ["PATH"] = f"{torch_lib}{os.pathsep}{current_path}"


configure_cuda_dll_paths()


def get_device() -> str:
    """Return 'cuda' when an NVIDIA GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def purge_vram(model: Any = None, label: str = "model") -> None:
    """
    Explicitly release a model from GPU memory and flush CUDA caches.

    On consumer GPUs with 6GB-8GB VRAM (such as an RTX 5050), keeping both
    Whisper and a SentenceTransformer loaded simultaneously can trigger CUDA
    Out-Of-Memory (OOM) errors. Calling this helper between pipeline stages
    ensures the previous model's tensors and allocator blocks are freed before
    loading the next model.
    """
    # Step 1: Drop the caller's reference so Python's refcount can reach zero.
    if model is not None:
        del model

    # Step 2: Run cyclic garbage collection to destroy any lingering reference
    # cycles inside CTranslate2 / PyTorch / HuggingFace wrappers.
    gc.collect()

    # Step 3: Flush PyTorch's CUDA caching allocator so freed VRAM is returned
    # to the GPU driver for the next model (e.g., CTranslate2 <-> PyTorch).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        logger.info(
            "Purged %s from VRAM (PyTorch allocated: %.1f MB, reserved: %.1f MB)",
            label,
            allocated_mb,
            reserved_mb,
        )
    else:
        logger.info("Purged %s from CPU memory.", label)
