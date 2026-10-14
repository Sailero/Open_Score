"""Minimal current-training stop and CUDA failure helpers."""
from pathlib import Path

def is_cuda_oom(error):
    message = str(error).lower()
    return (("out of memory" in message and ("cuda" in message or type(error).__name__ == "OutOfMemoryError"))
            or "cublas_status_alloc_failed" in message)

def write_stop(path, message):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(message) + "\n", encoding="utf-8")
