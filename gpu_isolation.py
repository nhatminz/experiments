from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
from pathlib import Path


@contextlib.contextmanager
def exclusive_gpu_lease(identity: str):
    """Prevent two Experiment runners from starting vLLM on one physical GPU."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    lock_root = Path(os.environ.get("EXPERIMENT_GPU_LOCK_ROOT", "/tmp"))
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"cmt-state-intervention-gpu-{digest}.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                "This physical GPU is already owned by another Experiment run "
                f"({owner}). Choose a different CUDA_VISIBLE_DEVICES value or "
                "wait for that run to finish."
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid={os.getpid()} identity={identity} "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
        )
        handle.flush()
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
