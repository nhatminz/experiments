"""Filesystem, provenance, and exact model-restoration helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .protocol import assert_finite_values


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    assert_finite_values(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for row in rows:
            assert_finite_values(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def sanitize_continuation_audit(path: Path, completed_ids: set[int]) -> None:
    """Drop records from a crash before its intervention commit."""
    if not path.is_file():
        return
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    kept = [row for row in rows if int(row["intervention_id"]) in completed_ids]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_parameters(model) -> dict[str, torch.Tensor]:
    snapshot = {
        name: parameter.detach().clone(memory_format=torch.preserve_format)
        for name, parameter in model.named_parameters()
    }
    if not snapshot:
        raise ValueError("Student model has no parameters")
    return snapshot


@torch.no_grad()
def restore_parameters_exact(model, snapshot: dict[str, torch.Tensor]) -> None:
    current = dict(model.named_parameters())
    if current.keys() != snapshot.keys():
        raise AssertionError("Model parameter names changed after the local update")
    for name, parameter in current.items():
        parameter.copy_(snapshot[name])
    for name, parameter in current.items():
        if not torch.equal(parameter, snapshot[name]):
            raise AssertionError(f"Exact theta_base restoration failed for {name}")


def snapshot_sha256(snapshot: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in snapshot.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def environment_payload(main_repo: Path) -> dict[str, Any]:
    versions: dict[str, str] = {"torch": torch.__version__}
    for package in ("transformers", "vllm", "numpy", "matplotlib"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as error:
            versions[package] = f"unavailable: {error}"
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        }
    return {
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
        "gpu": gpu,
        "main_repo": str(main_repo),
        "main_repo_commit": git_commit(main_repo),
    }


class ProgressLogger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        print(message, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")


REQUIRED_INTERVENTION_FIELDS = {
    "intervention_id",
    "local_reverse_kl_before",
    "local_reverse_kl_after",
    "delta_immediate",
    "future_quality_before_mean",
    "future_quality_after_mean",
    "delta_future",
    "base_restoration_verified_exact",
}


def validate_intervention_schema(payload: dict[str, Any]) -> None:
    missing = REQUIRED_INTERVENTION_FIELDS - payload.keys()
    if missing:
        raise ValueError(f"Intervention record is missing fields: {sorted(missing)}")
    assert_finite_values(payload)


def pending_candidates(
    candidates: list[dict[str, Any]], completed_ids: set[int]
) -> list[dict[str, Any]]:
    return [
        row for row in candidates if int(row["candidate_id"]) not in completed_ids
    ]
