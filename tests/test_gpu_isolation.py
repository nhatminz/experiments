from __future__ import annotations

import pytest

from gpu_isolation import exclusive_gpu_lease


def test_duplicate_physical_gpu_lease_fails_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU_LOCK_ROOT", str(tmp_path))
    with exclusive_gpu_lease("GPU-test-uuid"):
        with pytest.raises(RuntimeError, match="already owned"):
            with exclusive_gpu_lease("GPU-test-uuid"):
                pass


def test_different_physical_gpu_leases_can_coexist(tmp_path, monkeypatch):
    monkeypatch.setenv("EXPERIMENT_GPU_LOCK_ROOT", str(tmp_path))
    with exclusive_gpu_lease("GPU-a"):
        with exclusive_gpu_lease("GPU-b"):
            pass
