"""GPU / device selection utilities.

The workstation has two GPUs of different sizes and their CUDA indices are *not*
guaranteed (the larger card is not always ``cuda:0``). Selection is therefore
based on **free memory**, with an ``nvidia-smi`` cross-check when available and a
clean CPU fallback.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: float
    free_mb: float

    def __str__(self) -> str:
        return (f"cuda:{self.index} {self.name} "
                f"(free {self.free_mb:.0f} MB / total {self.total_mb:.0f} MB)")


def _nvidia_smi_free() -> dict[int, float]:
    """Return {index: free_MB} via nvidia-smi, or {} if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return {}
    free: dict[int, float] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                free[int(parts[0])] = float(parts[1])
            except ValueError:
                continue
    return free


def list_gpus() -> List[GpuInfo]:
    """Enumerate visible CUDA devices with free/total memory (MB)."""
    if not torch.cuda.is_available():
        return []
    smi_free = _nvidia_smi_free()
    gpus: List[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mb = props.total_memory / (1024 ** 2)
        if i in smi_free:
            free_mb = smi_free[i]
        else:
            # torch.cuda.mem_get_info reflects real free memory on the device.
            try:
                free_bytes, _ = torch.cuda.mem_get_info(i)
                free_mb = free_bytes / (1024 ** 2)
            except (RuntimeError, AssertionError):
                free_mb = total_mb - torch.cuda.memory_reserved(i) / (1024 ** 2)
        gpus.append(GpuInfo(index=i, name=props.name, total_mb=total_mb, free_mb=free_mb))
    return gpus


def select_device(preference: str = "auto") -> torch.device:
    """Resolve a device string.

    Args:
        preference: ``auto`` picks the CUDA device with the most free memory;
            ``cpu`` forces CPU; ``cuda`` / ``cuda:N`` request CUDA (falls back to
            CPU with a warning if CUDA is unavailable).
    """
    pref = (preference or "auto").lower()

    if pref == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        if pref not in ("auto", "cpu"):
            print(f"[device] CUDA unavailable; requested '{preference}'. Falling back to CPU.")
        return torch.device("cpu")

    if pref.startswith("cuda:"):
        idx = int(pref.split(":", 1)[1])
        if idx >= torch.cuda.device_count():
            raise ValueError(f"Requested {preference} but only "
                             f"{torch.cuda.device_count()} CUDA device(s) present.")
        return torch.device(pref)

    if pref == "cuda":
        pref = "auto"

    if pref == "auto":
        gpus = list_gpus()
        best = max(gpus, key=lambda g: g.free_mb)
        print("[device] Available GPUs:")
        for g in gpus:
            marker = "  <-- selected" if g.index == best.index else ""
            print(f"[device]   {g}{marker}")
        return torch.device(f"cuda:{best.index}")

    raise ValueError(f"Unrecognized device preference: {preference!r}")
