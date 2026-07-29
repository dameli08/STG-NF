import numpy as np
import torch

from stgnf.utils.device import select_device, list_gpus, GpuInfo


def test_cpu_selection_forced():
    dev = select_device("cpu")
    assert dev.type == "cpu"


def test_auto_selection_returns_device():
    dev = select_device("auto")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")


def test_auto_picks_largest_free_memory(monkeypatch):
    fake = [
        GpuInfo(index=0, name="small", total_mb=6000, free_mb=1000),
        GpuInfo(index=1, name="big", total_mb=11000, free_mb=9000),
    ]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr("stgnf.utils.device.list_gpus", lambda: fake)
    dev = select_device("auto")
    assert dev == torch.device("cuda:1")


def test_manual_override_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    import pytest
    with pytest.raises(ValueError):
        select_device("cuda:5")


def test_list_gpus_type():
    gpus = list_gpus()
    assert isinstance(gpus, list)
    for g in gpus:
        assert g.free_mb >= 0
        assert g.total_mb > 0
