"""Lightweight stdout logging with a consistent format."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "stgnf", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        root = logging.getLogger("stgnf")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("stgnf") else f"stgnf.{name}")
