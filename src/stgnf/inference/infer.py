"""Standalone inference: score a single RetailS clip / pose JSON with a trained
STG-NF model and turn per-frame anomaly scores into shoplifting *events*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from stgnf.data.adapter import Clip, load_clip
from stgnf.data.preprocessing import build_windows, normalize_segments
from stgnf.models.build import build_model
from stgnf.utils.config import Config
from stgnf.utils.device import select_device


@dataclass
class Event:
    start_frame: int
    end_frame: int
    peak_score: float


class PoseAnomalyInference:
    """Loads a trained STG-NF checkpoint and scores clips."""

    def __init__(self, checkpoint: str | Path, device: Optional[str] = None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = Config(ckpt["config"])
        self.device = select_device(device or self.cfg.get_path("device", "auto"))
        pp, ds, m = self.cfg.preprocessing, self.cfg.dataset, self.cfg.model
        V = 18 if ds.kp18_format else 17
        V = V - 5 if ds.headless else V
        pose_shape = (m.in_channels, pp.seg_len, V)
        self.model = build_model(self.cfg, pose_shape, self.device)
        self.model.load_state_dict(ckpt["state_dict"], strict=False)
        self.model.set_actnorm_init()
        self.model.eval()

    @torch.no_grad()
    def score_clip(self, clip: Clip | str | Path) -> Tuple[np.ndarray, List[list]]:
        """Return ``(per_frame_anomaly, metadata)`` for a clip.

        ``per_frame_anomaly`` is a length-``num_frames`` array (NaN where no window
        covers the frame); ``metadata`` is the per-window ``[stem, pid, start]``.
        """
        if not isinstance(clip, Clip):
            clip = load_clip(clip)
        pp, ds, m = self.cfg.preprocessing, self.cfg.dataset, self.cfg.model
        segs, meta, scores = build_windows(
            clip, seg_len=pp.seg_len, seg_stride=1, max_missing=pp.max_missing,
            interpolate_gaps=pp.interpolate_gaps, kp18_format=ds.kp18_format,
            headless=ds.headless)
        num_frames = clip.num_frames
        anomaly = np.full(num_frames, np.nan, dtype=np.float64)
        if segs.shape[0] == 0:
            return anomaly, meta
        norm = normalize_segments(
            segs, vid_res=ds.vid_res, symm_range=pp.normalization.symm_range,
            seg_norm=pp.normalization.seg_norm)
        x = torch.from_numpy(norm).to(self.device).float()
        if not m.model_confidence:
            x = x[:, :m.in_channels]
        label = torch.ones(x.shape[0], device=self.device)
        score_t = torch.from_numpy(scores).to(self.device).float().amin(dim=-1)
        _z, nll = self.model(x, label=label, score=score_t)
        if m.model_confidence:
            nll = nll * score_t
        window_anomaly = np.atleast_1d(nll.float().cpu().numpy().reshape(-1))

        offset = pp.seg_len // 2
        # per-person max anomaly across windows at each frame
        per_frame = np.full(num_frames, -np.inf, dtype=np.float64)
        for i, (_stem, _pid, start) in enumerate(meta):
            f = int(start) + offset
            if 0 <= f < num_frames:
                per_frame[f] = max(per_frame[f], float(window_anomaly[i]))
        covered = per_frame > -np.inf
        anomaly[covered] = per_frame[covered]
        return anomaly, meta

    @staticmethod
    def detect_events(anomaly: np.ndarray, threshold: float, min_len: int = 1) -> List[Event]:
        """Threshold a per-frame anomaly track into contiguous events."""
        mask = np.nan_to_num(anomaly, nan=-np.inf) >= threshold
        events: List[Event] = []
        i, n = 0, len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                seg = anomaly[i:j]
                peak = float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")
                events.append(Event(start_frame=i, end_frame=j - 1, peak_score=peak))
            i = j
        return events
