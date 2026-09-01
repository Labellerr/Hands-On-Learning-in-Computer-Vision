"""Offline temporal smoothing for YOLO/COCO-17 pose tracks.

Per-frame pose models (including YOLO11-pose) estimate each frame independently.
Drawing those keypoints raw makes skeleton overlays jitter. For recorded video,
interpolate short gaps then apply a Savitzky–Golay filter on each coordinate.

This does not change the detector. It is a post-process for overlay/analysis.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

N_KPT = 17


def _interp_track(vals: np.ndarray, mask: np.ndarray, max_gap: int = 8) -> tuple[np.ndarray, np.ndarray]:
    n = len(vals)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return vals.copy(), mask.copy()
    filled = np.interp(np.arange(n), idx, vals[idx])
    ok = mask.copy()
    for i in range(n):
        if mask[i]:
            continue
        left = idx[idx < i]
        right = idx[idx > i]
        if left.size and right.size and (right[0] - left[-1]) <= max_gap:
            ok[i] = True
        elif left.size and i - left[-1] <= max_gap // 2:
            ok[i] = True
        elif right.size and right[0] - i <= max_gap // 2:
            ok[i] = True
    return filled, ok


def smooth_pose_tracks(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float = 0.28,
    window: int = 15,
    poly: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """xy: (T, K, 2), conf: (T, K). Returns smoothed xy and visibility."""
    t, k, _ = xy.shape
    sm = np.zeros_like(xy, dtype=np.float32)
    vis = np.zeros((t, k), dtype=bool)
    w = window if window % 2 else window - 1
    for j in range(k):
        mask = conf[:, j] >= conf_thr
        sx, okx = _interp_track(xy[:, j, 0], mask)
        sy, oky = _interp_track(xy[:, j, 1], mask)
        ok = okx & oky
        ww = min(w, int(ok.sum()) | 1)
        if ww >= poly + 3:
            if ww % 2 == 0:
                ww -= 1
            sx = savgol_filter(sx, ww, poly, mode="interp")
            sy = savgol_filter(sy, ww, poly, mode="interp")
        sm[:, j, 0] = sx
        sm[:, j, 1] = sy
        vis[:, j] = ok
    return sm, vis
