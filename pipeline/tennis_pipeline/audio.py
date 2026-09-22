"""Racket-impact onsets from broadcast audio (crowds are silent during rallies).

Used to snap visually detected hits to precise contact times and to confirm them.
"""
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt

from .paths import match_dir

SR = 16000
HOP_S = 0.005


def onsets(video_id: str, video_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cache = match_dir(video_id) / "audio_onsets.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["strength"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    y = sosfilt(butter(4, 2000, "hp", fs=SR, output="sos"), x)
    hop, win = int(HOP_S * SR), int(0.01 * SR)
    n = (len(y) - win) // hop
    frames = np.lib.stride_tricks.as_strided(y, shape=(n, win), strides=(y.strides[0] * hop, y.strides[0]))
    loud = np.log(np.sqrt((frames ** 2).mean(1)) + 1e-6)
    flux = np.maximum(np.diff(loud, prepend=loud[0]), 0)
    env = np.convolve(flux, np.ones(3) / 3, "same")
    # Local adaptive threshold over ~4 s windows.
    k = int(4 / HOP_S)
    med = np.array([np.median(env[max(0, i - k):i + k]) for i in range(0, n, 200)])
    med = np.repeat(med, 200)[:n]
    pk, props = find_peaks(env, height=med + 0.35, distance=int(0.25 / HOP_S))
    t, strength = pk * HOP_S, props["peak_heights"]
    np.savez_compressed(cache, t=t, strength=strength)
    return t, strength


def snap(hit_t: np.ndarray, on_t: np.ndarray, on_s: np.ndarray, before: float = 0.25, after: float = 0.15):
    """Nearest strong onset in [t-before, t+after] for each hit; NaN if none."""
    out_t = np.full(len(hit_t), np.nan)
    out_s = np.full(len(hit_t), np.nan)
    for i, t in enumerate(hit_t):
        lo, hi = np.searchsorted(on_t, [t - before, t + after])
        if hi > lo:
            j = lo + int(np.argmax(on_s[lo:hi]))
            out_t[i], out_s[i] = on_t[j], on_s[j]
    return out_t, out_s
