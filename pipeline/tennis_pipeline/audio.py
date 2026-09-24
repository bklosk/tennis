"""Racket-impact onsets from broadcast audio, plus a rally-length feature.

Crowds are quiet during rallies, but the mix still contains commentary. Commentary and
most crowd wash sit in the center of the stereo image; racket impacts and bounces come
from the court mics and show up in the left-right difference. That side/mid ratio is the
provenance of the sound. Sharp, strong court onsets are the impacts used as a rally-length
feature: the ball tracker drops shots in long rallies, and those impacts continue after
the last tracked hit until the court goes quiet.
"""
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfilt

from . import video
from .paths import match_dir

SR = 16000
HOP_S = 0.005
# Side/mid RMS on a US Open broadcast: court impacts sit near 0.31, centered booth
# transients (commentary, crowd) near 0.17.
PROV_DEAD = 0.17
PROV_COURT = 0.31
# A hit-like impact: court provenance, a sharp high-band attack, and a strong onset.
# Bounces are duller and weaker, so they are not counted as shots.
HIT_PROV = 0.5
HIT_SHARP = 4.0
HIT_STRENGTH = 0.7
GAP_STOP = 2.5  # silence longer than this ends the rally
MIN_SEP = 0.28


class Onsets:
    """Times and strengths, plus provenance and sharpness aligned to those times.

    Iterates as ``(t, strength)`` so existing snap and serve callers keep working.
    """

    def __init__(self, t, strength, provenance, sharp):
        self.t = np.asarray(t, float)
        self.strength = np.asarray(strength, float)
        self.provenance = np.asarray(provenance, float)
        self.sharp = np.asarray(sharp, float)

    def __iter__(self):
        yield self.t
        yield self.strength


def court_provenance(ratio: np.ndarray) -> np.ndarray:
    """Map a side/mid RMS ratio onto 0 (booth) .. 1 (court)."""
    return np.clip((np.asarray(ratio, float) - PROV_DEAD) / (PROV_COURT - PROV_DEAD), 0, 1)


def _cached(z: np.lib.npyio.NpzFile) -> Onsets:
    n = len(z["t"])
    provenance = z["provenance"] if "provenance" in z.files else np.ones(n)
    sharp = z["sharp"] if "sharp" in z.files else np.full(n, np.nan)
    return Onsets(z["t"], z["strength"], provenance, sharp)


def onsets(video_id: str, video_path: Path | None) -> Onsets:
    cache = match_dir(video_id) / "audio_onsets.npz"
    cached = np.load(cache) if cache.exists() else None
    current = (cached is not None and "sharp" in cached.files and "audio_version" in cached.files
               and int(cached["audio_version"]) >= 2)
    if current:
        return _cached(cached)
    # An older cache has times and strengths only. Recompute when the full video is
    # still here; otherwise keep it so events can run after the broadcast is deleted.
    reusable = cached is not None and (video_path is None or not Path(video_path).exists()
                                        or video.is_pack(video_path))
    if reusable:
        return _cached(cached)
    if video_path is None or not Path(video_path).exists():
        raise FileNotFoundError(cache)
    if video.is_pack(video_path):
        raise FileNotFoundError(f"{video_id}: a main-camera pack has no audio; compute onsets from the full "
                                "video first (batch prep does this)")
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video_path), "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, 2)
    mid, side = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    y = sosfilt(butter(4, 2000, "hp", fs=SR, output="sos"), mid)
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
    provenance = court_provenance(np.array([_side_mid_ratio(side, mid, ti) for ti in t]))
    sharp = np.array([_sharpness(side, ti) for ti in t])
    np.savez_compressed(cache, t=t, strength=strength, provenance=provenance, sharp=sharp, audio_version=np.int32(2))
    return Onsets(t, strength, provenance, sharp)


def _side_mid_ratio(side: np.ndarray, mid: np.ndarray, t: float) -> float:
    i = int(t * SR)
    a, b = max(0, i - 160), min(len(side), i + 480)
    if b <= a:
        return 0.0
    rs = float(np.sqrt((side[a:b] ** 2).mean())) + 1e-8
    rm = float(np.sqrt((mid[a:b] ** 2).mean())) + 1e-8
    return rs / rm


def _sharpness(side: np.ndarray, t: float) -> float:
    """High-band attack: energy in the first 10 ms after the peak over the following tail."""
    i0 = int(np.clip(t * SR, 400, len(side) - 2000))
    w = side[i0 - 240:i0 + 240]
    c = i0 - 240 + int(np.argmax(np.abs(w)))
    seg = side[c - 80:c + 1280]
    y = sosfilt(butter(4, [2000, 7500], btype="band", fs=SR, output="sos"), seg)
    e0 = float(np.sqrt((y[80:80 + 160] ** 2).mean())) + 1e-9
    e1 = float(np.sqrt((y[80 + 320:80 + 960] ** 2).mean())) + 1e-9
    return e0 / e1


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


def shot_count(impacts: Onsets | None, t0: float, t_limit: float) -> int | None:
    """Hit-like court impacts from ``t0`` until a silence, not past ``t_limit``.

    This is a rally-length feature, not a replacement for tracked shots. Each kept
    onset is one impact. A gap longer than ``GAP_STOP`` ends the count.
    """
    if impacts is None or not len(impacts.t) or np.isnan(impacts.sharp).all():
        return None
    m = ((impacts.t >= t0 - 0.1) & (impacts.t <= t_limit)
         & (impacts.provenance >= HIT_PROV) & (impacts.sharp >= HIT_SHARP)
         & (impacts.strength >= HIT_STRENGTH))
    ts = impacts.t[m]
    if not len(ts):
        return 0
    n, last = 1, ts[0]
    for ti in ts[1:]:
        if ti - last > GAP_STOP:
            break
        if ti - last >= MIN_SEP:
            n += 1
            last = ti
    return n
