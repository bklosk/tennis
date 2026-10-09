"""Racket-impact candidates from the broadcast audio.

Onset strength is the half-wave-rectified spectral flux of a log spectrogram restricted to a band
where racket impacts dominate speech and crowd noise (default 1.5-10 kHz). It is normalised to a
robust z-score against a sliding median/MAD so the threshold adapts to each broadcast, then
peak-picked with a minimum gap. Processing is streamed, so a three-hour match never sits in RAM
as raw samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class OnsetConfig:
    sr: int = 32000
    n_fft: int = 512
    hop: int = 128
    fmin: float = 1500.0
    fmax: float = 10000.0
    hf_split: float = 4000.0
    lf_max: float = 1000.0
    norm_window_s: float = 1.5
    threshold_z: float = 3.0
    min_gap_s: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)


def envelope(chunks: Iterable[np.ndarray], cfg: OnsetConfig) -> pd.DataFrame:
    """Per-hop onset features: t, flux, e_hf, e_lf, e_band."""
    n_fft, hop, sr = cfg.n_fft, cfg.hop, cfg.sr
    win = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= cfg.fmin) & (freqs <= min(cfg.fmax, sr / 2))
    hf = (freqs >= cfg.hf_split) & (freqs <= min(cfg.fmax, sr / 2))
    lf = (freqs >= 100) & (freqs < cfg.lf_max)
    buf = np.zeros(0, np.float32)
    offset = 0
    next_frame = 0
    prev = None
    out_flux, out_hf, out_lf, out_band = [], [], [], []
    for chunk in chunks:
        buf = np.concatenate([buf, chunk.astype(np.float32)])
        start = next_frame * hop - offset
        n_avail = (len(buf) - start - n_fft) // hop + 1
        if n_avail <= 0:
            continue
        idx = start + np.arange(n_avail)[:, None] * hop + np.arange(n_fft)[None, :]
        spec = np.abs(np.fft.rfft(buf[idx] * win, axis=1)).astype(np.float32)
        logb = np.log1p(100.0 * spec[:, band])
        if prev is None:
            prev = logb[:1]
        diff = np.diff(np.vstack([prev, logb]), axis=0)
        out_flux.append(np.maximum(diff, 0).sum(axis=1))
        p2 = spec ** 2
        out_hf.append(p2[:, hf].sum(axis=1))
        out_lf.append(p2[:, lf].sum(axis=1))
        out_band.append(p2[:, band].sum(axis=1))
        prev = logb[-1:]
        next_frame += n_avail
        keep = next_frame * hop - offset
        buf = buf[keep:]
        offset += keep
    if not out_flux:
        return pd.DataFrame(columns=["t", "flux", "e_hf", "e_lf", "e_band"])
    flux = np.concatenate(out_flux)
    t = (np.arange(len(flux)) * hop + n_fft / 2) / sr
    return pd.DataFrame({"t": t, "flux": flux, "e_hf": np.concatenate(out_hf),
                         "e_lf": np.concatenate(out_lf), "e_band": np.concatenate(out_band)})


def robust_z(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    scale = 1.4826 * mad
    floor = max(float(np.median(scale)) * 0.25, 1e-6)
    return ((s - med) / np.maximum(scale, floor)).to_numpy()


def pick_onsets(env: pd.DataFrame, cfg: OnsetConfig) -> pd.DataFrame:
    if env.empty:
        return pd.DataFrame(columns=["t", "z", "flux", "hf_ratio", "level_db"])
    frame_rate = cfg.sr / cfg.hop
    z = robust_z(env["flux"].to_numpy(), max(3, int(cfg.norm_window_s * frame_rate)))
    half = max(1, int(cfg.min_gap_s * frame_rate / 2))
    zmax = pd.Series(z).rolling(2 * half + 1, center=True, min_periods=1).max().to_numpy()
    cand = np.flatnonzero((z >= cfg.threshold_z) & (z >= zmax))
    # enforce the minimum gap, strongest first
    order = cand[np.argsort(-z[cand])]
    taken = np.zeros(len(z), bool)
    keep = []
    gap = int(cfg.min_gap_s * frame_rate)
    for i in order:
        lo, hi = max(0, i - gap), min(len(z), i + gap + 1)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        keep.append(i)
    keep = np.sort(np.array(keep, int))
    e_hf = env["e_hf"].to_numpy()
    e_lf = env["e_lf"].to_numpy()
    e_band = env["e_band"].to_numpy()
    # energies over the first ~20 ms after the onset
    w = max(1, int(0.02 * frame_rate))
    hf_sum = np.array([e_hf[i:i + w].sum() for i in keep])
    lf_sum = np.array([e_lf[i:i + w].sum() for i in keep])
    band_sum = np.array([e_band[i:i + w].sum() for i in keep])
    return pd.DataFrame({
        "t": env["t"].to_numpy()[keep],
        "z": z[keep],
        "flux": env["flux"].to_numpy()[keep],
        "hf_ratio": np.log10((hf_sum + 1e-9) / (lf_sum + 1e-9)) if len(keep) else [],
        "level_db": 10 * np.log10(band_sum + 1e-12) if len(keep) else [],
    })


def detect_onsets(path_or_chunks, cfg: OnsetConfig | None = None) -> pd.DataFrame:
    """Onsets from a media file path or an iterable of mono float32 chunks at cfg.sr."""
    cfg = cfg or OnsetConfig()
    if isinstance(path_or_chunks, str):
        from .video import audio_chunks

        chunks = audio_chunks(path_or_chunks, sr=cfg.sr)
    else:
        chunks = path_or_chunks
    return pick_onsets(envelope(chunks, cfg), cfg)
