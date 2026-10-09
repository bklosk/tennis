"""Audio impact candidates.

Racket impacts are short broadband transients that stand out above speech and crowd noise once
the low frequencies are removed. Candidates are peaks of a high-band spectral flux (SuperFlux
style: compare each frame with the max of the two frames before it, after a small max-filter
across frequency), normalised by a running median/MAD so the threshold adapts to each video.
Each candidate carries features for a learned hit classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from uso.paths import match_dir
from uso.video import extract_audio, load_audio, video_path

N_FFT = 512
HOP = 128
BANDS = [(300, 1500), (1500, 3000), (3000, 6000), (6000, 12000)]


def _stft_logmag(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(N_FFT).astype(np.float32)
    n = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    return np.log1p(1000.0 * mag), freqs


def onset_envelope(x: np.ndarray, sr: int, chunk_s: float = 60.0):
    """High-band flux envelope at HOP resolution plus per-frame band log-energies."""
    sos = signal.butter(4, 1000, btype="highpass", fs=sr, output="sos")
    chunk = int(chunk_s * sr)
    env_parts, band_parts = [], []
    pad = N_FFT + 2 * HOP
    for s0 in range(0, len(x), chunk):
        a = max(0, s0 - pad)
        seg = x[a: s0 + chunk]
        if len(seg) < N_FFT + 3 * HOP:
            break
        L, freqs = _stft_logmag(signal.sosfilt(sos, seg).astype(np.float32), sr)
        Lraw, _ = _stft_logmag(seg.astype(np.float32), sr)
        sel = (freqs >= 1500) & (freqs <= 10000)
        Lm = ndimage.maximum_filter1d(L[:, sel], size=3, axis=1)
        prev = np.maximum(np.roll(Lm, 1, axis=0), np.roll(Lm, 2, axis=0))
        flux = np.maximum(0.0, L[:, sel] - prev).sum(1)
        flux[:2] = 0
        bands = np.stack([Lraw[:, (freqs >= lo) & (freqs < hi)].mean(1) for lo, hi in BANDS], 1)
        skip = (s0 - a) // HOP
        keep = min(chunk // HOP, len(flux) - skip)
        env_parts.append(flux[skip: skip + keep])
        band_parts.append(bands[skip: skip + keep])
    env = np.concatenate(env_parts).astype(np.float32)
    bands = np.concatenate(band_parts).astype(np.float32)
    t = (np.arange(len(env)) * HOP + N_FFT / 2) / sr
    return t, env, bands


def _running_stats(env: np.ndarray, sr: int, win_s: float = 2.0):
    """Running median and MAD on a decimated grid (fast), interpolated back."""
    step = 8
    dec = env[::step]
    w = max(3, int(win_s * sr / HOP / step) | 1)
    med = ndimage.median_filter(dec, size=w, mode="nearest")
    mad = ndimage.median_filter(np.abs(dec - med), size=w, mode="nearest")
    xi = np.arange(len(env)) / step
    return np.interp(xi, np.arange(len(dec)), med), np.interp(xi, np.arange(len(dec)), mad)


def candidates(x: np.ndarray, sr: int, z_min: float = 3.0, min_sep: float = 0.06) -> pd.DataFrame:
    t, env, bands = onset_envelope(x, sr)
    med, mad = _running_stats(env, sr)
    z = (env - med) / (1.4826 * mad + 1e-3 + 0.05 * np.median(env))
    dist = max(1, int(min_sep * sr / HOP))
    peaks, _ = signal.find_peaks(z, height=z_min, distance=dist)
    pre = int(0.04 * sr / HOP)
    post = int(0.03 * sr / HOP)
    rows = []
    for p in peaks:
        a, b = max(0, p - pre), min(len(env), p + post)
        before = bands[a:max(a + 1, p - 1)].mean(0)
        at = bands[p: p + 2].mean(0)
        after = bands[min(len(env) - 1, p + post)]
        rows.append(
            dict(
                t=float(t[p]), z=float(z[p]), flux=float(env[p]), bg=float(med[p]),
                rise_b0=float(at[0] - before[0]), rise_b1=float(at[1] - before[1]),
                rise_b2=float(at[2] - before[2]), rise_b3=float(at[3] - before[3]),
                decay_b2=float(at[2] - after[2]), lvl_b1=float(at[1]), lvl_b2=float(at[2]),
                sharp=float(env[p] / (env[a:b].mean() + 1e-6)),
            )
        )
    return pd.DataFrame(rows)


def run(video_id: str, force: bool = False) -> pd.DataFrame:
    out = match_dir(video_id) / "audio_onsets.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    wav = extract_audio(video_path(video_id), match_dir(video_id) / "audio.wav")
    x, sr = load_audio(wav)
    df = candidates(x, sr)
    df.to_parquet(out)
    return df
