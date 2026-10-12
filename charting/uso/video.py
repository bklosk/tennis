"""Video and audio access by presentation time.

Times are always seconds from the start of the file (stream PTS × time base); frame indices are
never used as time.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

from uso.paths import DOWNLOADS


def video_path(video_id: str) -> Path:
    return DOWNLOADS / f"{video_id}.mp4"


@dataclass
class Meta:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool

    @classmethod
    def probe(cls, path: Path) -> "Meta":
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout
        info = json.loads(out)
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
        return cls(
            width=int(v["width"]), height=int(v["height"]), fps=num / den if den else 0.0,
            duration=float(info["format"]["duration"]),
            has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
        )


def iter_frames(path: Path, start: float = 0.0, end: float | None = None, every: float = 0.0,
                size: tuple[int, int] | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (t, BGR frame) from start to end. `every` > 0 keeps at most one frame per `every` s.
    `size` = (w, h) resizes in the decoder."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        tb = float(s.time_base)
        if start > 0:
            c.seek(int(max(0.0, start - 1.0) / tb), stream=s, backward=True)
        nxt = start
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = frame.pts * tb
            if t < start - 1e-6:
                continue
            if end is not None and t > end:
                break
            if every > 0:
                if t + 1e-6 < nxt:
                    continue
                nxt += every
                if nxt <= t:
                    nxt = t + every
            if size is not None:
                img = frame.reformat(width=size[0], height=size[1], format="bgr24").to_ndarray()
            else:
                img = frame.to_ndarray(format="bgr24")
            yield t, img


def is_cfr(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate,avg_frame_rate",
         "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout.strip().split(",")
    f = [eval(x) if "/" in x and not x.endswith("/0") else 0.0 for x in out]  # noqa: S307 (ffprobe ratio)
    return len(f) == 2 and f[0] > 0 and abs(f[0] - f[1]) / f[0] < 0.002


def ff_frames(path: Path, fps: float, size: tuple[int, int], start: float = 0.0, end: float | None = None,
              hw: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    """Fast fixed-rate sampling through ffmpeg (multithreaded software decode, which beats
    VideoToolbox ~10x on an M3 Pro for 1080p H.264). Frame k is the source frame
    nearest to start + k/fps; for constant-frame-rate files (checked by is_cfr) that is exact to
    half a source frame. Use iter_frames/frames_at when exact presentation times matter."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error"]
    if hw:
        cmd += ["-hwaccel", "videotoolbox"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-an", "-vf", f"fps={fps},scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    # stderr is dropped: stopping early always ends in a broken-pipe message, and decode
    # problems are caught up front by scripts/verify_videos.py
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 8)
    n = w * h * 3
    k = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


def frames_at(path: Path, times: list[float], size: tuple[int, int] | None = None,
              tol: float = 0.02) -> dict[float, tuple[float, np.ndarray]]:
    """Nearest decoded frame for each requested time, decoding each contiguous window once.
    Returns {requested_t: (actual_t, frame)}."""
    times = sorted(times)
    out: dict[float, tuple[float, np.ndarray]] = {}
    if not times:
        return out
    # group requests into windows to avoid a seek per frame
    groups, cur = [], [times[0]]
    for t in times[1:]:
        if t - cur[-1] > 4.0:
            groups.append(cur)
            cur = [t]
        else:
            cur.append(t)
    groups.append(cur)
    for g in groups:
        want = list(g)
        prev = None
        for t, img in iter_frames(path, start=max(0.0, g[0] - 0.1), end=g[-1] + 0.1, size=size):
            while want and t >= want[0]:
                w = want.pop(0)
                if prev is not None and abs(prev[0] - w) < abs(t - w):
                    out[w] = prev
                else:
                    out[w] = (t, img)
            if not want:
                break
            prev = (t, img)
        for w in want:  # past the end
            if prev is not None:
                out[w] = prev
    return out


def extract_audio(path: Path, out_wav: Path, sr: int = 44100) -> Path:
    if not out_wav.exists():
        tmp = out_wav.with_suffix(".tmp.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(tmp)],
            check=True,
        )
        tmp.rename(out_wav)
    return out_wav


def load_audio(wav: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(wav), dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    return x, sr
