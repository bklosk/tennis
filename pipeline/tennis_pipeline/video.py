import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

T = TypeVar("T")
R = TypeVar("R")
PACK_INDEX = "index.json"


def probe(path: Path) -> dict:
    if is_pack(path):
        return dict(_pack_index(str(path))["source"])
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    stream = info["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(num) / float(den),
        "duration": float(info["format"]["duration"]),
    }


@lru_cache
def hwaccel() -> tuple[str, ...]:
    """ffmpeg hardware-decode flags; software decoding unless TENNIS_HWACCEL opts in.

    Hardware decoders lose here because every frame is copied back and scaled on the CPU.
    720p60 -> 30 fps chunks: NVDEC 224 vs software 525 fps on an L40S droplet (8 vCPUs);
    VideoToolbox 290 vs software 989 fps on an M3 Pro. The 2 fps scene sampling is worse still
    on VideoToolbox: 5x vs 86x realtime. TENNIS_HWACCEL=cuda or =videotoolbox opts in anyway.
    """
    mode = os.environ.get("TENNIS_HWACCEL", "none")
    if mode == "videotoolbox" and sys.platform == "darwin":
        return ("-hwaccel", "videotoolbox")
    if mode == "cuda" and shutil.which("nvidia-smi"):
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
        if "cuda" in out.split():
            return ("-hwaccel", "cuda")
    return ()


def iter_frames(
    path: Path,
    fps: float,
    size: tuple[int, int],
    start: float | None = None,
    duration: float | None = None,
    accel: tuple[str, ...] | None = None,
    keyframes_only: bool = False,
) -> Iterator[np.ndarray]:
    """Yield BGR frames resampled to `fps` and resized to `size` (width, height).

    `keyframes_only` decodes only keyframes (~6x faster); the fps filter repeats each keyframe
    until the next, so temporal resolution drops to the stream's keyframe interval.

    `path` may be a main-camera pack (see `pack_segments`); times stay in the source video's
    timeline, and a read is limited to the pack part that contains `start`.
    """
    if is_pack(path):
        if start is None:
            raise ValueError(f"{path} holds main-camera segments only; reads need a start time")
        part = _pack_part(path, start, duration)
        if part is None:
            return
        path, start, duration = part
    width, height = size
    cmd = ["ffmpeg", "-v", "error", *(hwaccel() if accel is None else accel)]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += [
        "-vf", f"fps={fps},scale={width}:{height}",
        "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 8)
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_clip(path: Path, start: float, duration: float, fps: float = 30.0,
              size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    frames = list(iter_frames(path, fps, size, start=start, duration=duration))
    if not frames and hwaccel():
        # Some streams or drivers reject hardware decoding; fall back to software.
        frames = list(iter_frames(path, fps, size, start=start, duration=duration, accel=()))
    if not frames:
        return np.zeros((0, size[1], size[0], 3), np.uint8)
    return np.stack(frames)


def is_pack(path: Path | str) -> bool:
    return (Path(path) / PACK_INDEX).is_file()


@lru_cache(maxsize=256)
def _pack_index(path: str) -> dict:
    return json.loads((Path(path) / PACK_INDEX).read_text())


def _pack_part(path: Path, start: float, duration: float | None):
    """(part file, start within it, duration clipped to it) for a read at source time `start`."""
    for part in _pack_index(str(path))["parts"]:
        if part["start"] <= start < part["end"]:
            dur = part["end"] - start if duration is None else min(duration, part["end"] - start)
            return Path(path) / part["file"], start - part["start"], dur
    return None


def keyframe_times(path: Path) -> np.ndarray:
    """Video keyframe times in the pipeline's timeline (seconds from the container start)."""
    fmt = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=start_time", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout.strip()
    origin = float(fmt) if fmt not in ("", "N/A") else 0.0
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,flags",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    times = [float(t) for t, flags in (ln.split(",")[:2] for ln in out.splitlines() if "," in ln)
             if "K" in flags and t not in ("", "N/A")]
    return np.array(sorted(times)) - origin


def pack_segments(src: Path, segments: list[tuple[float, float]], out_dir: Path, pad: float = 2.0,
                  merge_gap: float = 8.0) -> dict:
    """Copy only the given time spans of `src` (no re-encode, no audio) into `out_dir`.

    Each part starts on a keyframe at or before its span, so it decodes exactly like the source:
    at the native frame rate, reads at source time t return the same frames as reads from `src`.
    Resampled to 30 fps, a read can pick the neighbouring 59.94 fps frame instead in some
    stretches (the two resamplers' phases differ by a timestamp tick). Spans closer than
    `merge_gap` share a part. `probe` of the pack reports the source video's properties.
    """
    info = probe(src)
    keys = keyframe_times(src)
    spans = []
    for s, e in sorted(segments):
        s, e = max(s - pad, 0.0), min(e + pad, info["duration"])
        if spans and s - spans[-1][1] < merge_gap:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, (s, e) in enumerate(spans):
        before = keys[keys <= s + 1e-6]
        k0 = float(before[-1]) if len(before) else 0.0
        name = f"part{k:04d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{k0:.6f}", "-i", str(src), "-t", f"{e - k0:.3f}",
                        "-map", "0:v:0", "-c", "copy", "-an", "-avoid_negative_ts", "make_zero", str(out_dir / name)],
                       check=True)
        parts.append({"file": name, "start": round(k0, 6), "end": round(e, 3)})
    index = {"source": info, "source_bytes": src.stat().st_size, "parts": parts,
             "bytes": int(sum((out_dir / p["file"]).stat().st_size for p in parts))}
    (out_dir / PACK_INDEX).write_text(json.dumps(index, indent=1))
    _pack_index.cache_clear()
    return index


def prefetch(items: Iterable[T], load: Callable[[T], R]) -> Iterator[tuple[T, R]]:
    """Yield (item, load(item)) while the next item loads in a background thread.

    Decoding runs in an ffmpeg subprocess, so the reader thread overlaps GPU work on the
    previous chunk without contending for the GIL.
    """
    items = list(items)
    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load, items[0])
        for i, item in enumerate(items):
            result = future.result()
            if i + 1 < len(items):
                future = pool.submit(load, items[i + 1])
            yield item, result
