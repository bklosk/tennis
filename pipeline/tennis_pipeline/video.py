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


def probe(path: Path) -> dict:
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
    """
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
