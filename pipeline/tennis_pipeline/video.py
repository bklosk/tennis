import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


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


def iter_frames(
    path: Path,
    fps: float,
    size: tuple[int, int],
    start: float | None = None,
    duration: float | None = None,
) -> Iterator[np.ndarray]:
    """Yield BGR frames resampled to `fps` and resized to `size` (width, height)."""
    width, height = size
    cmd = ["ffmpeg", "-v", "error"]
    if sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
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
    if not frames:
        return np.zeros((0, size[1], size[0], 3), np.uint8)
    return np.stack(frames)
