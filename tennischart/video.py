"""Video and audio I/O through ffmpeg/ffprobe. Times are seconds from the start of the file.

Frames are decoded by presentation timestamp with ffmpeg's `fps` filter, so variable frame rate
rips come out on a regular clock. Interlaced sources are bob-deinterlaced (`yadif=1`), black
bars are cropped and non-square pixels are resampled to square before anything sees a frame.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Iterator

import numpy as np


class FFmpegError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FFmpegError(f"{tool} not found on PATH (install ffmpeg)")
    return path


def _ratio(s: str | None, default: float = 0.0) -> float:
    if not s or s in ("0/0", "N/A"):
        return default
    try:
        return float(Fraction(s.replace(":", "/")))
    except (ValueError, ZeroDivisionError):
        return default


@dataclass
class VideoInfo:
    path: str
    duration: float
    width: int
    height: int
    sar: float = 1.0
    fps: float = 30.0            # nominal (r_frame_rate)
    avg_fps: float = 30.0
    field_order: str = "progressive"
    interlaced: bool = False
    has_audio: bool = True
    audio_rate: int = 0
    codec: str = ""
    crop: tuple[int, int, int, int] | None = None   # w, h, x, y in source pixels
    extra: dict = field(default_factory=dict)

    @property
    def vfr(self) -> bool:
        return abs(self.fps - self.avg_fps) > 0.05 * max(self.fps, 1e-6)

    @property
    def display_size(self) -> tuple[float, float]:
        w, h = (self.crop[0], self.crop[1]) if self.crop else (self.width, self.height)
        return w * self.sar, float(h)

    def output_size(self, width: int | None = None) -> tuple[int, int]:
        """Square-pixel output size, optionally scaled to `width` (both even)."""
        dw, dh = self.display_size
        if width is None:
            width = int(round(dw))
        height = dh * width / dw
        return int(width) // 2 * 2, int(round(height / 2)) * 2

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str, detect_interlace: bool = True, detect_bars: bool = True) -> VideoInfo:
    _require("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise FFmpegError(out.stderr.strip() or f"ffprobe failed on {path}")
    meta = json.loads(out.stdout)
    vs = [s for s in meta.get("streams", []) if s.get("codec_type") == "video"]
    aus = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    if not vs:
        raise FFmpegError(f"no video stream in {path}")
    v = vs[0]
    duration = float(meta.get("format", {}).get("duration") or v.get("duration") or 0.0)
    info = VideoInfo(
        path=path,
        duration=duration,
        width=int(v["width"]),
        height=int(v["height"]),
        sar=_ratio(v.get("sample_aspect_ratio"), 1.0) or 1.0,
        fps=_ratio(v.get("r_frame_rate"), 30.0) or 30.0,
        avg_fps=_ratio(v.get("avg_frame_rate"), 30.0) or 30.0,
        field_order=v.get("field_order", "progressive") or "progressive",
        has_audio=bool(aus),
        audio_rate=int(aus[0].get("sample_rate", 0)) if aus else 0,
        codec=v.get("codec_name", ""),
    )
    info.interlaced = info.field_order not in ("progressive", "unknown", "")
    if detect_interlace and duration > 0:
        try:
            info.interlaced = measure_interlace(path, duration) or info.interlaced
        except FFmpegError:
            pass
    if detect_bars and duration > 0:
        try:
            info.crop = detect_crop(path, duration, info.width, info.height)
        except FFmpegError:
            info.crop = None
    return info


def measure_interlace(path: str, duration: float, frames: int = 400) -> bool:
    """Run ffmpeg's idet on a stretch from the middle of the file (metadata lies for rips)."""
    start = max(0.0, duration * 0.4)
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-i", path, "-an",
         "-frames:v", str(frames), "-vf", "idet", "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.findall(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)",
                   out.stderr)
    if not m:
        return False
    tff, bff, prog = (int(x) for x in m[-1])
    return (tff + bff) > 1.5 * prog and (tff + bff) > 20


def detect_crop(path: str, duration: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Letterbox / pillarbox detection with cropdetect at a few points in the file."""
    votes: dict[tuple[int, int, int, int], int] = {}
    for frac in (0.2, 0.45, 0.7):
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{duration * frac:.3f}", "-i", path,
             "-an", "-frames:v", "60", "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
            capture_output=True, text=True)
        m = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out.stderr)
        if m:
            c = tuple(int(x) for x in m[-1])
            votes[c] = votes.get(c, 0) + 1  # type: ignore[index]
    if not votes:
        return None
    w, h, x, y = max(votes.items(), key=lambda kv: kv[1])[0]
    if w >= width - 8 and h >= height - 8:
        return None
    if w < width * 0.5 or h < height * 0.5:
        return None
    return (w, h, x, y)


def _filters(info: VideoInfo, fps: float | None, width: int | None) -> tuple[str, int, int]:
    f = []
    if info.interlaced:
        f.append("yadif=1")
    if info.crop:
        w, h, x, y = info.crop
        f.append(f"crop={w}:{h}:{x}:{y}")
    ow, oh = info.output_size(width)
    f.append(f"scale={ow}:{oh}:flags=area,setsar=1")
    if fps:
        f.append(f"fps={fps}")
    return ",".join(f), ow, oh


def frames(info: VideoInfo, fps: float, start: float = 0.0, duration: float | None = None,
           width: int | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time, RGB uint8 frame) on a regular `fps` clock from `start` for `duration` s."""
    _require("ffmpeg")
    vf, ow, oh = _filters(info, fps, width)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", info.path]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-sn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = ow * oh * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 4)
    k = 0
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
            k += 1
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


def frame_at(info: VideoInfo, t: float, width: int | None = None) -> np.ndarray:
    for _, fr in frames(info, fps=max(info.fps, 1.0), start=max(0.0, t), duration=1.0 / max(info.fps, 1.0) * 1.5,
                        width=width):
        return fr
    raise FFmpegError(f"no frame at {t:.3f}s in {info.path}")


def window(info: VideoInfo, t0: float, t1: float, fps: float, width: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All frames in [t0, t1) on an `fps` clock: (times, frames[N, H, W, 3])."""
    ts, fs = [], []
    for t, fr in frames(info, fps=fps, start=max(0.0, t0), duration=max(0.0, t1 - max(0.0, t0)), width=width):
        ts.append(t)
        fs.append(fr)
    if not fs:
        return np.zeros(0), np.zeros((0, 1, 1, 3), np.uint8)
    return np.array(ts), np.stack(fs)


def audio_chunks(path: str, sr: int = 32000, chunk_s: float = 60.0) -> Iterator[np.ndarray]:
    """Mono float32 audio in chunks of `chunk_s` seconds (last chunk may be shorter)."""
    _require("ffmpeg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path, "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = int(sr * chunk_s) * 2
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(n)
            if not buf:
                break
            if len(buf) % 2:
                buf = buf[:-1]
            yield np.frombuffer(buf, np.int16).astype(np.float32) / 32768.0
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.kill()
        proc.wait()


def read_audio(path: str, sr: int = 32000) -> np.ndarray:
    chunks = list(audio_chunks(path, sr))
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def save_jpeg(rgb: np.ndarray, path: str, quality: int = 90) -> None:
    from PIL import Image

    Image.fromarray(rgb).save(path, quality=quality)
