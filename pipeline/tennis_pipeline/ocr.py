"""OCR of on-screen broadcast graphics: the score bug and the serve-speed readout.

For matches without official point-by-point data (all pre-2011 matches, plus AO 2024 and 2025+)
this produces the equivalent table from the broadcast itself:

1. discover  sample ~160 frames across the match, OCR them whole, and locate the score bug
             (rows starting with a player's surname) and the speed graphic ("NNN MPH/KM/H")
             by where those texts keep appearing;
2. read      OCR only those two regions once per second (reusing the last parse while the crop
             is unchanged);
3. points    debounce score reads into on-screen states and expand them into points with
             tennis scoring rules (`score.py`); attach each serve-speed readout to the point in
             which it appeared.

Outputs in outputs/VIDEO_ID/: ocr_regions.json, ocr_reads.parquet, ocr_points.csv,
ocr_summary.json. `align` uses ocr_points.csv automatically when official data is missing.
"""
import difflib
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import score, video
from .paths import match_dir

SPEED_RE = re.compile(r"(\d{2,3})\s*(M\s*\.?\s*P\s*\.?\s*H|KM\s*/?\s*H|KMH|KPH)", re.I)
POINT_TOKENS = {"0", "00", "O", "15", "30", "40", "AD", "A", "ADV"}
MPH_TO_KMH = 1.609344


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2

    @property
    def h(self):
        return self.y1 - self.y0


class RapidEngine:
    """PP-OCR (detection + recognition) via RapidOCR/onnxruntime; CPU is fast enough on crops."""

    def __init__(self):
        from rapidocr import RapidOCR

        self.engine = RapidOCR()

    def read(self, img: np.ndarray) -> list[Word]:
        # Flags are passed every call: RapidOCR keeps the last call's use_det/use_rec settings.
        r = self.engine(img, use_det=True, use_cls=True, use_rec=True)
        # With no detections RapidOCR returns a recognition-only result that has no boxes.
        if r is None or r.txts is None or getattr(r, "boxes", None) is None:
            return []
        out = []
        for text, box, conf in zip(r.txts, r.boxes, r.scores):
            box = np.asarray(box)
            out.append(Word(str(text), *box.min(0), *box.max(0), float(conf)))
        return out

    def recognize(self, img: np.ndarray) -> tuple[str, float]:
        """Recognition only, for a crop known to hold one short text (a score cell)."""
        if img.size == 0:
            return "", 0.0
        r = self.engine(img, use_det=False, use_cls=False, use_rec=True)
        if r is None or not r.txts:
            return "", 0.0
        return str(r.txts[0]), float(r.scores[0])


class VLMEngine:
    """Qwen3-VL transcription; slower, but reads stylised or low-resolution graphics better."""

    PROMPT = ("Transcribe all text in this image exactly as shown, one line per row of text, "
              "keeping numbers in order from left to right. Output only the text.")

    def __init__(self, model_id: str | None = None):
        from .vlm import DEFAULT_MODEL, VLM

        self.vlm = VLM(model_id or DEFAULT_MODEL)

    def read(self, img: np.ndarray) -> list[Word]:
        with tempfile.NamedTemporaryFile(suffix=".png") as fh:
            cv2.imwrite(fh.name, img)
            text = self.vlm.ask(fh.name, self.PROMPT, max_tokens=80)
        h = img.shape[0]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        step = h / max(len(lines), 1)
        return [Word(ln, 0, k * step, img.shape[1], (k + 1) * step, 1.0) for k, ln in enumerate(lines)]

    def recognize(self, img: np.ndarray) -> tuple[str, float]:
        with tempfile.NamedTemporaryFile(suffix=".png") as fh:
            cv2.imwrite(fh.name, img)
            return self.vlm.ask(fh.name, "Read the text in this image. Output only the text.", max_tokens=8), 1.0


def make_engine(name: str):
    return VLMEngine() if name == "vlm" else RapidEngine()


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


def surname_keys(name: str) -> list[str]:
    """Candidate on-screen forms of a player's name: surname (and compound surname parts)."""
    parts = [p for p in re.split(r"[\s\-]+", name) if p]
    keys = {_norm(parts[-1])}
    if len(parts) >= 3:
        keys.add(_norm("".join(parts[1:])))
        keys.add(_norm(parts[-2]))
    return [k for k in keys if len(k) >= 2]


def name_match(token: str, player: str) -> float:
    t = _norm(token)
    if len(t) < 3:
        return 0.0
    best = 0.0
    for key in surname_keys(player):
        if t == key:
            return 1.0
        if len(t) <= 4 and key.startswith(t):  # three-letter codes: "AGA", "FED"
            best = max(best, 0.85)
        best = max(best, difflib.SequenceMatcher(None, t, key).ratio())
    return best


def group_rows(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    med_h = float(np.median([w.h for w in words])) or 1.0
    rows = []
    for w in sorted(words, key=lambda w: w.yc):
        if rows and abs(w.yc - np.mean([v.yc for v in rows[-1]])) < 0.6 * med_h:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def parse_speed(words: list[Word]) -> float | None:
    """Serve speed in km/h from a speed graphic, or None."""
    for row in group_rows(words):
        m = SPEED_RE.search(" ".join(w.text for w in row))
        if not m:
            continue
        v = float(m.group(1))
        kmh = v if m.group(2).upper().startswith("K") else v * MPH_TO_KMH
        if 90 <= kmh <= 265:
            return round(kmh, 1)
    return None


def _tokens(row: list[Word]) -> list[str]:
    toks = []
    for w in row:
        toks += re.findall(r"[A-Za-z][A-Za-z.'\-]*|\d+", w.text)
    return toks


def parse_score(words: list[Word], player1: str, player2: str, points_column: bool | None = None) -> dict | None:
    """Games per set and current point score for both players from a score bug.

    Returns {"games1", "games2", "pts1", "pts2", "has_points"} or None. `points_column` forces
    whether the last number column is the point score (decided per match from how often
    15/30/40/AD appear); None guesses from this read alone.
    """
    found = {}
    for row in group_rows(words):
        toks = _tokens(row)
        names = [t for t in toks if not t.isdigit() and t.upper() not in POINT_TOKENS]
        if not names:
            continue
        s1 = max(name_match(t, player1) for t in names)
        s2 = max(name_match(t, player2) for t in names)
        who, s = (1, s1) if s1 >= s2 else (2, s2)
        if s < 0.8 or who in found:
            continue
        last_name = max(i for i, t in enumerate(toks) if t in names)
        found[who] = [t.upper() for t in toks[last_name + 1:] if t.isdigit() or t.upper() in POINT_TOKENS]
    if 1 not in found or 2 not in found:
        return None
    a, b = found[1], found[2]
    if not a or not b:
        return None
    if points_column is None:
        points_column = (a[-1] in POINT_TOKENS - {"0", "00", "O"} or b[-1] in POINT_TOKENS - {"0", "00", "O"})
    pts1 = pts2 = None
    if points_column and len(a) == len(b) and len(a) >= 2:
        pts1, pts2 = a[-1], b[-1]
        a, b = a[:-1], b[:-1]
    if len(a) != len(b) or not all(t.isdigit() for t in a + b):
        return None
    return {"games1": [int(t) for t in a], "games2": [int(t) for t in b], "pts1": pts1, "pts2": pts2,
            "has_points": pts1 is not None}


def _union(boxes: list[tuple], pad: int, shape) -> list[int] | None:
    if not boxes:
        return None
    arr = np.array(boxes)
    centre = np.median((arr[:, :2] + arr[:, 2:]) / 2, 0)
    keep = arr[np.hypot(*(((arr[:, :2] + arr[:, 2:]) / 2 - centre).T)) < 80]
    if len(keep) < 2:
        return None
    x0, y0 = keep[:, :2].min(0) - pad
    x1, y1 = keep[:, 2:].max(0) + pad
    return [int(max(x0, 0)), int(max(y0, 0)), int(min(x1, shape[1])), int(min(y1, shape[0]))]


def split_word(w: Word) -> list[Word]:
    """Split a multi-token detection ("6 3 40") into tokens with x-extents by character position."""
    toks = list(re.finditer(r"\S+", w.text))
    if len(toks) <= 1:
        return [w]
    per = (w.x1 - w.x0) / max(len(w.text), 1)
    return [Word(m.group(), w.x0 + m.start() * per, w.y0, w.x0 + m.end() * per, w.y1, w.conf) for m in toks]


def _is_score_token(t: str) -> bool:
    return t.isdigit() or t.upper() in POINT_TOKENS


def _columns(num_boxes: list[tuple], region: list[int], min_share: float = 0.05) -> list[list[int]]:
    """x-extents (relative to the region) of the score bug's number columns."""
    boxes = [b for b in num_boxes if region[0] <= (b[0] + b[2]) / 2 <= region[2]
             and region[1] <= (b[1] + b[3]) / 2 <= region[3]]
    if len(boxes) < 4:
        return []
    boxes.sort(key=lambda b: (b[0] + b[2]) / 2)
    med_h = float(np.median([b[3] - b[1] for b in boxes]))
    clusters = [[boxes[0]]]
    for b in boxes[1:]:
        last = clusters[-1][-1]
        if (b[0] + b[2]) / 2 - (last[0] + last[2]) / 2 > 0.8 * med_h:
            clusters.append([b])
        else:
            clusters[-1].append(b)
    cols = []
    for c in clusters:
        if len(c) < max(2, min_share * len(boxes)):
            continue
        x0 = min(b[0] for b in c) - 0.3 * med_h - region[0]
        x1 = max(b[2] for b in c) + 0.3 * med_h - region[0]
        cols.append([int(max(x0, 0)), int(min(x1, region[2] - region[0]))])
    return cols


def discover_regions(frames: list[np.ndarray], engine, player1: str, player2: str) -> dict:
    """Locate the score bug (and its number columns) and the speed graphic on sampled frames."""
    score_boxes, speed_boxes, num_boxes = [], [], []
    shape = frames[0].shape
    for img in frames:
        words = engine.read(img)
        for row in group_rows(words):
            text = " ".join(w.text for w in row)
            if SPEED_RE.search(text):
                speed_boxes.append((min(w.x0 for w in row), min(w.y0 for w in row),
                                    max(w.x1 for w in row), max(w.y1 for w in row)))
            toks = [t for t in _tokens(row) if not t.isdigit()]
            if toks and max(max(name_match(t, player1), name_match(t, player2)) for t in toks) >= 0.8:
                # Include the number columns to the right of the name on the same row.
                score_boxes.append((min(w.x0 for w in row), min(w.y0 for w in row),
                                    max(w.x1 for w in row), max(w.y1 for w in row)))
                num_boxes += [(p.x0, p.y0, p.x1, p.y1) for w in row for p in split_word(w) if _is_score_token(p.text)]
    score_region = _union(score_boxes, 12, shape)
    return {"score": score_region, "speed": _union(speed_boxes, 10, shape),
            "score_columns": _columns(num_boxes, score_region) if score_region else [],
            "n_frames": len(frames), "score_hits": len(score_boxes), "speed_hits": len(speed_boxes)}


def _crop(img: np.ndarray, box: list[int], min_h: int = 48) -> tuple[np.ndarray, float]:
    x0, y0, x1, y1 = box
    c = img[y0:y1, x0:x1]
    f = 1.0
    if c.size and c.shape[0] < min_h:
        f = min_h / c.shape[0]
        c = cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return c, f


def read_score(crop: np.ndarray, engine, columns: list[list[int]], scale: float = 1.0) -> list[Word]:
    """Name words from detection; each number cell read on its own when columns are known.

    Text detectors drop isolated single characters ("0"), so cells are recognised directly.
    """
    words = engine.read(crop)
    out = []
    for row in group_rows(words):
        parts = [p for w in row for p in split_word(w)]
        names = [p for p in parts if not _is_score_token(p.text)]
        if not names:
            continue
        out += names
        if not columns:
            out += [p for p in parts if _is_score_token(p.text)]
            continue
        y0, y1 = min(w.y0 for w in row), max(w.y1 for w in row)
        pad = 0.2 * (y1 - y0)
        ya, yb = int(max(y0 - pad, 0)), int(min(y1 + pad, crop.shape[0]))
        for cx0, cx1 in columns:
            xa, xb = int(cx0 * scale), int(cx1 * scale)
            text, conf = engine.recognize(crop[ya:yb, xa:xb])
            tok = re.sub(r"[^0-9A-Za-z]", "", text).upper()
            if tok and conf >= 0.5 and _is_score_token(tok):
                out.append(Word(tok, xa, y0, xb, y1, conf))
    return out


def _changed(prev: np.ndarray | None, cur: np.ndarray, level: int = 40, share: float = 0.002) -> bool:
    return prev is None or prev.shape != cur.shape or (np.abs(prev - cur) > level).mean() > share


def read_regions(frame_iter, regions: dict, engine, player1: str, player2: str) -> pd.DataFrame:
    """OCR the discovered regions on each (t, frame); reuse a parse while the crop is unchanged."""
    rows, last = [], {}
    for t, img in frame_iter:
        rec = {"t": float(t)}
        for kind in ("score", "speed"):
            box = regions.get(kind)
            if not box:
                continue
            crop, scale = _crop(img, box)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[::2, ::2].astype(np.int16)
            prev = last.get(kind)
            if not _changed(prev[0] if prev else None, gray):
                words = prev[1]
            else:
                words = (read_score(crop, engine, regions.get("score_columns") or [], scale)
                         if kind == "score" else engine.read(crop))
                last[kind] = (gray, words)
            rec[f"{kind}_text"] = " | ".join(w.text for w in words)
            if kind == "speed":
                rec["speed_kmh"] = parse_speed(words)
                rec["speed_server"] = speed_server(words, player1, player2)
            else:
                rec["score_words"] = words
        rows.append(rec)
    return pd.DataFrame(rows)


def speed_server(words: list[Word], player1: str, player2: str) -> int | None:
    """Some speed graphics name the server ("AGASSI 121 MPH"); 1 or 2 if so."""
    toks = [t for w in words for t in _tokens([w]) if not t.isdigit()]
    s1 = max((name_match(t, player1) for t in toks), default=0.0)
    s2 = max((name_match(t, player2) for t in toks), default=0.0)
    if max(s1, s2) < 0.8 or s1 == s2:
        return None
    return 1 if s1 > s2 else 2


def _speed_events(reads: pd.DataFrame, same_graphic_s: float = 6.0) -> list[tuple]:
    """(first time shown, km/h, named server or None) per speed graphic; repeat reads collapse."""
    out = []
    if "speed_kmh" not in reads:
        return out
    for r in reads.dropna(subset=["speed_kmh"]).itertuples():
        srv = getattr(r, "speed_server", None)
        srv = None if srv is None or pd.isna(srv) else int(srv)
        if out and abs(r.speed_kmh - out[-1][1]) < 0.5 and r.t - out[-1][3] <= same_graphic_s:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] or srv, r.t)
            continue
        out.append((r.t, r.speed_kmh, srv, r.t))
    return [(t0, v, s) for t0, v, s, _ in out]


def attach_speeds(points: pd.DataFrame, speeds: list[tuple]) -> pd.DataFrame:
    """Serve speed of each point: the last readout shown before its score change (2+ => second serve)."""
    points = points.copy()
    points["Speed_KMH"] = np.nan
    points["ServeNumber"] = np.nan
    points["server_hint"] = np.nan
    if points.empty or not speeds:
        return points
    t_hi = points.video_t_hi.to_numpy(float)
    bucket = {}
    for ev in speeds:
        ts, v, srv = ev[0], ev[1], ev[2] if len(ev) > 2 else None
        j = int(np.searchsorted(t_hi, ts - 1.0))
        if j < len(points):
            bucket.setdefault(j, []).append((v, srv))
    # Several points can share one score change (unseen points); speeds go to the last of them.
    for j, vs in bucket.items():
        k = points.index[np.where(t_hi == t_hi[j])[0][-1]]
        points.loc[k, "Speed_KMH"] = vs[-1][0]
        points.loc[k, "ServeNumber"] = min(len(vs), 2)
        named = [s for _, s in vs if s is not None]
        if named:
            points.loc[k, "server_hint"] = named[-1]
    return points


def build_points(reads: pd.DataFrame, meta: dict, rules: score.Rules) -> tuple[pd.DataFrame, dict]:
    p1, p2 = meta["player1"], meta["player2"]
    parsed = [parse_score(w, p1, p2) if isinstance(w, list) else None
              for w in reads.get("score_words", pd.Series([None] * len(reads)))]
    with_pts = [p for p in parsed if p]
    points_column = bool(with_pts) and np.mean([p["has_points"] for p in with_pts]) > 0.2
    states = []
    for w, p in zip(reads.get("score_words", [None] * len(reads)), parsed):
        if p and p["has_points"] != points_column:
            p = parse_score(w, p1, p2, points_column)
        states.append(score.parse_state(p["games1"], p["games2"], p["pts1"], p["pts2"], rules) if p else None)
    timeline = pd.DataFrame({"t": reads.t, "state": states})
    stable = score.stable_states(timeline, rules)
    points = score.points_from_states(stable, rules, f"ocr-{meta['video_id']}")
    speeds = _speed_events(reads)
    points = attach_speeds(points, speeds)
    stats = {"samples": int(len(reads)), "score_parsed": int(sum(s is not None for s in states)),
             "score_states": len(stable), "points": int(len(points)),
             "points_inferred": int(points.inferred.sum()) if len(points) else 0,
             "score_breaks": int(points.attrs.get("breaks", 0)) if len(points) else 0,
             "speed_graphics": len(speeds),
             "points_with_speed": int(points.Speed_KMH.notna().sum()) if len(points) else 0,
             "points_column": points_column}
    return points, stats


def run(video_id: str, video_path: Path, engine_name: str = "rapidocr", fps: float = 1.0,
        n_discover: int = 160) -> dict:
    from .align import video_match

    meta = video_match(video_id)
    meta["video_id"] = video_id
    rules = score.rules_for(meta["tournament"], int(meta["year"]), meta["draw"])
    out_dir = match_dir(video_id)
    engine = make_engine(engine_name)
    reg_path = out_dir / "ocr_regions.json"
    if reg_path.exists():
        regions = json.loads(reg_path.read_text())
    else:
        dur = video.probe(video_path)["duration"]
        times = np.linspace(min(30, dur * 0.05), dur - 5, n_discover)
        frames = [f for t in times for f in video.read_clip(video_path, float(t), 1.5 / 30)[:1]]
        regions = discover_regions(frames, engine, meta["player1"], meta["player2"])
        reg_path.write_text(json.dumps(regions, indent=2))
    if not regions.get("score"):
        summary = {"video_id": video_id, "error": "score bug not found", **regions}
        (out_dir / "ocr_summary.json").write_text(json.dumps(summary, indent=2))
        return summary
    frame_iter = ((k / fps, f) for k, f in enumerate(video.iter_frames(video_path, fps, (1280, 720))))
    reads = read_regions(frame_iter, regions, engine, meta["player1"], meta["player2"])
    stored = reads.drop(columns=["score_words"], errors="ignore")
    stored["score_words_json"] = [json.dumps([[w.text, w.x0, w.y0, w.x1, w.y1, w.conf] for w in ws])
                                  if isinstance(ws, list) else None for ws in reads.get("score_words", [])]
    stored.to_parquet(out_dir / "ocr_reads.parquet", index=False)
    return _finish(video_id, meta, rules, reads, {"engine": engine_name, "regions": regions})


def rebuild(video_id: str) -> dict:
    """Re-derive points from cached reads (after parser or rule changes) without re-running OCR."""
    from .align import video_match

    meta = video_match(video_id)
    meta["video_id"] = video_id
    rules = score.rules_for(meta["tournament"], int(meta["year"]), meta["draw"])
    reads = pd.read_parquet(match_dir(video_id) / "ocr_reads.parquet")
    reads["score_words"] = [[Word(t, float(a), float(b), float(c), float(d), float(e)) for t, a, b, c, d, e in json.loads(s)]
                            if isinstance(s, str) else None for s in reads.score_words_json]
    return _finish(video_id, meta, rules, reads, {"rebuilt": True})


def _finish(video_id: str, meta: dict, rules: score.Rules, reads: pd.DataFrame, extra: dict) -> dict:
    out_dir = match_dir(video_id)
    points, stats = build_points(reads, meta, rules)
    points.to_csv(out_dir / "ocr_points.csv", index=False)
    summary = {"video_id": video_id, "rules": rules.__dict__, **extra, **stats}
    (out_dir / "ocr_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
