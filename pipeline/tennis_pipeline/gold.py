"""Expand the stroke gold set to ~200 forehand/backhand labels with Qwen3-VL.

The pilot showed Qwen3-VL-8B cannot be asked "forehand or backhand?" (it says forehand) and has
a strong "left" bias when asked which side the racket is on (61% vs gold). This protocol only
keeps answers that survive checks that bias cannot pass:

- crops are re-decoded at the source resolution (near players are ~250 px tall at 1080p);
- every question is asked on the crop and on its mirror image; an answer counts only if it
  flips with the mirror;
- two independent questions vote: the racket's image side (one word), and racket-head and
  torso points (grounding), whose x difference gives the side;
- "both hands on the racket?" vetoes a forehand label (two-handed forehands are rare);
- the image side becomes a stroke using which end the player is at (the far player faces the
  camera) and handedness from the Sackmann player file.

Before anything is added, `calibrate` runs the same protocol on the existing hand-checked gold
labels; the expansion refuses to write unless accepted labels agree >= 90% there.

Sampling spreads labels over broadcast eras (SD 2000-06, early HD 2007-12, HD 2013-26), favours
near-court players (60%) and left-handers (30%), and round-robins across matches.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import events, video
from .align import video_match
from .paths import CACHE, match_dir

GOLD_PATH = Path(__file__).resolve().parents[1] / "eval" / "stroke_gold.csv"
GOLD_COLUMNS = ["video_id", "hit_id", "stroke", "labeler", "sheet", "era", "side", "hand", "confidence", "votes"]
CROP_DIR = CACHE / "gold_crops"
SHEET_DIR = CACHE / "gold_sheets"
VOTES_PATH = CACHE / "gold_vlm_votes.jsonl"
ERAS = (("2000-06", 2000, 2006), ("2007-12", 2007, 2012), ("2013-26", 2013, 2026))
MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"

SIDE_PROMPT = (
    "This image shows a tennis player at the moment of hitting the ball. Where is the racket "
    "relative to the player's body: on the LEFT side of the image or the RIGHT side of the image? "
    "Answer with one word: left or right."
)
POINT_PROMPT = (
    "Locate the head of the tennis racket and the centre of the player's chest in this image. "
    'Answer only with JSON like {"racket": [x, y], "chest": [x, y]}.'
)
TWO_HANDS_PROMPT = "Is the tennis player holding the racket with both hands? Answer with one word: yes or no."
FLIP = {"left": "right", "right": "left"}


def era_of(year: int) -> str:
    for name, a, b in ERAS:
        if a <= year <= b:
            return name
    return "other"


def load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_PATH) if GOLD_PATH.exists() else pd.DataFrame(columns=GOLD_COLUMNS)
    for c in GOLD_COLUMNS:
        if c not in df:
            df[c] = None
    return df[GOLD_COLUMNS]


def candidate_shots(video_ids: list[str]) -> pd.DataFrame:
    frames = []
    for vid in video_ids:
        path = match_dir(vid) / "shots_aligned.parquet"
        if not path.exists():
            print(f"{vid}: no shots_aligned.parquet; run track/events/crops/align first")
            continue
        s = pd.read_parquet(path)
        s["video_id"] = vid
        s["year"] = int(video_match(vid)["year"])
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    s = pd.concat(frames, ignore_index=True)
    s = s[~s.is_serve.astype(bool) & s.box.notna() & s.hand.isin(["R", "L"]) & s.point_number.notna()]
    s = s.copy()
    s["era"] = s.year.map(era_of)
    return s


def image_side_to_stroke(side_img: str, court_side: str, hand: str) -> str:
    """Racket image side -> stroke. The near player shows their back, the far player their front."""
    player_side = side_img if court_side == "near" else FLIP[side_img]
    return "forehand" if player_side == ("right" if hand == "R" else "left") else "backhand"


def prepare_crop(r, video_path) -> tuple[str, str] | None:
    """Contact-frame hitter crop at source resolution, plus its mirror image."""
    out = CROP_DIR / f"{r.video_id}_{r.hit_id}.jpg"
    mirror = out.with_name(out.stem + "_m.jpg")
    if not out.exists():
        info = video.probe(video_path)
        w, h = info["width"], info["height"]
        clip = video.read_clip(video_path, max(float(r.t), 0), 1.5 / 30, size=(w, h))
        if not len(clip):
            return None
        box = np.array(r.box, float) * np.array([w / 1280, h / 720, w / 1280, h / 720])
        crop, _ = events.hitter_crop(clip[0], box, out=512)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mirror), crop[:, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(out), str(mirror)


def _word(text: str, choices: tuple) -> str | None:
    t = text.strip().lower()
    for c in choices:
        if t.startswith(c) or re.search(rf"\b{c}\b", t[:20]):
            return c
    return None


def _point_side(text: str, margin: float = 0.02) -> str | None:
    """Image side of the racket from grounding output (coordinates may be 0-1000 or pixels)."""
    def xy(key):
        m = re.search(rf'"?{key}"?\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)', text, re.I)
        return (float(m.group(1)), float(m.group(2))) if m else None

    racket, chest = xy("racket"), xy("chest") or xy("torso")
    if racket is None or chest is None:
        return None
    scale = 1000.0 if max(racket[0], chest[0]) <= 1000 else 512.0
    dx = (racket[0] - chest[0]) / scale
    if abs(dx) < margin:
        return None
    return "right" if dx > 0 else "left"


def vlm_votes(vlm, crop: str, mirror: str) -> dict:
    side_a = _word(vlm.ask(crop, SIDE_PROMPT, max_tokens=4), ("left", "right"))
    side_b = _word(vlm.ask(mirror, SIDE_PROMPT, max_tokens=4), ("left", "right"))
    pt_a = _point_side(vlm.ask(crop, POINT_PROMPT, max_tokens=60))
    pt_b = _point_side(vlm.ask(mirror, POINT_PROMPT, max_tokens=60))
    two = _word(vlm.ask(crop, TWO_HANDS_PROMPT, max_tokens=4), ("yes", "no"))
    return {"side": side_a, "side_mirror": side_b, "point": pt_a, "point_mirror": pt_b, "two_hands": two}


def decide(v: dict, court_side: str, hand: str) -> tuple[str | None, str, str]:
    """(stroke or None, confidence, reason) from the raw votes."""
    consistent = []
    for a, b in ((v["side"], v["side_mirror"]), (v["point"], v["point_mirror"])):
        if a is not None and b is not None and b == FLIP[a]:
            consistent.append(a)
    if not consistent:
        return None, "none", "no_mirror_consistent_vote"
    if len(set(consistent)) > 1:
        return None, "none", "votes_disagree"
    stroke = image_side_to_stroke(consistent[0], court_side, hand)
    if stroke == "forehand" and v["two_hands"] == "yes":
        return None, "none", "two_hands_on_forehand"
    return stroke, "high" if len(consistent) == 2 else "medium", "ok"


def _votes_str(v: dict) -> str:
    ab = {"left": "L", "right": "R", None: "-", "yes": "Y", "no": "N"}
    return (f"side={ab[v['side']]}/{ab[v['side_mirror']]};pt={ab[v['point']]}/{ab[v['point_mirror']]};"
            f"two={ab[v['two_hands']]}")


class Labeler:
    def __init__(self, model_id: str = MODEL, video_path_fn=None):
        from .vlm import VLM

        self.vlm = VLM(model_id)
        self.model_id = model_id
        self.video_path_fn = video_path_fn
        self.cache = {}
        if VOTES_PATH.exists():
            for line in VOTES_PATH.read_text().splitlines():
                rec = json.loads(line)
                self.cache[(rec["model"], rec["key"])] = rec["votes"]

    def votes(self, r) -> dict | None:
        key = f"{r.video_id}_{r.hit_id}"
        if (self.model_id, key) in self.cache:
            return self.cache[(self.model_id, key)]
        crops = prepare_crop(r, self.video_path_fn(r.video_id))
        if crops is None:
            return None
        v = vlm_votes(self.vlm, *crops)
        self.cache[(self.model_id, key)] = v
        VOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with VOTES_PATH.open("a") as fh:
            fh.write(json.dumps({"model": self.model_id, "key": key, "votes": v}) + "\n")
        return v

    def label(self, r) -> dict:
        v = self.votes(r)
        if v is None:
            return {"stroke": None, "confidence": "none", "reason": "no_frame", "votes": ""}
        stroke, conf, reason = decide(v, r.side, r.hand)
        return {"stroke": stroke, "confidence": conf, "reason": reason, "votes": _votes_str(v)}


def calibrate(labeler: Labeler, shots: pd.DataFrame, gold: pd.DataFrame) -> dict:
    """Protocol accuracy on existing human-reviewed gold labels (the go/no-go for expansion)."""
    human = gold[~gold.labeler.astype(str).str.startswith("qwen")]
    m = (human[["video_id", "hit_id", "stroke"]].rename(columns={"stroke": "stroke_gold"})
         .merge(shots.drop(columns=["stroke"], errors="ignore"), on=["video_id", "hit_id"]))
    rows = []
    for r in m.itertuples():
        res = labeler.label(r)
        rows.append({"side": r.side, "hand": r.hand, "gold": r.stroke_gold, **res})
    d = pd.DataFrame(rows)
    acc = d[d.stroke.notna()]
    return {
        "n_gold_available": int(len(d)),
        "accepted": int(len(acc)),
        "acceptance_rate": round(len(acc) / max(len(d), 1), 3),
        "accuracy_when_accepted": round(float((acc.stroke == acc.gold).mean()), 3) if len(acc) else None,
        "accuracy_by_side": acc.groupby("side").apply(lambda g: round(float((g.stroke == g.gold).mean()), 3),
                                                      include_groups=False).to_dict() if len(acc) else {},
        "reasons": d.reason.value_counts().to_dict() if len(d) else {},
    }


def plan_quotas(pool: pd.DataFrame, need: int, near_share: float = 0.6, lefty_share: float = 0.3) -> dict:
    """{(era, side, hand): target} spreading `need` over eras, then near/far and hand."""
    eras = sorted(pool.era.unique())
    if not eras or need <= 0:
        return {}
    weight = {}
    for era in eras:
        for side, s_share in (("near", near_share), ("far", 1 - near_share)):
            for hand, h_share in (("L", lefty_share), ("R", 1 - lefty_share)):
                weight[(era, side, hand)] = s_share * h_share / len(eras)
    avail = pool.groupby(["era", "side", "hand"]).size().to_dict()
    # Water-filling: strata that run out pass their share to the others in proportion to weight.
    targets = {k: 0.0 for k in weight}
    open_ = [k for k in weight if avail.get(k, 0) > 0]
    left = float(min(need, sum(avail.values())))
    while left > 1e-9 and open_:
        total_w = sum(weight[k] for k in open_)
        spent = 0.0
        for k in list(open_):
            add = min(left * weight[k] / total_w, avail.get(k, 0) - targets[k])
            targets[k] += add
            spent += add
            if avail.get(k, 0) - targets[k] <= 1e-9:
                open_.remove(k)
        left -= spent
        if spent <= 1e-9:
            break
    rounded = {k: int(np.floor(v)) for k, v in targets.items()}
    rest = need - sum(rounded.values())
    for k in sorted(targets, key=lambda k: targets[k] - rounded[k], reverse=True):
        if rest <= 0:
            break
        if rounded[k] < avail.get(k, 0):
            rounded[k] += 1
            rest -= 1
    return rounded


def _round_robin(g: pd.DataFrame, seed: int) -> pd.DataFrame:
    g = g.sample(frac=1, random_state=seed)
    g["_rank"] = g.groupby("video_id").cumcount()
    return g.sort_values("_rank")


def contact_sheets(rows: pd.DataFrame, per_sheet: int = 12):
    SHEET_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(0, len(rows), per_sheet):
        tiles = []
        for r in rows.iloc[k:k + per_sheet].itertuples():
            img = cv2.imread(str(CROP_DIR / f"{r.video_id}_{r.hit_id}.jpg"))
            if img is None:
                continue
            img = cv2.resize(img, (320, 320))
            cv2.rectangle(img, (0, 0), (320, 40), (0, 0, 0), -1)
            cv2.putText(img, f"{r.stroke} ({r.confidence})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"{r.era} {r.side} {r.hand}-hand {r.hit_id}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)
            tiles.append(img)
        if not tiles:
            continue
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.concatenate([np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)], 0)
        path = SHEET_DIR / f"gold2_sheet{k // per_sheet:02d}.jpg"
        cv2.imwrite(str(path), grid)
        paths.append(str(path))
    return paths


def expand(video_ids: list[str], video_path_fn, target: int = 200, model_id: str = MODEL, seed: int = 0,
           min_calibration: float = 0.9, force: bool = False) -> dict:
    shots = candidate_shots(video_ids)
    if shots.empty:
        raise SystemExit("no aligned shots found for the given matches")
    gold = load_gold()
    labeler = Labeler(model_id, video_path_fn)
    report = {"model": model_id, "calibration": calibrate(labeler, shots, gold)}
    acc = report["calibration"]["accuracy_when_accepted"]
    if not force and (acc is None or acc < min_calibration):
        report["written"] = 0
        report["stopped"] = (f"protocol agrees {acc} with existing gold (< {min_calibration}); not writing "
                             "labels. Try a larger Qwen3-VL (e.g. --model mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit) "
                             "or --force to write them for review anyway.")
        return report
    tried = set(zip(gold.video_id, gold.hit_id))
    new, attempts, reasons, quotas = [], 0, {}, {}
    # The VLM abstains on some candidates, so re-plan what is still missing over untried shots.
    for _ in range(3):
        need = target - len(gold) - len(new)
        pool = shots[[k not in tried for k in zip(shots.video_id, shots.hit_id)]]
        plan = plan_quotas(pool, need)
        if need <= 0 or not any(plan.values()):
            break
        for key, k in plan.items():
            quotas[key] = quotas.get(key, 0) + k
        for (era, side, hand), k in plan.items():
            got = 0
            stratum = pool[(pool.era == era) & (pool.side == side) & (pool.hand == hand)]
            for r in _round_robin(stratum, seed).itertuples():
                if got >= k:
                    break
                tried.add((r.video_id, r.hit_id))
                attempts += 1
                res = labeler.label(r)
                reasons[res["reason"]] = reasons.get(res["reason"], 0) + 1
                if res["stroke"] is None:
                    continue
                new.append({"video_id": r.video_id, "hit_id": r.hit_id, "stroke": res["stroke"],
                            "labeler": f"qwen3-vl:{model_id.split('/')[-1]}", "sheet": f"gold2_{era}",
                            "era": era, "side": side, "hand": hand, "confidence": res["confidence"],
                            "votes": res["votes"]})
                got += 1
    new_df = pd.DataFrame(new, columns=GOLD_COLUMNS)
    if len(new_df):
        pd.concat([gold, new_df], ignore_index=True).to_csv(GOLD_PATH, index=False)
    report.update({
        "quotas": {"/".join(k): v for k, v in quotas.items()}, "attempted": attempts, "written": int(len(new_df)),
        "gold_total": int(len(gold) + len(new_df)), "reasons": reasons,
        "written_by_era": new_df.era.value_counts().to_dict(), "written_by_side": new_df.side.value_counts().to_dict(),
        "written_by_hand": new_df.hand.value_counts().to_dict(),
        "review_sheets": contact_sheets(new_df),
    })
    return report
