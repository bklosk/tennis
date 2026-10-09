"""Priors estimated from Match Charting Project points.

Used as weak evidence by the contact decoder (rally length, serve faults) and by stroke fusion
(stroke family and side by shot position, return side by serve geometry and handedness).
`python -m tennischart data priors` regenerates resources/priors.json.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from functools import lru_cache

from .config import RESOURCES_DIR
from .mcp.load import load_matches, load_points
from .mcp.notation import parse_point
from .scoring import format_for, new_match
from .targets import best_of_for

PRIORS_PATH = os.path.join(RESOURCES_DIR, "priors.json")


def bucket_of(index: int) -> str:
    return {0: "serve", 1: "return", 2: "serve_plus_one"}.get(index, "rally")


def _norm(c: Counter) -> dict:
    tot = sum(c.values())
    return {k: round(v / tot, 5) for k, v in sorted(c.items())} if tot else {}


def compute_priors(min_year: int = 2001) -> dict:
    pts = load_points()
    matches = load_matches().set_index("match_id")
    rally = defaultdict(Counter)
    fam = defaultdict(lambda: defaultdict(Counter))
    side = defaultdict(lambda: defaultdict(Counter))
    ret = defaultdict(Counter)
    faults = defaultdict(Counter)
    for mid, g in pts.groupby("match_id", sort=False):
        if mid not in matches.index:
            continue
        mr = matches.loc[mid]
        year = int(mr["year"]) if mr["year"] is not None else 0
        if year < min_year:
            continue
        gender = mr["gender"]
        hands = {1: mr["Pl 1 hand"] or "R", 2: mr["Pl 2 hand"] or "R"}
        fmt = format_for(mr["Tournament"], year, gender, best_of_for(gender, mr["Round"]))
        try:
            st = new_match(fmt, int(g["Svr"].iloc[0]))
        except ValueError:
            continue
        for first, second, svr_s, won in zip(g["1st"], g["2nd"], g["Svr"], g["PtWinner"]):
            p = parse_point(first, second)
            svr = int(svr_s) if svr_s in ("1", "2") else st.server
            court_side = st.side
            if p.ok and p.final.code is None and p.attempts[0].shots:
                faults[gender]["first_fault" if p.attempts[0].is_fault else "first_in"] += 1
                rally[gender][min(p.rally_length, 30)] += 1
                for s in p.shots[1:]:
                    b = bucket_of(s.index)
                    fam[gender][b][s.family] += 1
                    if s.side in ("forehand", "backhand"):
                        side[gender][b][s.side] += 1
                serve = p.shots[0]
                if len(p.shots) > 1 and serve.direction in (4, 5, 6) and p.shots[1].side in ("forehand", "backhand"):
                    rhand = hands[3 - svr]
                    ret[f"{rhand}|{court_side}|{serve.direction}"][p.shots[1].side] += 1
            try:
                st = st.point(int(won))
            except ValueError:
                break
            if st.done:
                break
    return {
        "source": "Match Charting Project (CC BY-NC-SA 4.0), Australian Open + US Open, "
                  f"{min_year}+, computed by tennischart.priors",
        "rally_length": {g: _norm(c) for g, c in rally.items()},
        "first_serve_fault_rate": {g: round(c["first_fault"] / max(1, sum(c.values())), 4)
                                   for g, c in faults.items()},
        "family_by_bucket": {g: {b: _norm(c) for b, c in d.items()} for g, d in fam.items()},
        "side_by_bucket": {g: {b: _norm(c) for b, c in d.items()} for g, d in side.items()},
        "return_side": {k: {"forehand": round((c["forehand"] + 1) / (sum(c.values()) + 2), 4),
                            "n": sum(c.values())} for k, c in sorted(ret.items())},
    }


def write_priors(path: str = PRIORS_PATH, min_year: int = 2001) -> dict:
    pri = compute_priors(min_year)
    with open(path, "w") as f:
        json.dump(pri, f, indent=1)
    return pri


@lru_cache(maxsize=1)
def load_priors(path: str = PRIORS_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
