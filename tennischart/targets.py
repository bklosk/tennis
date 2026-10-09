"""Target match list (data/matches.csv) joined to Match Charting Project matches."""

from __future__ import annotations

import os
import re
import unicodedata

import pandas as pd

from .config import DATA_DIR
from .mcp.load import load_matches as load_mcp_matches
from .scoring import MatchFormat, format_for

RETIREMENT = re.compile(r"\b(?:RET|DEF|W/O|ABN|ABD)\b", re.I)


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", " ", s.lower().replace("-", " ")).strip()


def surname_tokens(name: str) -> set[str]:
    toks = fold(name).split()
    return set(toks[1:]) if len(toks) > 1 else set(toks)


def _pair_score(a1: str, a2: str, b1: str, b2: str) -> int:
    """Name-token overlap when a1 plays b1 and a2 plays b2; 0 unless both surnames agree."""
    if not (surname_tokens(a1) & set(fold(b1).split())) or not (surname_tokens(a2) & set(fold(b2).split())):
        return 0
    return len(set(fold(a1).split()) & set(fold(b1).split())) + len(set(fold(a2).split()) & set(fold(b2).split()))


def best_of_for(gender: str, round_: str = "") -> int:
    """Grand Slam singles: men's main draw best of five, everything else best of three."""
    return 5 if gender.upper().startswith("M") and round_ not in ("Q1", "Q2", "Q3") else 3


def load_targets(path: str | None = None) -> pd.DataFrame:
    path = path or os.path.join(DATA_DIR, "matches.csv")
    t = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in t.columns:
        t[c] = t[c].str.strip()
    t["year"] = t["year"].astype(int)
    t["gender"] = t["draw"].str[0].str.upper()
    t["played"] = t["match_played"].str.lower().eq("true")
    t["retired"] = t["result"].str.contains(RETIREMENT, na=False) & t["played"]
    return t


def match_format(row: pd.Series) -> MatchFormat:
    return format_for(row["tournament"], int(row["year"]), row["gender"],
                      best_of_for(row["gender"], row.get("round", "")))


def join_mcp(targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add `mcp_match_id` (or '') to each target by tournament/year/gender/round + surnames."""
    t = load_targets() if targets is None else targets.copy()
    m = load_mcp_matches().copy()
    m["tour"] = m["Tournament"].str.lower()
    out = []
    for _, row in t.iterrows():
        cand = m[(m["tour"] == row["tournament"].lower()) & (m["year"] == row["year"])
                 & (m["gender"] == row["gender"]) & (m["Round"] == row["round"])]
        best, best_score = "", 0
        for _, c in cand.iterrows():
            score = max(_pair_score(row["player1"], row["player2"], c["Player 1"], c["Player 2"]),
                        _pair_score(row["player1"], row["player2"], c["Player 2"], c["Player 1"]))
            if score > best_score:
                best, best_score = c["match_id"], score
        out.append(best if best_score > 0 else "")
    t["mcp_match_id"] = out
    return t


def coverage(tournament: str = "US Open", first: int = 2001, last: int = 2025) -> pd.DataFrame:
    t = join_mcp()
    t = t[(t["tournament"] == tournament) & t["year"].between(first, last)]
    t = t.assign(charted=t["mcp_match_id"] != "")
    return t.groupby("gender").agg(targets=("match_id", "size"), played=("played", "sum"),
                                   charted=("charted", "sum"))
