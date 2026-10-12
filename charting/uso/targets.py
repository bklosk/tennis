"""The target set: MCP-charted US Open quarterfinals, semifinals and finals, 2001-2025.

:func:`us_open_targets` lists the charted targets with player metadata, point
counts and the best full-match video from the old manifest
(``charting/data/video_candidates.csv``).

Join with the manifest: same year, draw (men/women) and round, and the same
pair of ASCII-folded surnames (last name token, hyphens split, order-free).
For targets that miss, a fallback accepts a manifest row from the same
year/draw/round whose players each share a name token (3+ letters) with a
distinct MCP player (covers name-order differences such as "Na Li"/"Li Na").
Among a match's manifest rows only full-match uploads are eligible:
``verified_official_full_match`` first, then ``candidate_full_match``; ties go to
the longest upload.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from uso.mcp import DEFAULT_DATA_DIR, load_matches, load_points
from uso.scoring import grand_slam_best_of

__all__ = ["DEFAULT_MANIFEST", "TARGET_ROUNDS", "us_open_targets", "surname_key", "fold_name"]

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "video_candidates.csv"
TARGET_ROUNDS = ("QF", "SF", "F")
FULL_MATCH_STATUS_RANK = {"verified_official_full_match": 0, "candidate_full_match": 1}


def fold_name(name: str) -> str:
    """Lowercase ASCII with accents removed and punctuation collapsed to spaces."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]+", " ", s).strip()


def _surname(name: str) -> str:
    toks = fold_name(name).split()
    return toks[-1] if toks else ""


def surname_key(p1: str, p2: str) -> tuple[str, str]:
    """Order-free pair of folded surnames."""
    return tuple(sorted((_surname(p1), _surname(p2))))  # type: ignore[return-value]


def _tokens(name: str) -> set[str]:
    return {t for t in fold_name(name).split() if len(t) >= 3}


def _loose_match(a1: str, a2: str, b1: str, b2: str) -> bool:
    ta1, ta2, tb1, tb2 = _tokens(a1), _tokens(a2), _tokens(b1), _tokens(b2)
    return bool((ta1 & tb1 and ta2 & tb2) or (ta1 & tb2 and ta2 & tb1))


def _load_manifest(path: str | Path) -> pd.DataFrame:
    v = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in v.columns:
        v[c] = v[c].str.strip()
    v["year"] = pd.to_numeric(v["year"], errors="coerce").astype("Int64")
    v["gender"] = v["draw"].str.lower().map({"men": "m", "women": "w"})
    v["round"] = v["round"].str.upper()
    v["duration_seconds"] = pd.to_numeric(v["duration_seconds"], errors="coerce").astype("Int64")
    v["key"] = [surname_key(a, b) for a, b in zip(v["player1"], v["player2"])]
    return v


def us_open_targets(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    manifest_path: str | Path | None = DEFAULT_MANIFEST,
    matches_df: pd.DataFrame | None = None,
    points_df: pd.DataFrame | None = None,
    years: tuple[int, int] = (2001, 2025),
) -> pd.DataFrame:
    """MCP-charted US Open QF/SF/F singles matches in ``years`` (inclusive).

    Columns: match_id, year, date, gender, round, player1, player2, hand1, hand2,
    best_of (Grand Slam rule: men 5, women 3), best_of_file (MCP match file),
    n_points, and from the manifest: video_id, video_status, duration_seconds,
    video_title, video_channel, watch_url, n_full_match_videos,
    manifest_match_id, manifest_join ('surname', 'token' or None).
    """
    m = matches_df if matches_df is not None else load_matches(data_dir)
    t = m[
        (m["Tournament"] == "US Open")
        & m["Round"].isin(TARGET_ROUNDS)
        & m["year"].between(years[0], years[1])
    ].copy()
    p = points_df if points_df is not None else load_points(data_dir, match_ids=t["match_id"])
    n_points = p.groupby("match_id").size()
    out = pd.DataFrame({
        "match_id": t["match_id"],
        "year": t["year"].astype(int),
        "date": t["date"],
        "gender": t["gender"],
        "round": t["Round"],
        "player1": t["Player 1"],
        "player2": t["Player 2"],
        "hand1": t["Pl 1 hand"].where(t["Pl 1 hand"].isin(["R", "L"]), "U"),
        "hand2": t["Pl 2 hand"].where(t["Pl 2 hand"].isin(["R", "L"]), "U"),
        "best_of": [grand_slam_best_of(g, r) for g, r in zip(t["gender"], t["Round"])],
        "best_of_file": t["best_of"],
    })
    out["n_points"] = out["match_id"].map(n_points).fillna(0).astype(int)
    out["round_order"] = out["round"].map({r: i for i, r in enumerate(TARGET_ROUNDS)})
    out = out.sort_values(["year", "gender", "round_order", "match_id"]).drop(columns="round_order")
    out = out.reset_index(drop=True)

    video_cols = ["video_id", "video_status", "duration_seconds", "video_title", "video_channel",
                  "watch_url", "n_full_match_videos", "manifest_match_id", "manifest_join"]
    if manifest_path is None or not Path(manifest_path).exists():
        for c in video_cols:
            out[c] = None
        return out

    v = _load_manifest(manifest_path)
    v = v[v["tournament"] == "US Open"]
    picks = []
    for r in out.itertuples(index=False):
        same = v[(v["year"] == r.year) & (v["gender"] == r.gender) & (v["round"] == r.round)]
        key = surname_key(r.player1, r.player2)
        rows = same[same["key"] == key]
        how = "surname" if len(rows) else None
        if not len(rows):
            loose = pd.Series(
                [_loose_match(r.player1, r.player2, a, b) for a, b in zip(same["player1"], same["player2"])],
                index=same.index, dtype=bool,
            )
            rows = same[loose]
            how = "token" if len(rows) else None
        full = rows[rows["match_status"].isin(list(FULL_MATCH_STATUS_RANK))].copy()
        if len(full):
            full["rank"] = full["match_status"].map(FULL_MATCH_STATUS_RANK)
            full = full.sort_values(["rank", "duration_seconds"], ascending=[True, False], na_position="last")
            b = full.iloc[0]
            picks.append((b["video_id"], b["match_status"], b["duration_seconds"], b["title"], b["channel"],
                          b["watch_url"], len(full), b["match_id"], how))
        else:
            mid = rows["match_id"].iloc[0] if len(rows) else None
            picks.append((None, None, pd.NA, None, None, None, 0, mid, how))
    vid = pd.DataFrame(picks, columns=video_cols)
    vid["duration_seconds"] = vid["duration_seconds"].astype("Int64")
    return pd.concat([out, vid], axis=1)
