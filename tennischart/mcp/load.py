"""Load Match Charting Project CSVs (latin-1, stray whitespace stripped)."""

from __future__ import annotations

import glob
import os
from functools import lru_cache

import pandas as pd

from ..config import DATA_DIR

MCP_DIR = os.path.join(DATA_DIR, "match-charting")


def _read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    for c in df.columns:
        df[c] = df[c].str.strip()
    df.columns = [c.strip() for c in df.columns]
    return df


@lru_cache(maxsize=4)
def load_matches(mcp_dir: str = MCP_DIR) -> pd.DataFrame:
    frames = []
    for gender in ("m", "w"):
        path = os.path.join(mcp_dir, f"charting-{gender}-matches.csv")
        if os.path.exists(path):
            d = _read(path)
            d["gender"] = gender.upper()
            frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    m["year"] = pd.to_numeric(m["Date"].str[:4], errors="coerce").astype("Int64")
    m["best_of"] = pd.to_numeric(m["Best of"], errors="coerce").astype("Int64")
    return m


@lru_cache(maxsize=4)
def load_points(mcp_dir: str = MCP_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(mcp_dir, "charting-*-points-*.csv"))):
        d = _read(path)
        d["gender"] = "M" if "-m-" in os.path.basename(path) else "W"
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["Pt"] = pd.to_numeric(p["Pt"], errors="coerce").astype("Int64")
    p = p.sort_values(["match_id", "Pt"], kind="stable").reset_index(drop=True)
    return p


def points_for(match_id: str, mcp_dir: str = MCP_DIR) -> pd.DataFrame:
    p = load_points(mcp_dir)
    return p[p["match_id"] == match_id].reset_index(drop=True)
