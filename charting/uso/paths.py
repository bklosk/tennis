"""Filesystem layout shared by every stage."""

from pathlib import Path

PKG = Path(__file__).resolve().parent
PROJECT = PKG.parent  # charting/
REPO = PROJECT.parent  # repo root
DATA = REPO / "data"
MCP_DIR = DATA / "match-charting"
DOWNLOADS = REPO / "downloads"
OUTPUTS = REPO / "outputs"
WEIGHTS = REPO / ".cache" / "weights"
EVAL_MATCHES = PROJECT / "data" / "eval_matches.csv"
BATCH_MATCHES = PROJECT / "data" / "batch_matches.csv"  # archive matches decoded for analytics, not evaluation


def match_dir(video_id: str) -> Path:
    d = OUTPUTS / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d
