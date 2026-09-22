from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DOWNLOADS = ROOT / "downloads"
CACHE = ROOT / ".cache"
WEIGHTS = CACHE / "weights"
SACKMANN = CACHE / "sackmann"
OUTPUTS = ROOT / "outputs"
SACKMANN_BASE = "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main"


def match_dir(video_id: str) -> Path:
    path = OUTPUTS / video_id
    path.mkdir(parents=True, exist_ok=True)
    return path
