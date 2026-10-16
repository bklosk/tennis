"""Which model labels the audio onsets, and the artifacts that belong to that choice.

The decoder thresholds and the learned contact model are fitted to one labeler's answers, so each
labeler keeps its own label file, decoder parameters and contact models:

  * `luna`: OpenAI Decisions API (gpt-6-luna). It produced the published evaluation (reports/), and
    every function defaults to it so that evaluation reproduces unchanged.
  * `jev`: TypeSafe Jev 1.14 on the same contact sheets and questions (reports/jev/). The batch
    runner and `uso.uncharted` default to it. Matches charted with Jev use the pose-only stroke model,
    because the API stroke answers add nothing to it (see DECISIONS.md).
"""

from __future__ import annotations

import json
from pathlib import Path

from uso.paths import OUTPUTS, match_dir

NAMES = ("luna", "jev")
BATCH_DEFAULT = "jev"
_SUFFIX = {"luna": "", "jev": "_jev"}


def _check(labeler: str) -> str:
    if labeler not in NAMES:
        raise ValueError(f"unknown onset labeler {labeler!r}; expected one of {NAMES}")
    return labeler


def labels_path(video_id: str, labeler: str = "luna") -> Path:
    return match_dir(video_id) / ("onset_api.parquet" if _check(labeler) == "luna" else "onset_jev.parquet")


def params_path(labeler: str = "luna") -> Path:
    return OUTPUTS / f"decoder_params{_SUFFIX[_check(labeler)]}.json"


def params(labeler: str = "luna") -> dict:
    from uso import rally

    return {**rally.DECODER, **json.loads(params_path(labeler).read_text())}


def hitmodel_path(labeler: str = "luna", loo_video: str | None = None) -> Path:
    s = _SUFFIX[_check(labeler)]
    return OUTPUTS / (f"hitmodel{s}_loo_{loo_video}.pkl" if loo_video else f"hitmodel{s}.pkl")


def stroke_model_name(labeler: str = "luna") -> str:
    return "stroke_model.pkl" if _check(labeler) == "luna" else "stroke_model_pose.pkl"


def uses_api_strokes(labeler: str = "luna") -> bool:
    return _check(labeler) == "luna"


def client(labeler: str = "luna"):
    if _check(labeler) == "jev":
        from uso.jev import JevClient

        return JevClient()
    from uso.decisions import DecisionClient

    return DecisionClient()


# the `source` column of every exported row: which pipeline version made it
SOURCE = {"luna": "model:uso-0.1", "jev": "model:uso-0.2-jev"}

# tokens per onset request, for the pre-batch estimate (measured: luna 669, Jev 1,163)
TOKENS_PER_ONSET = {"luna": 700, "jev": 1200}
