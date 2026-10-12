"""Post-vision steps for processed matches: candidates, API onset labels, decoding + alignment,
stroke pose features and stroke API questions.

    uv run python scripts/post.py VIDEO_ID [...] [--steps candidates,onsets,shots,strokefeats,strokeapi]
"""

from __future__ import annotations

import argparse
import json
import time

from uso.paths import OUTPUTS

STEPS = ["candidates", "onsets", "shots", "strokefeats", "strokeapi"]


def params():
    p = OUTPUTS / "decoder_params.json"
    return json.loads(p.read_text()) if p.exists() else None


def run(vid: str, steps: list[str]):
    from uso import onset_api, rally, shots, strokes
    from uso.decisions import DecisionClient

    cl = DecisionClient()
    for st in steps:
        t0 = time.time()
        if st == "candidates":
            info = dict(rows=len(rally.features(vid)))
        elif st == "onsets":  # strong onsets, then soft onsets inside rally context
            onset_api.run(vid, z_min=8.0, client=cl)
            info = dict(rows=len(onset_api.run_soft(vid, client=cl)))
        elif st == "shots":
            r = shots.build(vid, params())
            info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["info"].items()}
        elif st == "strokefeats":
            info = dict(rows=len(strokes.run_local_features(vid, force=True)))
        elif st == "strokeapi":
            a = strokes.run_api(vid, client=cl)
            b = strokes.run_api_imgside(vid, client=cl)
            info = dict(api=len(a), imgside=len(b))
        else:
            raise ValueError(st)
        info.update(step=st, video=vid, seconds=round(time.time() - t0, 1), spent_usd=round(cl.spent(), 3))
        print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--steps", default=",".join(STEPS))
    a = ap.parse_args()
    for v in a.videos:
        run(v, a.steps.split(","))


if __name__ == "__main__":
    main()
