"""Post-vision steps for processed matches: candidates, API onset labels, decoding + alignment,
stroke pose features and stroke API questions.

    uv run python scripts/post.py VIDEO_ID [...] [--steps candidates,onsets,shots,strokefeats,strokeapi]
                                  [--labeler luna|jev]

With --labeler jev the onsets are labelled by Jev into onset_jev.parquet and decoded with Jev's
decoder parameters and contact models; the strokeapi step is skipped (the pose-only stroke model
needs no API answers).
"""

from __future__ import annotations

import argparse
import json
import time

from uso.paths import OUTPUTS

STEPS = ["candidates", "onsets", "shots", "strokefeats", "strokeapi"]


def params(labeler: str = "luna"):
    from uso import labeler as labeler_mod

    p = labeler_mod.params_path(labeler)
    if not p.exists() and labeler != "luna":  # luna falls back to the decoder defaults, as it always did
        raise FileNotFoundError(f"{p} is missing: run scripts/tune_decoder.py --labeler {labeler} first")
    return json.loads(p.read_text()) if p.exists() else None


def run(vid: str, steps: list[str], labeler: str = "luna"):
    from uso import labeler as labeler_mod
    from uso import onset_api, rally, shots, strokes

    cl = labeler_mod.client(labeler)
    for st in steps:
        t0 = time.time()
        if st == "candidates":
            info = dict(rows=len(rally.features(vid)))
        elif st == "onsets":  # strong onsets, then soft onsets inside rally context
            onset_api.run(vid, z_min=8.0, client=cl, labeler=labeler)
            info = dict(rows=len(onset_api.run_soft(vid, client=cl, labeler=labeler)))
        elif st == "shots":
            r = shots.build(vid, params(labeler), labeler)
            info = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["info"].items()}
        elif st == "strokefeats":
            info = dict(rows=len(strokes.run_local_features(vid, force=True)))
        elif st == "strokeapi" and not labeler_mod.uses_api_strokes(labeler):
            info = dict(skipped="pose-only stroke model")
        elif st == "strokeapi":
            a = strokes.run_api(vid, client=cl)
            b = strokes.run_api_imgside(vid, client=cl)
            info = dict(api=len(a), imgside=len(b))
        else:
            raise ValueError(st)
        info.update(step=st, video=vid, labeler=labeler, seconds=round(time.time() - t0, 1), spent_usd=round(cl.spent(), 3))
        print(json.dumps(info), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--labeler", default="luna", choices=["luna", "jev"])
    a = ap.parse_args()
    for v in a.videos:
        run(v, a.steps.split(","), a.labeler)


if __name__ == "__main__":
    main()
