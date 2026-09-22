# US Open CV

This repo fine tunes TrackNet to chart/code/label US open matches between 2000-2026 by generating labels with VLMs!

The general idea is that classifiers are cheap but vlms are expensive, so I generated 2,000 or so labeled frames using expensive models and evaluated tracknet against that.

## Charting pipeline (pilot)

`pipeline/` turns a full-match broadcast into analysis tables: ball trajectory, player court
positions, every shot (player, forehand/backhand/serve/overhead, volley, contact and bounce
coordinates, speed), and points aligned to the official point-by-point data (serve speed,
serve placement, winner). It runs locally on Apple Silicon; the same code runs on CUDA.

```bash
cd pipeline
uv run python -m tennis_pipeline.cli scenes  VIDEO_ID ...   # main-camera classifier (VLM-labelled)
uv run python -m tennis_pipeline.cli track   VIDEO_ID       # court, ball (TrackNet), players; cached per chunk
uv run python -m tennis_pipeline.cli events  VIDEO_ID       # bounces, hits, serves (+ audio snapping)
uv run python -m tennis_pipeline.cli crops   VIDEO_ID       # hitter crops + pose features
uv run python -m tennis_pipeline.cli align   VIDEO_ID       # group into points, align to official data
uv run python -m tennis_pipeline.cli strokes VIDEO_ID ...   # VLM-labelled stroke classifier
uv run python -m tennis_pipeline.cli report  VIDEO_ID ...   # CSV exports, court maps, QA clip
uv run python -m tennis_pipeline.cli ocr     VIDEO_ID       # score bug + serve speed, for matches without official data
```

Improvement tooling from the pilot's next steps (see the pilot doc for the workflow):

```bash
uv run python -m tennis_pipeline.cli balllabels VIDEO_ID ... --action sample --n 2000   # pick + pre-label frames
uv run python -m tennis_pipeline.cli balllabels --action export                        # decode 3-frame inputs
uv run python -m tennis_pipeline.cli balllabels --action review --labeler NAME         # click-through labeling
uv run python -m tennis_pipeline.cli balltrain --epochs 30                             # fine-tune TrackNet
uv run python -m eval.serve_eval VIDEO_ID ... --sweep 0.35,0.5,0.65                    # serve detector vs official
uv run python -m tennis_pipeline.cli gold VIDEO_ID ... --target 200                    # Qwen3-VL gold expansion
uv run pytest                                                                          # synthetic-data tests
```

Pilot results on three 2024 matches (accuracy, cost projection, next steps) are in
[`docs/charting-pilot.md`](docs/charting-pilot.md).

Videos are read from `downloads/VIDEO_ID.mp4`; outputs go to `outputs/VIDEO_ID/`. Pretrained
weights (TrackNet ball, court keypoints, CatBoost bounce) come from the yastrebksv
TennisProject repositories and live in `.cache/weights/`. Labels are bootstrapped with
Qwen3-VL-8B running locally through MLX.

Public availability does not grant permission to download, train on, or redistribute a
copyrighted broadcast. Do not bypass DRM, authentication, geographic restrictions, or
other access controls.
