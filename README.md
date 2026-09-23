# US Open CV

This repo fine tunes TrackNet to chart/code/label US open matches between 2000-2026 by generating labels with VLMs!

The general idea is that classifiers are cheap but vlms are expensive, so I generated 2,000 or so labeled frames using expensive models and evaluated tracknet against that.

## Charting pipeline (pilot)

`pipeline/` turns a full-match broadcast into analysis tables: ball trajectory, player court
positions, every shot (player, forehand/backhand/serve/overhead, volley, contact and bounce
coordinates, speed), and points aligned to the official point-by-point data (serve speed,
serve placement, winner). It runs locally on Apple Silicon (TrackNet on the Neural Engine,
~32 fps tracking on an M3 Pro); the same code runs on CUDA.

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
uv run python -m tennis_pipeline.cli balllabels VIDEO_ID ... --action auto            # trajectory-teacher labels, no review
uv run python -m tennis_pipeline.cli balltrain --epochs 30                             # fine-tune TrackNet
uv run python -m eval.serve_fit VIDEO_ID ...                                           # refit serve weights vs official
uv run python -m eval.serve_eval VIDEO_ID ... --sweep 0.5,0.7                          # serve detector vs official
uv run python -m tennis_pipeline.cli gold VIDEO_ID ... --target 200                    # Qwen3-VL gold expansion
uv run pytest                                                                          # synthetic-data tests
```

Pilot results on three 2024 matches (accuracy, cost projection, next steps) are in
[`docs/charting-pilot.md`](docs/charting-pilot.md); the full-match rerun on the M3 Pro (Neural
Engine tracking, retuned serve detection) is in [`docs/m3-pilot.md`](docs/m3-pilot.md). GPU
throughput on a DigitalOcean L40S, the
revised full-dataset cost, and the budget-capped droplet launcher are in
[`docs/gpu-pilot.md`](docs/gpu-pilot.md):

```bash
uv run python -m tennis_pipeline.cloud pilot VIDEO_ID ... --budget 4   # full pipeline on an L40S droplet
uv run python -m tennis_pipeline.cloud bench VIDEO.mp4 --budget 0.6    # throughput benchmark
```

## Full-dataset batch run

All 198 verified US Open matches run on one budget-capped L40S droplet, streamed from the Mac that
has the videos (about $52 estimated). Scenes, audio and main-camera packs are prepared on the Mac.
On the droplet, one GPU worker tracks match after match while CPU workers finish the previous ones.
The run is resumable. See [`docs/batch-run.md`](docs/batch-run.md):

```bash
uv run python -m tennis_pipeline.batch prep --manifest usopen --pack        # Mac: prep + plan (upload, hours, cost)
uv run python -m tennis_pipeline.cloud batch --manifest usopen --budget 70  # run or resume on a droplet
uv run python -m tennis_pipeline.batch status --manifest usopen             # progress
```

Videos are read from `downloads/VIDEO_ID.mp4`; outputs go to `outputs/VIDEO_ID/`. Pretrained
weights (TrackNet ball, court keypoints, CatBoost bounce) come from the yastrebksv
TennisProject repositories and live in `.cache/weights/`. Labels are bootstrapped with
Qwen3-VL-8B running locally through MLX.

## Next steps / TODO

- [x] Run the three pilot matches to the end locally and tune the serve detector against official
      data ([`docs/m3-pilot.md`](docs/m3-pilot.md)).
- [ ] Run the 3-match pilot on an L40S from the Mac, where the videos already are (YouTube
      blocks datacenter IPs), to confirm throughput on real broadcasts now that the court model
      is FP32 again:
      `uv run python -m tennis_pipeline.cloud pilot KCcKkUnjbzA Fl33UXv6jKI Ce3dRYHWIBI --budget 4`.
      Keep `DIGITALOCEAN_ACCESS_TOKEN` in the environment or a secret store, never in the repo.
- [ ] Label about 2,000 ball frames (`balllabels`) and fine-tune TrackNet (`balltrain`). This is
      the biggest lever for rally length, serve detection and bounces.
- [ ] Improve rally-count accuracy (64% within ±1 shot on the full pilot matches). Rallies of
      9+ shots are undercounted by ~4.6: near-side returns after high bounces show no image-y
      reversal, so the hit rule misses them.
- [ ] Handle faults: only 18 of 168 first-serve faults are seen as separate attempts, and the
      returner's knock-aways after an out serve count as rally shots.
- [ ] Re-fit the serve weights (`eval/serve_fit.py`) once matches from other eras are processed.
- [ ] Validate `--scene-keyframes` against full-decode scene segments before using it for the
      full run. It is about 4× faster.
- [x] Prepare the full run for a GPU droplet. Main-camera packs and streaming uploads handle the
      video transfer, and OCR runs automatically for matches without official data
      ([`docs/batch-run.md`](docs/batch-run.md)).
- [ ] Before the full run, run `batch prep --manifest usopen --pack` and check the matches it
      flags. The scene classifier and court model have only been checked on 2024 broadcasts, and
      pre-2007 matches are likely 4:3 SD video.
- [ ] Full 198-match batch run (`cloud batch --manifest usopen --budget 70`, about $52 estimated).
      Watch the first matches' tracking fps and GPU utilisation in the progress line.
- [ ] Expand the stroke gold set (`cli gold`, Mac only) beyond 51 labels and re-check the
      geometric stroke rule.
- [ ] Reduce the remaining CPU-bound work. The GPU is busy only about 64% of the time on an
      L40S; YOLO preprocessing on the GPU is the next candidate (or `--gpu-workers 2` on a
      droplet with more vCPUs).

Public availability does not grant permission to download, train on, or redistribute a
copyrighted broadcast. Do not bypass DRM, authentication, geographic restrictions, or
other access controls.
