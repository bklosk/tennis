# US Open charting from broadcast video

`charting/` turns a full-match broadcast into a shot table: for every racket contact it gives the
time in the video, the point and shot number, who hit it, **both players' court coordinates at the
contact**, and (for rally shots) the **stroke side and family**; for every point it gives the
**rally length**. Everything is evaluated against human charting from the Match Charting Project
(MCP). Results are in [`reports/evaluation.md`](reports/evaluation.md).

All outputs are machine-generated: match ids carry a `-machine` suffix, every row has
`source=model:uso-0.1`, and MCP values appear only in `mcp_*` columns kept for evaluation.

## Results (10 held-out MCP-charted matches, 2001–2025, 1,453 charted points)

| What | Result |
|---|---|
| Points found and aligned to MCP | 89% |
| Rally length | 62% exact, 87% within one contact (dev: 66% / 87%) |
| Forehand/backhand | 91% (near player 93%, far 90%; calibration error 0.04); 87% on 2001–10 SD footage, 95% from 2016 |
| Stroke family | 85% accuracy but macro-F1 0.37: groundstrokes 97%, volleys 84%, slices 42%, drops/lobs/overheads rarely found |
| Player positions | No gold set; against charted facts the positions are right: server behind the baseline 99.9%, returner position separates wide from T serves (AUC 0.93), receiver position separates shot direction 1 vs 3 (AUC 0.98), net shots vs others (AUC 0.97) |
| Decisions API spend | $7.69 of the $15 budget (77M input tokens), including ~$0.5 of experiments |

The guide's targets (85% exact rally length, 97%/93% stroke side near/far, 0.75 family macro-F1)
are not met; see `reports/evaluation.md` for per-match, per-era and per-gender tables and
`DECISIONS.md` for what was tried.

## Quick start

```bash
cd charting
uv sync
uv run pytest                                   # parser, scoring state machine, decoder tests

uv run python scripts/fetch_videos.py           # download + verify the 14 evaluation videos (user-approved)
uv run python scripts/process.py --wait         # vision stages for every verified video (resumable)
uv run python scripts/post.py VIDEO_ID ...      # candidates, API onset labels, decoding, stroke features
uv run python scripts/tune_decoder.py           # (dev only) rally-decoder parameters
uv run python scripts/evaluate_all.py           # stroke model, metrics, reports/, outputs/*.csv
```

To **spot-check** a match, render a short overlay clip (court lines, both players with court
coordinates, a flash and stroke label at each detected contact next to MCP's letter, rally count
vs MCP, top-down map, original audio):

```bash
uv run python scripts/overlay_clip.py VIDEO_ID [--point MCP_PT | --start SECONDS] [--seconds 10]
```

Clips go to `outputs/clips/` (gitignored: they are broadcast footage and are not redistributed).

To chart a match that has **no MCP record**, put the video at `downloads/VIDEO_ID.mp4` and run

```bash
uv run python -m uso.uncharted VIDEO_ID [--hand-near R --hand-far R]
```

It runs the same stages and the same frozen decoder and stroke model (no MCP input) and writes
`outputs/VIDEO_ID/uncharted_points.csv` and `uncharted_shots.csv`. Hitters are given by court end
(near = camera end); naming them needs who served first, which the video alone does not give.
Expect about $0.4–0.6 of Decisions API calls and ~25 minutes of compute per hour of video on an
M3 Pro.

The OpenAI key is read from `OPENAI_API_KEY` or the repo-root `.env` (gitignored). Every paid
request goes through `uso/decisions.py`, which caches responses by input hash, records spend in
`outputs/decisions/ledger.jsonl`, checks an estimate before each batch, and refuses to send a
request that could take total spend past **$14** (the user's limit is $15).

## Pipeline

| Stage | Module | What it does |
|---|---|---|
| Truth | `uso/mcp.py`, `uso/scoring.py`, `uso/targets.py` | MCP grammar (99.93% of 521k strings parse, round-trip tested), scoring state machine (server, side, end changes, 2022+ final-set tiebreak), US Open QF+ targets |
| Scene | `uso/scene.py`, `uso/court.py` | 1 fps court registration with a 14-keypoint court network + RANSAC homography + line-overlap check; a sample is "live view" when the court is seen from behind the near baseline (high camera, low camera and offset low cameras all pass) |
| People | `uso/people.py` | RF-DETR Small (fp16, TorchScript) at 4 fps in two views: the full frame for the near player and an upscaled crop of the far half (located through the homography) for the small far player |
| Tracks | `uso/players.py` | Viterbi per court half picks the player among detections (keeps ball kids, umpire, line judges out); ground point → meters |
| Pose | `uso/pose.py` | YOLO11m-pose on upscaled crops of the two tracked players |
| Audio | `uso/audio.py` | High-band spectral-flux onsets with a running median/MAD threshold: candidate contact times |
| Onset labels | `uso/onset_api.py` | Decisions API (`gpt-6-luna`) sees both players at −0.1/0/+0.1 s around each onset and answers serve / hit / ball bounce / none per player |
| Rally decoding | `uso/rally.py`, `uso/points.py` | Serves (API serve label, server behind the baseline, a quiet spell before); then the best alternating sequence of contacts under an inter-contact interval prior; faults and double faults from repeated serves |
| Alignment | `uso/align.py`, `uso/truth.py` | Needleman–Wunsch between video points and MCP points on server end, serve side, serve attempts and rally length (both starting layouts tried) |
| Shots | `uso/shots.py` | Contacts with both players' positions, named hitter via the alignment, paired MCP shot for evaluation |
| Strokes | `uso/strokes.py`, `uso/stroke_model.py` | Pose features from six crops around each contact (mirrored by court end and handedness, plus an orientation-free shoulder-axis projection), fused with API answers in gradient-boosted classifiers for forehand/backhand and family |
| Report | `uso/report.py`, `scripts/evaluate_all.py` | Metrics, position checks, exports |

### Coordinates

Meters, origin on the ground at the centre of the net. `x` runs across the court, positive to the
right as seen from the camera end; `y` runs along it, positive toward the far end. Baselines are
at `y = ±11.885`, singles sidelines at `x = ±4.115`. `hitter_end` says which end (near = camera
end) the hitter is on. Positions are the player's ground point (ankles when visible, else the
bottom of the box), sampled at 4 fps and interpolated to the contact time.

## Outputs (gitignored, under `outputs/`)

* `VIDEO_ID/shots.csv`: one row per serve attempt and rally contact: `t_video_s`, `video_point`,
  `mcp_point` (when aligned), `shot_no` (serve = 1; fault serves 0), `kind`, `hitter_end`,
  `hitter`, `hitter_hand`, `hitter_x_m`, `hitter_y_m`, `opponent_x_m`, `opponent_y_m`,
  `stroke_side` + `stroke_p_forehand`, `stroke_family` + per-family probabilities,
  `interpolated` (a contact the decoder inferred between two detected ones), and `mcp_*` truth.
* `VIDEO_ID/points.csv`: one row per detected point: start/end time, server end, serve side,
  serve attempts, `rally_length_pred` (MCP convention: the in-play serve counts, double fault = 0),
  and the aligned MCP point and rally length.
* `all_shots.csv`, `all_points.csv`: every evaluation match together.

## Video archive (DigitalOcean Space)

```bash
uv run python scripts/archive_videos.py --parallel 3     # resumable; --dry-run shows what is left
```

Every US Open QF/SF/F from 2001–2025 that has a full-match upload in the video manifest (164 of
the 350 targets, listed in `data/archive_targets.csv`) is downloaded, decode-verified, uploaded as
a **private** object and deleted locally (the 14 evaluation videos stay in `downloads/`). Layout:

* `s3://benklosky-data/tennis/usopen-video/YEAR/MATCH_ID__VIDEO_ID.mp4` (metadata: video id,
  match id, sha256, source URL)
* `s3://benklosky-data/tennis/usopen-video/manifest.csv`: what was archived, size, duration, sha256
* `s3://benklosky-data/tennis/usopen-video/missing.csv`: targets with no archived full match and
  why (highlights only, no video found, walkover)

Credentials come from `~/.s3cfg`. Matches before 2011 are SD sources and are stored at 720p;
later matches at 1080p (~1.9 GB per hour of video).

## Data, models and licences

* MCP charting and the slam point-by-point data are CC BY-NC-SA 4.0 (attribution to Jeff
  Sackmann / Tennis Abstract; non-commercial; share-alike). Derived tables carry the same terms.
* RF-DETR Small (Apache 2.0). YOLO11 pose weights (Ultralytics, AGPL-3.0). The court keypoint
  network is the pretrained model from yastrebksv/TennisCourtDetector, whose repository states no
  licence; it is used here non-commercially.
* Broadcast videos were downloaded with the user's approval for this evaluation (2026-10-08) and
  are not redistributed; frames leave the machine only as Decisions API inputs.
