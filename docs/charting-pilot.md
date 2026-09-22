# US Open charting pilot: 3 matches, 2024

Local run on an M3 Pro. Outputs are written to `outputs/VIDEO_ID/`. Accuracy is measured
against the official point-by-point record and 51 hand-checked stroke labels.

## What the pilot shows

The pipeline produces shot-level charts with named players, court coordinates,
forehand/backhand and official serve speed. Point alignment and stroke type are good enough for
analytics now. Rally length and serve detection are the weak spots; both trace back to
ball-tracking gaps, which fine-tuning TrackNet on US Open footage should address.

| Headline | Value |
|---|---|
| Forehand/backhand accuracy (51 gold labels) | 90% |
| Official points matched (full match) | 94% |
| Server end consistent when a serve is detected | 100% |
| Rally length within ±1 shot | 64–68% |
| Projected cost, all 178 matches (L40S) | $28–60 |

## Per-match results

| Match | Video (min) | Main-camera min processed | Points matched | Serve detected | Rally ±1 | Ball found in play | Shots with bounce x/y | Shots |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Navarro vs Badosa · QF | 59.9 | 27.3 of 29.1 (full) | 105 / 112 | 85% | 64% | 74% | 70% | 383 |
| Draper vs de Minaur · SF | 104.1 | 11.0 of 49.8 (first 55 segments) | 38 | 95% | 66% | 75% | 82% | 158 |
| Sinner vs Fritz · F | 127.6 | 15.1 of 60.3 (first 51 segments) | 78 | 54% | 68% | 82% | 73% | 228 |

The two longer matches were limited to their first ~55 camera segments (about the first set)
to fit the local run; tracking is cached per chunk, so the rest resumes with the same command.

## Component quality

| Component | Measured | Note |
|---|---|---|
| Crowd / close-up filter | 99.2% agreement with 480 VLM labels (5-fold CV); 16/16 spot-checked by eye | Keeps 48% of broadcast time |
| Court calibration | ≈2 px keypoint reprojection on checked frames | 14-keypoint model + RANSAC; fits above 6 px rejected |
| Player positions | 98–100% of in-play sample frames | Far player needs a 1.5× upscaled court crop |
| Ball detection | 74–82% of in-play frames | Pretrained TrackNet, not yet fine-tuned on US Open |
| Point alignment to official data | 94% of points (full match) | Server end consistent 100% when detected (the aligner rewards this, so it is a consistency check) |
| Serve detection | 54–95% of points | Weakest on the men's final (big serves) |
| Rally length | 64–68% within ±1 shot | Overcounts 0–1 shot points; undercounts 9+ shot rallies |
| Forehand / backhand | 90% on 51 gold labels (far 96%, near 83%) | Geometric rule; left-hander flip verified on Draper |
| Bounce coordinates | 70–82% of shots | CatBoost bounce model + two-segment kink fallback |
| Serve speed | Official radar joined for 97 of 105 points | Video-only estimate unreliable (r ≈ 0.1) |

## What the VLM was good for

| Task | Model | Outcome |
|---|---|---|
| Scene labels (main camera or not) | Qwen3-VL-8B, one-word yes/no | Reliable — used to train the filter |
| Stroke labels, tennis semantics prompt | Qwen3-VL-8B | Answered "forehand" for nearly everything |
| Stroke labels, "racket left or right?" | Qwen3-VL-8B | 61% vs gold — strong "left" bias; not used |
| Classifier trained on those labels | MobileNet embedding + logistic | 74% vs gold; rule (90%) used instead |

Qwen3-VL-8B (4-bit, MLX) labeled whole frames well but could not tell racket side on ~60 px
players. Stroke gold labels were made by visually reviewing canonical hitter crops; they live in
`pipeline/eval/stroke_gold.csv`.

## Revised cost for all 178 matches

The filter keeps 47.7% of broadcast time, more than the 30% assumed earlier: 369 h of video
becomes 176 h of main-camera footage, or 19.0M frames at 30 fps. TrackNet dominates at
~366 GFLOPs per frame.

| Assumed end-to-end L40S throughput | L40S cost incl. ~3 h overhead |
|---:|---:|
| 150 fps | $60 |
| 250 fps | $38 |
| 350 fps | $28 |

Cost = $1.57/h × (19.0M frames ÷ fps + 3 h), against a $75 budget. The L40S throughput is an
assumption, not yet measured. Measured locally: 11–12 fps on the M3 Pro, which would take
~19 days for the full set.

## Next steps, in order

1. Run one match on an L40S for an hour (~$1.60) to replace the throughput assumption with a
   measurement.
2. Fine-tune TrackNet on ~2,000 labeled US Open frames — the largest lever for rally length,
   serve detection and bounce coverage.
3. Add a serve detector for big servers (the men's final found only 54% of serves).
4. For pre-2011 matches without official data, OCR the on-screen serve-speed and score graphics.
5. Expand the gold set to ~200 strokes across eras, especially near-court players and
   left-handers.

## Next-step tooling (built, not yet run on footage)

Steps 2–5 are implemented and covered by synthetic-data tests (`pipeline/tests/`); none has been
run on real broadcasts yet, so the pilot numbers above are unchanged.

**2. TrackNet fine-tune.** `balllabels --action sample` picks ~2,000 frames spread evenly over the
given matches, weighted toward where the pretrained model fails: 35% in-rally misses, 20% around
serves, 15% bounces, 15% far-court, 15% random (between-point negatives). Each frame is pre-labeled
from the cached track (detection, interpolation, or a quadratic fit across the gap), so review is
mostly accept/nudge. Matches with scene segments but no tracks contribute unlabeled frames, which is
how older eras get in. Labels are versioned in `pipeline/eval/ball_labels.csv`; the split is by camera
segment (or whole match with `--holdout`). `balltrain` fine-tunes from the pretrained weights with
the same input and 256-class heatmap target, then writes `.cache/weights/tracknet_ft.pt` and a report
comparing pretrained and fine-tuned precision/recall at 5 and 10 px, per bucket, on the held-out
split. Tracking picks up the fine-tuned weights automatically; cached chunks tracked with other
weights get only the ball re-run (court and players are reused).

**3. Serve detector.** `serve.py` finds serves from the server's body rather than the ball: settled
stance behind the baseline, the person box growing upward as the arms go above the head (contact at
its tallest point), and no opponent shot in the previous 2 s. It also needs one piece of ball or
sound evidence (tracked toss, racket onset in the audio, ball leaving toward the opponent, a return,
or a raw hit at contact). Accepted serves are flagged on or inserted among the hits, and the server's
pre-serve ball bounces are dropped (a likely source of the 0–1 shot overcount). All candidates,
with the reason for any rejection, go to `serve_candidates.parquet`. `eval/serve_eval.py` re-runs
events and alignment from cached tracks with and without the detector, sweeps the threshold, and
lists why each missed serve was rejected. The Sinner–Fritz final is the match to tune on.

**4. OCR for matches without official data.** Official point-by-point data exists only from 2011, and
is also missing for AO 2024 and all 2025+ events. `ocr` locates the score bug (rows starting with a
player's surname, plus its number columns) and the speed graphic on ~160 sampled frames, then reads
them once per second. Score cells are recognised one by one because text detectors drop isolated
"0"s. Reads are debounced into score states, and tennis rules (`score.py`: deuce, tiebreaks, and
final-set formats by tournament and year) expand them into points in the official schema, including
server and winner. Points the broadcast never showed are inferred and flagged. Each speed readout
attaches to its point (a second readout means a second serve). `align` falls back to
`ocr_points.csv` automatically and uses the score-change times as an extra constraint. One limitation: the video alone
cannot distinguish "player 1 starts far and player 2 serves first" from "player 1 starts near and
serves first". When the speed graphic names the server this is resolved; otherwise
`align_summary.json` sets `server_identity_ambiguous` and near/far names may be swapped. Also fixed:
2011–2017 official files (no `RallyCount`/`ServeNumber`) previously broke `align`.

**5. Gold set expansion.** `gold` targets ~200 forehand/backhand labels across broadcast eras
(2000–06 SD, 2007–12, 2013–26), 60% near-court and 30% left-handed where available. Because
Qwen3-VL-8B showed a strong "left" bias, answers only count if they flip when the crop is mirrored.
Two independent questions vote (racket side in one word; racket and chest points), and "both
hands on the racket" vetoes a forehand. Crops are re-decoded at the source resolution. The protocol
is first scored against the existing hand-checked labels, and nothing is written unless it agrees at
least 90%. If the 8B model fails that, try `--model mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`.
New rows carry `labeler=qwen3-vl:…`, era, side, hand and the raw votes. Contact sheets for spot-checks
go to `.cache/gold_sheets/`, and `strokes` reports rule accuracy by side, hand, era and labeler.

## Output files per match

| File | Contents |
|---|---|
| `shots.csv` | player, stroke (serve/FH/BH/overhead), volley, contact x/y (m), bounce x/y (m), in/out, audio-snapped time, point number |
| `points.csv` | official point number, server, rally count (video + official), serve speed, serve width/depth, winner, shot sequence |
| `ball_trajectory.csv` | every 30 fps frame: ball pixel x/y, ground projection (meters), bounce flag, point number |
| `player_positions.csv` | 7.5 fps: near/far player court x/y (m), bounding box, player name, point number |
| `court_maps.png`, `point_N.mp4` | per-player serve and shot bounce maps; annotated clip of the longest rally |
