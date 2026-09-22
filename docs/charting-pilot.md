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

## Output files per match

| File | Contents |
|---|---|
| `shots.csv` | player, stroke (serve/FH/BH/overhead), volley, contact x/y (m), bounce x/y (m), in/out, audio-snapped time, point number |
| `points.csv` | official point number, server, rally count (video + official), serve speed, serve width/depth, winner, shot sequence |
| `ball_trajectory.csv` | every 30 fps frame: ball pixel x/y, ground projection (meters), bounce flag, point number |
| `player_positions.csv` | 7.5 fps: near/far player court x/y (m), bounding box, player name, point number |
| `court_maps.png`, `point_N.mp4` | per-player serve and shot bounce maps; annotated clip of the longest rally |
