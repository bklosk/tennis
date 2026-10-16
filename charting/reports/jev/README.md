# Jev 1.14 (images + Box) vs the pipeline, on one charted match

One held-out test match: **2023 US Open women's QF, Ostapenko v Gauff** (`8gP591_wx_k`, 56 min,
1080p, 93 MCP points). Run 2026-10-15 on `jev-1.14.0` (Images and Box are a TypeSafe preview
feature). The benchmark spent **$0.392** of the $1.50 cap (5,838 requests, none failed); labelling
the dev matches for the tuning below brought the total to $1.21.

Jev got exactly what gpt-6-luna got: the same contact sheets, the same text, the same questions
(Decisions `predicate` questions became Jev `noul` questions). The one change: Jev's minimum
image side is 280 px, so the 384×216 full-court context frame in the stroke request was upscaled
to 498×280. Luna's answers are the cached production ones. MCP is the truth throughout. Box is
compared with RF-DETR, which is a reference, not ground truth; the charted-fact checks below are
neutral between the two.

`scripts/jev_bench.py` sends the requests (cached, ledger in `outputs/jev/`, hard cap in
`uso/jev.py`); `scripts/jev_score.py` writes `metrics.json` here and two spot-check images to `outputs/8gP591_wx_k/jev/`
(broadcast frames, gitignored).

## Contacts and rally length (`uso/onset_api.py` → decoder)

4,434 onset sheets, each asking what the near and far player are doing at the sound.

| | luna | Jev |
|---|---|---|
| Contact AUC per onset (5,012 onset×player labels from MCP-constrained decoding, 280 contacts) | 0.840 | **0.889** |
| near / far player | 0.869 / 0.851 | **0.902 / 0.880** |
| Rally length exact, API only, production decoder settings | 51.7% (87 pts) | **63.3%** (90 pts) |
| Rally length within one | 83.9% | 84.4% |
| Rally bias (contacts) | −0.23 | −0.52 |
| Points the decoder found (MCP has 93) | 95 | **122** |

* **Paired, on the 86 points both aligned:** exact 52.3% → 64.0%; Jev alone right on 21 points, luna
  alone on 11 (sign test p = 0.11, two-sided). Within one: 84.9% → 86.0%.
* **Production for reference:** luna + learned contact model is 55.2% exact / 87.4% within one. That
  model was trained on luna's answers, so it isn't a fair partner for Jev; with it, Jev gives 63.3% /
  85.6%.
* **Retuning** bias and serve threshold on this match (optimistic for both) doesn't move luna; Jev
  reaches 64.4% / 86.7% and 113 points.
* **The catch is serve detection.** Jev's `serve` answers fire more often, so the decoder opens 29
  extra points. Alignment skips them on a charted match, but an uncharted match has nothing to
  align against. Jev also undercounts contacts more (bias −0.52).
* Labels: inside each aligned point, the decoder picks MCP's number of contacts using the mean of
  both models' scores, so neither model defines its own truth. Server and serve times still come from
  the luna production decode.

## Strokes (`uso/strokes.py`), 260 rally contacts with an MCP side

| | luna | Jev | pose model (production) |
|---|---|---|---|
| Forehand/backhand, asked directly | 49.6% (AUC 0.61) | 57.3% (AUC 0.66) | **97.3%** |
| near / far player | 52.7% / 46.6% | 68.2% / 46.6% | |
| Racket image-side question → forehand | 56.5% (AUC 0.68) | 70.8% (AUC 0.78) | |
| Two hands on the racket vs MCP backhand, AUC | 0.58 | 0.72 | |
| Stroke family accuracy (93% groundstrokes) | 90.4% | 90.0% | |
| Slice AUC (10 slices) | 0.64 | 0.76 | |

Jev is better than luna on every stroke question but nowhere near pose, and on the far player
(facing the camera) it is still a coin flip. Both found 2 of the 10 slices. Jev calls 9
groundstrokes volleys where luna calls 9 of them slices.

## Box: players vs RF-DETR, 416 contact frames

Full frames at 1568×882, snapped to RF-DETR sample times (within 0.125 s of the contact). Two
passes: the first asked for "whole body and racket", which made Jev's boxes wider than RF-DETR's
person boxes (1.17× near, 1.35× far) and shifted them toward the racket. The second, "person
only", is the fair one and is shown here.

| | near player | far player |
|---|---|---|
| RF-DETR box size (px, 1080p) | 100 × 190 | 50 × 92 |
| IoU median | 0.62 | 0.44 |
| IoU ≥ 0.5 / ≥ 0.7 | 75% / 35% | 39% / 8% |
| Ceiling: RF-DETR vs itself rounded to Jev's 0.01 grid | 0.88 | 0.78 |
| Wrong person (IoU 0) | 0.7% | 4.6% |
| Court position difference, median / p90 | 0.27 m / 0.53 m | 0.65 m / 1.36 m |
| Confidence vs IoU, Spearman | −0.01 | 0.18 |

Charted-fact checks with each source's positions at the same frames:

| | RF-DETR | Jev (person only) | Jev (with racket) |
|---|---|---|---|
| Server behind the baseline | 100% | 95.2% | 86.8% |
| Returner position, wide vs T serve, AUC | 0.93 | 0.91 | 0.92 |
| Receiver position, shot direction 1 vs 3, AUC | 0.999 | 0.993 | 0.993 |
| Distance from net, net shots vs others, AUC | 0.89 | 0.89 | 0.88 |

Jev's boxes are edges rounded to 2 decimals (19 px × 11 px steps at 1080p), which alone caps IoU
for the 50 px-wide far player at ~0.78. Confidence does not say when a box is bad, so it can't be
used to gate the far player. The positions keep most of the charted signal, but RF-DETR is
better on every check, and it runs locally.

## Box: the ball (experiment, no ground truth)

The same frames, with "the tennis ball" plus a `noul` asking whether it is visible. See
`outputs/8gP591_wx_k/jev/ball_crops.jpg` (yellow: Jev's ball box) and `box_overlays.jpg` (green: Jev, first pass with racket;
magenta: RF-DETR).

* Median ball box **38 × 54 px**, for a ball of roughly 8 px. Some boxes cover the scoreboard or
  half the court.
* On the 156 frames within 0.05 s of a detected contact, the box is at the hitter only 50% of the time.
* `ball_visible` median 0.52. It is low on some misses (0.21 on a box on the scoreboard), but not
  reliably.

Not usable for ball position. The pipeline's audio-and-pose route stays.

## Cost

| | requests | Jev tok/request | Jev $ | luna tok/request | luna $ (same requests) |
|---|---|---|---|---|---|
| Onset labels | 4,434 | 1,163 | 0.217 | 669 | 0.297 |
| Stroke questions | 286 | 1,784 | 0.021 | 998 | 0.029 |
| Image side | 286 | 1,127 | 0.014 | 639 | 0.018 |
| Box (two passes) | 832 | 4,013 | 0.140 | — | — |

Jev counts about 1.75× luna's tokens for the same image but costs $0.042/M against $0.10/M, so
the Decisions-API stages cost **27% less** ($0.25 vs $0.34 per match). Box at $0.07 per pass per
match is cheap but replaces a local model that is free and better.

## After the benchmark: tuned and switched (2026-10-15)

The four dev matches were labelled with Jev the production way: strong onsets, then a soft pass
chosen by Jev's own answers ($0.82). Then:

* **Serve over-firing fixed** with a serve threshold per court end, tuned on dev (near 0.8, far
  0.6). Jev reads the near player's pre-serve routine as a serve: 498 onsets at p ≥ 0.4, against
  luna's 200.
* **Decoder retuned** on dev with an objective that also counts invented points (exact × coverage
  × precision).
* **Contact model retrained** on Jev's answers; it blends at 0.7 / −1.0.

| | luna production | Jev production |
|---|---|---|
| Dev, 4 matches (decoder in-sample, contact model leave-one-out): exact / within one | 66.3% / 86.7% | 65.0% / 88.0% |
| Dev: detected points that are real | 89.4% | **95.4%** |
| Held-out 8gP591_wx_k: exact / within one | 55.2% / 87.4% | **60.7%** / 84.5% |
| Held-out: points found (93 charted) | 95 | 89 |
| Held-out, 82 points both aligned: exact | 54.9% | 61.0% (9 vs 4 discordant, p ≈ 0.27) |

Rally length is level within noise. Jev invents fewer points, and with the pose-only stroke model it
costs ~$0.24 per video-hour against ~$0.37. It is now the default for new matches
(`uso/labeler.py`). The published evaluation stays on luna.

## Verdict (from the benchmark, before tuning)

* **Onset labels: worth switching, after one fix.** Jev separates contacts from non-contacts
  better and, on the 86 points both aligned, gets 10 more rally lengths exactly right, for 27% less. Before
  it replaces luna it needs serve detection tamed (a stricter serve threshold or a serve-specific
  question; 113–122 points found for 93) and a run on the dev matches to tune the decoder and
  retrain the learned contact model on Jev's answers. One match (p = 0.11) is not enough to
  switch on.
* **Strokes: keep pose.** Jev improves every API stroke question, but forehand/backhand is 57–71%
  against pose's 97%. Its answers could still be features for the stroke model (two hands: AUC
  0.72; slice: 0.76), which needs Jev answers on the training matches.
* **Box for players: keep RF-DETR.** Usable near player, weak far player, coarse edges, and a
  confidence that doesn't help.
* **Box for the ball: no.**
