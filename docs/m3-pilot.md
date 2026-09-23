# Full-match pilot on the M3 Pro

The three 2024 pilot matches were tracked to the end on the M3 Pro; pilot 1 had stopped the two
long ones after about a set. The run was used to speed up tracking on Apple silicon and to retune
serve detection against the official point-by-point record. All accuracy numbers below are on
full matches.

## Headline

| | Pilot 1 / before | Now |
|---|---:|---:|
| Tracking throughput, M3 Pro, end to end | 11.5 fps | 32.2 fps |
| Scene-pass decode | 5× realtime | 86× realtime |
| Official points aligned (3 matches) | — | 452 / 468 (96.6%) |
| Serves detected, aligned points | 54–95% | 85–97% (91% overall) |
| Server end consistent when a serve is detected | 96–99% (current detector) | 99.4–100% |
| Serves flagged vs official serve attempts | 1.85–2.04× | 1.00–1.13× |
| Rally length within ±1 shot | 55–61% (current detector) | 58–71% (64% overall) |
| Forehand/backhand vs 51 gold labels | 90% | 86% (90% on shots in aligned points) |

## Per-match results

| Match | Main-camera min tracked | Points aligned | Serve detected | Rally ±1 | Ball found in play | Players found | Shots with bounce x/y | Official serve speed joined |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Navarro vs Badosa · QF | 27.3 of 29.1 | 107 / 112 | 92.5% | 64.5% | 74% | 98% | 71% | 99 points |
| Draper vs de Minaur · SF | 47.0 of 49.8 | 173 / 181 | 97.1% | 70.5% | 78% | 97% | 77% | 167 points |
| Sinner vs Fritz · F | 55.1 of 60.3 | 172 / 175 | 84.9% | 57.6% | 80% | 99% | 78% | 163 points |

Main-camera minutes that were not tracked are chunks where fewer than half of the court-model
probes calibrated. Server end agreement was 100%, 99.4% and 100%.

## Tracking speed on Apple silicon

Measured on real main-camera chunks from the men's final:

| Change | Before | After |
|---|---:|---:|
| TrackNet on the Neural Engine (Core ML) instead of the M3 GPU | 19.6 fps | 44 fps (38 fps in the tracking loop) |
| End-to-end tracking (court + TrackNet + players, 30 s chunks) | 11.5 fps | 32.2 fps |
| Chunk decode, software instead of VideoToolbox | 290 fps | 989 fps |
| Scene pass (2 fps samples), software decode | 5× realtime | 86× realtime |
| TrackNet on the M3 GPU, contiguous instead of channels_last | 17.4 fps | 19.6 fps |

- **Neural Engine.** The Core ML model is converted from whichever ball weights are active (so a
  future fine-tune converts automatically) and cached by weights hash. The key change is on the
  output: the model returns a one-channel map that is positive exactly where the 256-class
  heatmap's argmax exceeds 127, which is all the tracker uses. Returning the full heatmap, as the
  earlier Core ML backend did, ran at 11.6 fps because of output transfer. The three input frames
  are passed as separate raw-pixel arrays, so the CPU does no reordering or scaling. On 1,800
  frames, ball positions from the Neural Engine and from PyTorch agreed within 2 px on 99.8% of
  the frames both detected, and 8 frames were detected by only one of them; PyTorch FP16 and FP32
  were identical. Memory stayed flat, unlike the earlier backend.
- **GPU work.** Player detection runs on the GPU while TrackNet runs on the Neural Engine. Also
  giving the GPU a share of the ball frames was slower end to end (25–28 vs 32 fps). The Metal
  backend asserts when two threads issue GPU commands at once, so when TrackNet itself runs on
  the M3 GPU (`--ball-backend torch`), players and ball run one after the other.
- **Decode.** Hardware decoding loses on both platforms because frames are copied back and
  scaled on the CPU. Software decoding is now the default everywhere; `TENNIS_HWACCEL=videotoolbox`
  or `=cuda` opts in.

The run tracked 147k new frames in 76 minutes (Draper–de Minaur 32.1 fps, Sinner–Fritz 32.4 fps);
waiting on decode was 0.4% of that and inline hitter crops 1.1%. Events and crops for all three
matches then took 86 s, and alignment, strokes and reports 38 s. At 32 fps the full 19M-frame
dataset would still take about a week on this Mac, so the L40S remains the plan for the full run.

**Court model bug (all GPUs).** The L40S tuning had switched the court model to FP16. That loses
every keypoint on real broadcasts: its BatchNorm layers subtract running means of up to ~11,000
from activations of the same size, and FP16 only resolves steps of 8 at that magnitude. Every
chunk would have failed calibration. The L40S benchmark used a fixed calibration, so it did not
show. The court model is back to FP32, with a test; by estimate this adds 2–3% to the L40S
tracking time.

## Serve detection

The body-based serve detector found nearly every serve but flagged about twice as many serves as
there were: at its default threshold, only 43% of accepted candidates on Navarro–Badosa were real.
A false serve starts a new video point; because the server alternates every game, the aligner can
then slip by a game, which flipped player 1's starting end on that match and made the clothing-colour
identity check close to a coin flip (purity 0.54–0.55).

Changes:

- **Position features.** Real servers stand right at the baseline (median 0.04 m behind it) and
  about 1.05 m from the centre mark; false candidates were a median 1.65 m behind and 3.1 m wide.
  Two new candidate features, `behind` and `off_mark`, carry this.
- **Contact timing.** The server's box is tallest at the toss, a median 0.4 s (up to 1.3 s) before
  contact, so detected serves landed before the real serve hit and the hit survived as an extra
  shot. Contact is now the server's own hit or the first sound up to 1.5 s after the toss peak.
- **Fitted weights.** `eval/serve_fit.py` aligns each match with rule-only serve detection, labels
  candidates against the official points (a real serve starts an aligned point on the official
  server's side; a candidate inside an aligned rally is not a serve), and fits the logistic weights.
  833 candidates that pass the hard gates were labeled across the three matches.

Held out one match at a time (weights fitted on the other two), at threshold 0.5:

| Held-out match | Precision, hand-set → fitted | Recall, hand-set → fitted |
|---|---:|---:|
| Navarro–Badosa | 0.48 → 0.84 | 1.00 → 0.83 |
| Draper–de Minaur | 0.51 → 0.84 | 1.00 → 0.96 |
| Sinner–Fritz | 0.53 → 0.95 | 1.00 → 0.94 |

End to end on the same cached tracks (previous code → current code):

| Match | Serves flagged | Points aligned | Serve detected | Server end agreement | Rally ±1 |
|---|---:|---:|---:|---:|---:|
| Navarro–Badosa | 302 → 167 | 101 → 107 | 98.0% → 92.5% | 96.0% → 100% | 55.4% → 64.5% |
| Draper–de Minaur | 488 → 294 | 171 → 173 | 98.8% → 97.1% | 97.6% → 99.4% | 57.3% → 70.5% |
| Sinner–Fritz | 484 → 255 | 159 → 172 | 88.1% → 84.9% | 98.6% → 100% | 61.0% → 57.6% |

On Sinner–Fritz the previous code aligned 13 fewer points; counted against all 175 official
points, rally length within ±1 went from 55% to 57%. Threshold sweep (serve detected / rally ±1):

| Threshold | Navarro–Badosa | Draper–de Minaur | Sinner–Fritz |
|---|---|---|---|
| Rule only (no detector) | 84.8% / 63.8% | 92.4% / 69.8% | 61.6% / 58.1% |
| 0.5 | 93.5% / 63.6% | 97.1% / 67.6% | 90.2% / 56.6% |
| **0.6 (default)** | 92.5% / 64.5% | 97.1% / 70.5% | 84.9% / 57.6% |
| 0.7 | 91.6% / 65.4% | 96.5% / 71.5% | 79.5% / 58.5% |

The weights and threshold were chosen on these three matches, so the held-out precision and
recall above are the better guide for new matches. Re-run `eval/serve_fit.py` as more matches are
processed, especially other eras.

## Forehand / backhand

86% on the 51 gold labels (far 89%, near 83%). Two of the seven misses are Draper–de Minaur
shots in video points that no longer align to an official point, so the player, and with it the
left-hander flip, is unknown; on shots in aligned points the rule is at 90%, as in pilot 1. One miss
is a rally shot flagged as a serve and three are near-side geometry errors.

## Tried and reverted

| Idea | Result |
|---|---|
| Split TrackNet between the Neural Engine and the GPU | 25–28 fps vs 32 fps end to end |
| No rule-based serve within 2 s of the opponent's shot | Rally ±1 −2.7 points on Navarro–Badosa (real serves after spurious "hits" were lost) |
| Near-player hits from 2D velocity changes (returns after high bounces) | Rally ±1 +1.2 / −0.4 / −2.4 points on the held-out matches |
| End an attempt when the serve is called out | The in/out call was wrong on 13 of 64 serves that were in |

## Where rally counts still go wrong

All three matches, by official rally length:

| Official shots | Points | Within ±1 | Mean error | Over by 2+ | Under by 2+ |
|---|---:|---:|---:|---:|---:|
| 0 (double faults and similar) | 32 | 38% | +2.2 | 20 | 0 |
| 1–2 | 168 | 80% | +0.7 | 32 | 1 |
| 3–4 | 92 | 73% | −0.1 | 9 | 16 |
| 5–8 | 90 | 53% | −0.8 | 13 | 29 |
| 9+ | 70 | 40% | −4.6 | 5 | 37 |

- **Long rallies.** Hits are found where the ball's image-y direction reverses. After a high bounce
  on the near side the ball is already rising in the image and the return keeps it rising, so
  there is no reversal. The near-half track also jumps between the ball and false detections.
  Fine-tuning TrackNet (the labeling tooling is ready) is still the main lever.
- **Faults.** Only 18 of 168 second-serve points had the first-serve fault detected as a
  separate attempt, and the returner's knock-aways after an out serve count as rally shots.
- **Scene filter.** A few rallies are split by a short non-main-camera gap, or mostly dropped.
