# Batch run on a DigitalOcean GPU droplet

How to chart the whole dataset (198 verified US Open full matches, 418 h of video) on one
budget-capped L40S droplet, from the Mac that holds the videos. YouTube blocks datacenter IPs, so
the videos have to come from that Mac.

## Running it

```bash
cd pipeline
# 1. Prep on the Mac (free): scenes, audio onsets, official-data check, main-camera packs.
uv run python -m tennis_pipeline.batch prep --manifest usopen --pack
# 2. Check the plan it prints, then run. Rerun the same command to resume.
export DIGITALOCEAN_ACCESS_TOKEN=...
uv run python -m tennis_pipeline.cloud batch --manifest usopen --budget 70
# Progress (also printed by the launcher every 5 min):
uv run python -m tennis_pipeline.batch status --manifest usopen
```

`--manifest` takes a preset (`usopen`) or a file with one video id per line; video ids can also be
passed directly. `cloud batch --plan` preps and prints the plan without creating a droplet. A
fine-tuned ball model in `.cache/weights/tracknet_ft.pt` is uploaded and used automatically.

The plan reports the upload size, GPU hours and cost, and the upstream bandwidth needed to keep the
GPU busy. It also lists matches whose main-camera share is outside 20–80%. Those are worth a look
before paying for GPU time: the scene classifier was trained on 2024 broadcasts, and older eras may
need `prep --retrain-scenes court`, which relabels every match with the court model.

Results sync to `outputs/` every 5 minutes and at the end:

| File | Contents |
|---|---|
| `outputs/VIDEO_ID/*` | the per-match outputs of a local run (`shots.csv`, `points.csv`, ...) |
| `outputs/VIDEO_ID/batch_status.json` | state, time and any error of every stage |
| `outputs/batch_progress.json` | counts per state, tracking fps, GPU busy and waiting time, ETA |
| `outputs/batch_metrics.csv` | one row of quality metrics per match |
| `outputs/batch_summary.json` | totals, failures, serve-speed calibration |
| `outputs/batch_run.json` | droplet cost, hours, upload volume |

## What makes it efficient

| Where | What | Effect |
|---|---|---|
| Mac | Scene classification, audio onsets and the official-data check run in prep | Saves $4–17 of droplet time and makes packs possible |
| Mac | Main-camera packs: stream-copied segments, no re-encode, no audio | Upload drops from ~450 GB to ~320 GB; frames decode exactly like the source |
| Mac | Only per-sample scene probabilities are uploaded, not embeddings | About 4 GB less upload |
| Upload | Streams in match order, two at a time, at most 8 videos waiting | The GPU starts after the first match arrives; disk stays bounded |
| Droplet | One GPU worker loads and compiles the tracking models once | No per-match load or compile |
| Droplet | Two low-priority CPU workers run OCR, events, crops, align, strokes and report for finished matches | The GPU tracks the next match meanwhile |
| Droplet | Each video is deleted once its match is finished | Disk use stays at a few videos (500 GiB boot disk) |

Matches without official point data (before 2011 and from 2025 on) need OCR of the score graphics,
which reads the whole broadcast, so they upload as full videos. `prep --ocr-local` reads them on the
Mac instead so they can be packed too. That saves about 75 GB of upload for about 20 h of Mac CPU.

## What keeps it safe

- **Budget cap.** The droplet deletes itself when `--budget` is spent, even if the Mac is gone, and
  the launcher stops and syncs first.
- **Dead man's switch.** If the launcher stops polling for 45 minutes (laptop asleep or offline), the
  droplet deletes itself rather than wait, at full price, for videos that will not come. The launcher
  keeps macOS awake while it runs, and a network blip does not end the run.
- **Resume.** Every stage result is recorded per match. Rerunning `cloud batch` skips finished
  matches, continues partly processed ones from their synced tracks, and retries failed stages once.
- **Failure isolation.** A failing match is recorded and skipped. Three tracking failures in a row
  (missing weights, a CUDA problem) abort the batch instead of failing every match at GPU prices.

Post-processing only needs the synced outputs, so stages after tracking can also be rerun on the
Mac. `uv run python -m tennis_pipeline.batch run --manifest usopen` runs only the stages that are
not done yet, and tracks any untracked match locally.

## Estimate

| Item | Estimate |
|---|---:|
| Main camera (47.7% of broadcast time) | 199 h, 21.5M frames at 30 fps |
| Tracking at 190 fps (L40S benchmark) | 31 h |
| Cost at $1.57/h, incl. setup and the FP32 court model | about $52 |
| Upload with packs / full videos | ~320 GB / ~450 GB |
| Upstream needed to keep the GPU busy | about 23 Mbps with packs |

With less upstream bandwidth than that, the GPU waits for videos; the progress line reports how long.
Uploading faster than tracking is harmless: at most 8 videos wait on the droplet.

## Not yet validated

- **Throughput on real broadcasts.** The 190 fps figure came from openly licensed footage with a
  fixed court calibration, before the court model went back to FP32. The first matches' fps is in the
  progress line; the plan's `--assume-fps` can be adjusted.
- **Older eras.** Scene filtering, the court model and TrackNet have only been checked on 2024
  broadcasts. Pre-2007 matches are likely 4:3 SD video stretched to 1280×720.
- **OCR at scale.** The score-graphic OCR has synthetic-data tests only.
- **Two GPU workers.** `--gpu-workers 2` tracks two matches at once. On the L40S's 8 vCPUs, CPU is
  probably the limit; check `gpu_util_while_tracking` in the progress file first.

The batch code itself was tested on CPU: unit tests with stand-in stages (resume, uploads, crashes,
aborts), an end-to-end run of prep, packing and tracking from a pack, and the real
events-to-report stages on synthetic tracks.
