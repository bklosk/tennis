# GPU droplet pilot (DigitalOcean L40S)

Measured on a DigitalOcean `gpu-l40sx1-48gb` droplet (NVIDIA L40S, 8 vCPUs, $1.57/h) with
`pipeline/eval/gpu_bench.py`. Total spend for all benchmark runs: $0.62 of the $5 budget.

## What could and could not run

YouTube serves a "confirm you're not a bot" check to datacenter IPs, both from the cloud agent
and from a DigitalOcean droplet. This repo does not bypass access controls, so the three-match
accuracy pilot needs videos supplied from a machine that already has them (see *Running the
pilot* below). Throughput does not depend on the match being a broadcast, so the benchmarks ran
on openly licensed 720p tennis footage (*2018 Davis Cup Americas Zone – Uruguay vs Mexico*,
Wikimedia Commons, CC BY-SA 4.0), looped to 6 minutes and re-encoded as 720p60 H.264 at
YouTube-like bitrate. A fixed broadcast-style court calibration stands in for the court model's
output (the model still runs, so its cost counts).

## Results

End-to-end tracking (court + TrackNet + players, 30 s chunks, 720p60 source):

| Revision | Tracking fps | GPU busy |
|---|---:|---:|
| Mac-era code on CUDA, sequential | 61.6 | 38% |
| + decode prefetch | 87.6 | 56% |
| + compiled TrackNet, lighter court and player passes | 141.9 | 48% |
| + player detection overlapped with TrackNet | **191.7** | 64% |

For comparison, the M3 Pro ran the whole tracking stage at 11–12 fps.

| Component | Before | After | Change |
|---|---|---|---|
| TrackNet | 170 fps (eager FP16) | 275 fps | `torch.compile`, batches padded to a fixed size |
| Player detection | 19.7 ms per sampled frame | 13.6 ms | near player at 640 px, batches of 32 |
| Court calibration | 201 ms per probe | 6.6 ms | FP16, power-of-two probe batches, probes every 2 s |
| TrackNet input downscale | 0.32 s per 30 s chunk (CPU) | on GPU | bit-exact with `cv2.resize` |
| Chunk decode (720p60 → 30 fps) | 525 fps software | — | NVDEC was slower (224 fps), so software is the Linux default |
| Scene-pass decode (2 fps samples) | 35× realtime | 150× realtime | optional `--scene-keyframes` (samples once per keyframe interval) |
| Hitter crops | one seek-and-decode per hit | from frames in memory during tracking | crops stage now only decodes leftovers, one span per chunk |

## Revised full-dataset cost (178 matches)

The scene filter keeps 47.7% of broadcast time, so 369 h of video is 176 h of main-camera footage,
or 19.0M frames at 30 fps.

| Item | Estimate |
|---|---:|
| Tracking at 191.7 fps | 27.5 h → $43 |
| Scene pass, keyframes only / full decode | 2.5 h → $4 / 10.6 h → $17 |
| Events, crops, alignment, strokes, reports | ~1.5 h → $2 |
| **Total** | **$49 with keyframe scenes, $62 with full decode** |

The binding constraint for the full run is getting the videos onto the droplet, not the GPU:
downloads have to happen from a residential connection, and 369 h of 720p is roughly 400 GB.

## Running the pilot

From a machine with the videos in `downloads/` (for example the Mac used for the first pilot):

```bash
cd pipeline
export DIGITALOCEAN_ACCESS_TOKEN=...   # scoped token with droplet + ssh_key access is enough
uv run python -m tennis_pipeline.cloud pilot KCcKkUnjbzA Fl33UXv6jKI Ce3dRYHWIBI --budget 4
```

The launcher creates an L40S droplet, uploads code, weights and videos (dependencies install in
parallel), reuses any local `scene_classifier.npz` and per-match segments, then runs scenes (if
needed), track, events, crops, align, strokes and report with per-stage timings. Results sync back
to `outputs/` every few minutes and at the end (`outputs/gpu_pilot_summary.json`,
`pilot_stage_times.jsonl`, `pilot_metrics.json`).

Every droplet has two kill switches: a self-destruct timer (armed over SSH and verified before
any work) that deletes it through the API when the budget is reached, even if the launcher dies,
and a local watchdog that stops the job, syncs results and destroys it first.
`uv run python -m tennis_pipeline.cloud cleanup` removes anything left by an interrupted run.

To re-measure throughput: `uv run python -m tennis_pipeline.cloud bench VIDEO.mp4 --budget 0.6`.
