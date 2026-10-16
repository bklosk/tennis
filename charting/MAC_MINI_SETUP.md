# Running the charting pipeline on the Mac mini

What the mini needs to chart the rest of the US Open archive: the free **vision** stages for the 103
matches that have none (`scripts/batch_mac.py`), then the paid **decode** with Jev onset labels for
all 147 that remain (`scripts/decode_batch.py`). Written 2026-10-16. Each step says whether it was
checked on a clean checkout (a fresh clone, empty `outputs/`, real Space) or only read from code.

## What runs where

| | Needs | Cost | Time on the M3 Pro (36 GB) these were measured on |
|---|---|---|---|
| Vision, 103 matches (218 video-h) | the GPU (MPS), ~25 MB of outputs per match | free | ~43 h (measured: 92 video-h in ~18 h, two matches at a time) |
| Decode, 44 matches already through vision (92 video-h) | Jev API, GPU for stroke features | ~$22 | ~3 h at two slots (estimated from per-stage timings) |
| Decode, the other 103 (218 video-h) | same | ~$53–58 | ~6.5 h at two slots (estimated the same way) |

Jev labels cost about $0.24 per video-hour. Time on the mini depends on its chip: **time one match
before committing to the whole run** (`--only VIDEO_ID`). I have not measured memory; start with
`--slots 1`, watch it, then try 2.

## 1. The machine

* **Apple silicon.** The pose and people stages hard-code the `mps` device (`uso/pose.py`,
  `uso/people.py`); an Intel mini will not work.
* **Disk: ~50 GB free.** Videos run 2 GB (median) to 8 GB (largest). Each is downloaded, used and
  deleted, with a few in flight at once. Outputs for all 160 archived matches come to about 4 GB.
* **Tools:** `brew install ffmpeg uv git`. `ffmpeg` is called as a command (decode and audio). `uv`
  fetches Python 3.12 itself (`requires-python >=3.11,<3.13`).
* **Keep it awake.** Both runners hold `caffeinate` while they run. If it sleeps anyway, set
  System Settings → Energy → prevent automatic sleeping.

## 2. Clone and install (checked)

```bash
git clone https://github.com/bklosk/tennis.git ~/dev/tennis
cd ~/dev/tennis/charting && uv sync --frozen    # the first sync downloads torch and the other wheels
uv run pytest -q          # 87 tests, 1–2 min
```

## 3. Credentials

* **`~/.s3cfg`** for the `benklosky-data` Space (`scripts/archive_videos.py` reads only these two
  values, with the endpoint hard-coded to nyc3). The same key is `SPACES_KEY` / `SPACES_SECRET` in
  the website repo's `.env`:

  ```ini
  [default]
  access_key = ...
  secret_key = ...
  ```

* **`~/dev/tennis/.env`** (gitignored, `chmod 600`) with the Jev key:

  ```
  TYPESAFE_API_KEY=...
  ```

  (`JEV_KEY` also works. On the Mac that did the benchmark the key is `JEV_API_KEY` in the website
  repo's `.env`.) `OPENAI_API_KEY` is only needed for `--labeler luna`.
* **GitHub push access** (HTTPS credentials or `gh auth login`) for the vision run only: it commits
  each match's small outputs and pushes them to the `charting-results` branch.

## 4. Court weights: copy one file by hand

`.cache/weights/court.pt` (42,312,975 bytes) is the 14-keypoint court network from
yastrebksv/TennisCourtDetector, used by the vision run's scene stage. Nothing downloads it. Copy it
from the Mac that has it, and compare the checksum:

```bash
mkdir -p ~/dev/tennis/.cache/weights
scp OLD_MAC:dev/tennis/.cache/weights/court.pt ~/dev/tennis/.cache/weights/
shasum -a 256 ~/dev/tennis/.cache/weights/court.pt
# 09aa8c4338459ba1d643f2dc329f45f464dedec3720fccc1a4abfd1f7b464d04
```

The upstream README links the weights ("Pretrained model") at
https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view; I did not check that its
file matches this checksum, so use the copy if you can. Decoding does not need `court.pt`.

The other weights download themselves on first use:

* `yolo11m-pose.pt` (ultralytics → `.cache/weights/`): checked from a clean checkout.
* `rf-detr-small.pth` (~390 MB, `rfdetr` → `~/.roboflow/models/`, md5-checked): read from the
  package source, not run on a clean machine. It is needed by the vision run only.

## 5. Pull the state (checked)

`outputs/` is gitignored, so a fresh checkout has none of what the runs need. Everything small goes
through the Space:

```bash
cd ~/dev/tennis/charting && uv run python scripts/space_state.py pull     # ~1.3 GB
```

It writes `outputs/archive_manifest_latest.csv` (the archive manifest); the Jev decoder parameters
(`decoder_params_jev.json`), contact model (`hitmodel_jev.pkl`), pose-only stroke model
(`stroke_model_pose.pkl`) and Jev spend ledger (`jev/ledger.jsonl`); and the vision outputs of all
57 matches already stored (the 44 batch matches and the 13 evaluation matches). It also records
those 57 in `outputs/batch/prior_stored.json`, so `batch_mac.py` neither redoes their vision stages
nor re-uploads them. It never overwrites a local file without `--force`. The models and ledger were
uploaded from the first Mac on 2026-10-16 (`space_state.py push-state`); re-run that there if you
retune anything.

## 6. Vision for the 103 untouched matches (free)

`batch_mac.py` lives on the `charting-results` branch, which it also pushes results to. Run it from
a worktree on that branch, sharing `outputs/` with the main checkout (read from code and from how
the first Mac was set up; not run on the mini):

```bash
cd ~/dev/tennis && mkdir -p outputs downloads .cache
git worktree add -b vision .claude/worktrees/vision origin/charting-results
cd .claude/worktrees/vision
for d in outputs downloads .cache; do ln -s ~/dev/tennis/$d $d; done
ln -s ~/dev/tennis/.env .env
cd charting && uv sync --frozen && cd ..
uv run --project charting python charting/scripts/batch_mac.py --hours 48 --slots 2
```

* Progress page: http://127.0.0.1:8765 (on a headless mini: `ssh -L 8765:localhost:8765 mini`).
* No new match starts after `--hours`, or once `outputs/batch/STOP` exists; running ones finish.
  Resumable: a match with `pose.parquet` counts as done.
* Run vision on **one machine at a time**: both machines push to `charting-results` and write to the
  same Space prefix.
* Start with `--slots 1` if you are unsure about memory.

## 7. Decode with Jev (paid)

**First raise the cap.** `HARD_CAP_USD` in `charting/uso/jev.py` is still **$1.50** (the benchmark
limit), and the pulled ledger already counts $1.21 of it. Set it to what you approve. The finished
estimate is ~$75–80 for all 147 (~$22 for the 44 now, ~$53–58 for the 103 after vision). The cap
counts the ledger on the machine that runs it, so decode on one machine at a time. The runner stops
on its own when it hits the cap and a rerun resumes.

```bash
cd ~/dev/tennis/charting
uv run python scripts/decode_batch.py --write-table      # rebuild data/batch_matches.csv from the vision outputs present
uv run python scripts/decode_batch.py --dry-run          # the plan and estimated spend (checked: 44 matches, $22.25)
uv run python scripts/decode_batch.py --only VIDEO_ID    # one match first, to time it
uv run python scripts/decode_batch.py --slots 2
```

Results land in `outputs/VIDEO_ID/` (`shots.csv`/`points.csv` for the 32 matches with an MCP record,
`uncharted_*.csv` for the rest) and are uploaded to the Space next to the vision outputs; rows say
`source=model:uso-0.2-jev`. Re-run `--write-table` after vision finishes so the 103 are included.

A full decode of one match on a clean checkout was checked: Ostapenko v Gauff (`8gP591_wx_k`), with
the video fetched from the Space and cached Jev answers, gave 89 points and 382 contacts, the same
as the first Mac, with no new paid requests and the pose weights downloaded on the way.

## Before the big run

Jev was tuned on 4 dev matches and checked on one held-out match (60.7% exact rally length against
luna's 55.2%, with fewer invented points; `reports/jev/README.md`). I suggested running the other 9
held-out matches first (~$3.30) for a firmer comparison. That isn't wired up yet. Labelling them is
safe (`scripts/post.py VIDEO_ID --steps onsets --labeler jev` writes only `onset_jev.parquet`), but
scoring needs `scripts/jev_score.py` generalised from its one-match version. Don't use `post.py
--steps shots` for it: that rewrites `shots.parquet`, the luna results for that match.

## What I did not check

* Any vision run on the mini, including the `charting-results` worktree recipe in step 6.
* `rf-detr-small.pth` downloading on a clean machine, and the Google Drive file matching `court.pt`.
* Memory use with two slots, and the mini's speed.
