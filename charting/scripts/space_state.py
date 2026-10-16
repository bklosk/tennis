"""Move the pipeline's state between machines through the DigitalOcean Space.

    uv run python scripts/space_state.py push-state             # from the machine that tuned the models
    uv run python scripts/space_state.py pull [--videos ID ...] [--no-outputs] [--force]

`outputs/` is gitignored, so a fresh checkout has none of what decoding needs. Two kinds of state
go through the Space (s3://benklosky-data/tennis/):

* charting-state/: the Jev decoder parameters, the Jev contact model, the pose-only stroke model and
  the Jev spend ledger (the files `uso.labeler` names). `push-state` uploads them; `pull` downloads
  them.
* charting-outputs/YEAR/MATCH_ID__VIDEO_ID/: the vision outputs that scripts/batch_mac.py uploads for
  every match. `pull` fetches them into outputs/VIDEO_ID/ (every stored match, or --videos), so
  decoding does not redo the vision stages, and batch_mac.py sees those matches as done. It also
  records them in outputs/batch/prior_stored.json, the marker batch_mac.py uses for "already in the
  Space", so that it does not upload them all again on its first run.

`pull` also writes the archive manifest (usopen-video/manifest.csv, the same rows the batch runners
read) to outputs/archive_manifest_latest.csv. Existing local files are kept
unless --force is given, so a pull never overwrites work on the machine you run it on (above all
the ledger: the Jev spend cap counts the ledger on the machine that runs it). Credentials are read
from ~/.s3cfg and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_videos import BUCKET, s3_client  # noqa: E402

from uso import labeler  # noqa: E402
from uso.paths import OUTPUTS  # noqa: E402

STATE_PREFIX = "tennis/charting-state/"
OUTPUTS_PREFIX = "tennis/charting-outputs"
MANIFEST_KEY = "tennis/usopen-video/manifest.csv"
STATE_FILES = [labeler.params_path("jev").name, labeler.hitmodel_path("jev").name,
               labeler.stroke_model_name("jev"), "jev/ledger.jsonl"]  # relative to outputs/
SKIP = {"audio.wav", "rally_p.parquet"}  # regenerated locally; never uploaded
WORKERS = 8


def _remote_sizes(s3, prefix: str) -> dict[str, int]:
    return {o["Key"]: o["Size"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
            for o in page.get("Contents", [])}


def _download(s3, key: str, dst: Path, size: int, force: bool) -> str:
    if dst.exists() and not force:
        return "kept" if dst.stat().st_size == size else "kept (size differs)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    s3.download_file(BUCKET, key, str(part))
    if part.stat().st_size != size:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{key}: downloaded {part.stat().st_size} bytes, expected {size}")
    part.replace(dst)
    return "downloaded"


def push_state(s3) -> None:
    for rel in STATE_FILES:
        src = OUTPUTS / rel
        if not src.exists():
            raise SystemExit(f"{src} is missing: tune and fit first (tune_decoder.py, fit_hitmodel.py, fit_stroke_pose.py "
                             f"with --labeler jev)")
        s3.upload_file(str(src), BUCKET, STATE_PREFIX + rel, ExtraArgs={"ACL": "private"})
        print(f"[state] uploaded {rel} ({src.stat().st_size:,} bytes)")


def pull(s3, videos: list[str] | None, outputs: bool, force: bool) -> None:
    jobs: list[tuple[str, Path, int]] = []
    state = _remote_sizes(s3, STATE_PREFIX)
    for rel in STATE_FILES:
        if STATE_PREFIX + rel not in state:
            print(f"[state] {rel} is not in the Space yet (run push-state on the machine that has it)")
            continue
        jobs.append((STATE_PREFIX + rel, OUTPUTS / rel, state[STATE_PREFIX + rel]))
    body = s3.get_object(Bucket=BUCKET, Key=MANIFEST_KEY)["Body"].read()
    manifest = OUTPUTS / "archive_manifest_latest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if force or not manifest.exists():
        manifest.write_bytes(body)
        print(f"[manifest] wrote {manifest}")
    else:
        print(f"[manifest] kept {manifest}")
    pulled: list[str] = []
    if outputs:
        by_video: dict[str, list[tuple[str, str, int]]] = {}
        for key, size in _remote_sizes(s3, OUTPUTS_PREFIX + "/").items():
            _year, match_dir, name = key[len(OUTPUTS_PREFIX) + 1:].split("/", 2)
            if name not in SKIP:
                by_video.setdefault(match_dir.rsplit("__", 1)[1], []).append((key, name, size))
        unknown = [v for v in (videos or []) if v not in by_video]
        if unknown:
            raise SystemExit(f"no vision outputs in the Space for: {unknown}")
        for v in videos or sorted(by_video):
            jobs += [(key, OUTPUTS / v / name, size) for key, name, size in by_video[v]]
            pulled.append(v)

    def one(job):
        key, dst, size = job
        return _download(s3, key, dst, size, force)

    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(one, jobs))
    got = sum(r == "downloaded" for r in results)
    print(f"[pull] {len(jobs)} files: {got} downloaded, {len(jobs) - got} kept")
    if pulled:
        marker = OUTPUTS / "batch" / "prior_stored.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        stored = set(json.loads(marker.read_text())) if marker.exists() else set()
        marker.write_text(json.dumps(sorted(stored | set(pulled))))
        print(f"[pull] {len(pulled)} matches with vision outputs; recorded as stored in {marker.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("push-state", help="upload the Jev models, parameters and ledger to the Space")
    p = sub.add_parser("pull", help="download state, the manifest and vision outputs")
    p.add_argument("--videos", nargs="*", help="video ids (default: every match in data/batch_matches.csv)")
    p.add_argument("--no-outputs", action="store_true", help="only the models, parameters, ledger and manifest")
    p.add_argument("--force", action="store_true", help="overwrite local files")
    a = ap.parse_args()
    s3 = s3_client()
    if a.cmd == "push-state":
        push_state(s3)
    else:
        pull(s3, a.videos, not a.no_outputs, a.force)


if __name__ == "__main__":
    main()
