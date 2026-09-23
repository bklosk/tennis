"""Batch runs over many matches: prep where the videos are, then a runner on a GPU machine.

    uv run python -m tennis_pipeline.batch prep --manifest usopen --pack   # the Mac, with the videos
    uv run python -m tennis_pipeline.batch plan --manifest usopen          # hours, upload size, cost
    uv run python -m tennis_pipeline.batch run  --manifest usopen          # on the GPU machine
    uv run python -m tennis_pipeline.batch status --manifest usopen

`cloud batch` runs prep and plan locally, then runs `batch run` on a DigitalOcean droplet while
it streams the videos up (see docs/batch-run.md).

Prep takes everything that needs neither the GPU nor the droplet off the droplet: scene
classification (so only main-camera footage has to be uploaded), audio onsets (packs carry no
audio), and the check for official point data (matches without it need OCR of the full video).

The runner keeps the GPU on tracking, the only GPU-bound stage:
  * one GPU worker process loads (and compiles) the tracking models once and tracks matches
    back to back;
  * low-priority CPU worker processes run each tracked match's remaining stages (OCR, events,
    crops, align, strokes, report) while the GPU tracks the next match;
  * a scenes worker handles matches that arrive without prep.
Stage results go to outputs/VIDEO_ID/batch_status.json, so a rerun resumes where the last one
stopped (failed stages are retried once per run) and one bad match does not stop the batch.
Progress goes to outputs/batch_progress.json; the summary to outputs/batch_summary.json and
outputs/batch_metrics.csv.
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from .paths import CACHE, DATA, DOWNLOADS, OUTPUTS

POST_STAGES = ("ocr", "events", "crops", "align", "strokes", "report")
STAGES = ("scenes", "track", *POST_STAGES)
DEPENDS = {"ocr": "track", "events": "track", "crops": "events", "align": "events", "strokes": "align",
           "report": "strokes"}
FINISHED = {"done", "skipped"}
PRESETS = {
    "usopen": lambda r: r["tournament"] == "US Open" and r["match_status"] == "verified_official_full_match",
}
UPLOADS_DONE = "UPLOADS_DONE"
FPS = 30.0


def load_manifest(spec: str | None = None, ids: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Video ids from explicit ids, a preset name (see PRESETS) or a file with one id per line."""
    out = list(ids)
    if spec in PRESETS:
        with open(DATA / "videos.csv") as fh:
            out += [r["video_id"] for r in csv.DictReader(fh) if PRESETS[spec](r)]
    elif spec:
        out += [ln.split("#")[0].strip() for ln in Path(spec).read_text().splitlines()]
    seen = set()
    return [v for v in out if v and not (v in seen or seen.add(v))]


def video_path(vid: str) -> Path:
    from .cli import video_path as resolve

    return resolve(vid)


# ---------------------------------------------------------------- per-match status

def status_path(vid: str) -> Path:
    return OUTPUTS / vid / "batch_status.json"


def load_status(vid: str) -> dict:
    path = status_path(vid)
    return json.loads(path.read_text()) if path.exists() else {}


def record(vid: str, stage: str, state: str, seconds: float = 0.0, **extra):
    st = load_status(vid)
    st[stage] = {"state": state, "seconds": round(seconds, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    path = status_path(vid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(path)


def stage_state(st: dict, stage: str) -> str | None:
    return st.get(stage, {}).get("state")


def is_complete(vid: str) -> bool:
    st = load_status(vid)
    return all(stage_state(st, s) in FINISHED for s in ("track", *POST_STAGES))


# ---------------------------------------------------------------- stages (run inside workers)

SKIP = object()


def needs_ocr(vid: str) -> bool:
    """No official point-by-point data for this match, so points come from the score graphics."""
    prep = OUTPUTS / vid / "prep.json"
    if prep.exists():
        return bool(json.loads(prep.read_text())["needs_ocr"])
    return _official_missing(vid)


def _official_missing(vid: str) -> bool:
    from .align import sackmann_points

    try:
        sackmann_points(vid)
    except KeyError:
        return True
    return False


def _scenes(vid: str, keyframes: bool = False):
    from . import scenes

    scenes.embed_video(vid, video_path(vid), keyframes_only=keyframes)
    segs = scenes.predict_segments(vid)
    return {"segments": int(len(segs)), "main_camera_s": round(float(segs.duration.sum()), 1)}


def _load_models():
    from .process import TrackModels

    return TrackModels.load()


def _track(vid: str, models=None):
    from .process import track_match

    stats = track_match(vid, video_path(vid), models=models)
    return {k: stats[k] for k in ("frames", "fps", "decode_wait_s")}


def _ocr(vid: str):
    if not needs_ocr(vid):
        return SKIP
    if (OUTPUTS / vid / "ocr_points.csv").exists():  # done during prep (`--ocr-local`)
        return {"cached": True}
    from . import ocr, video

    path = video_path(vid)
    if video.is_pack(path):
        raise ValueError("OCR needs the full video; prep uploads full videos for matches without official data")
    summary = ocr.run(vid, path)
    if summary.get("error"):
        raise RuntimeError(summary["error"])
    return {"points": summary.get("points")}


def _events(vid: str):
    from .process import events_match

    path = video_path(vid)
    hits = events_match(vid, path if path.exists() else None)
    return {"hits": int(len(hits)), "serves": int(hits.is_serve.sum()) if len(hits) else 0}


def _crops(vid: str):
    from .process import crops_match

    return {"hits_with_features": int(len(crops_match(vid, video_path(vid))))}


def _align(vid: str):
    from . import align

    s = align.run(vid)
    return {k: s[k] for k in ("aligned_points", "official_points", "points_source")}


def _strokes(vid: str):
    from . import strokes

    strokes.run([vid], report_path=OUTPUTS / vid / "stroke_report.json")


def _report(vid: str):
    from . import report

    return {"shots": report.match_report(vid)["shots"]}


STAGE_FNS = {"scenes": _scenes, "track": _track, "ocr": _ocr, "events": _events, "crops": _crops,
             "align": _align, "strokes": _strokes, "report": _report, "load_models": _load_models}


def run_stage(vid: str, stage: str, *args) -> bool:
    t = time.time()
    try:
        result = STAGE_FNS[stage](vid, *args)
    except Exception as e:  # one match's failure must not stop the batch
        record(vid, stage, "failed", time.time() - t, error=f"{type(e).__name__}: {e}"[:500],
               trace=traceback.format_exc(limit=6)[-2000:])
        print(f"{vid} {stage} FAILED after {time.time() - t:.0f} s: {type(e).__name__}: {e}", flush=True)
        return False
    state = "skipped" if result is SKIP else "done"
    record(vid, stage, state, time.time() - t, **(result if isinstance(result, dict) else {}))
    print(f"{vid} {stage} {state} in {time.time() - t:.0f} s"
          + (f" {json.dumps(result)}" if isinstance(result, dict) else ""), flush=True)
    return True


_MODELS = None
_MODELS_ERROR = None


def _gpu_init():
    # A failure here is reported by each track task rather than breaking the pool, so the error
    # that ends up in the status (and in the abort message) is the real cause.
    global _MODELS, _MODELS_ERROR
    try:
        _MODELS = STAGE_FNS["load_models"]()
    except Exception as e:
        _MODELS_ERROR = f"{type(e).__name__}: {e}"[:500]


def _cpu_init():
    # Post-processing must not slow tracking: lower priority, and few threads per process.
    os.nice(10)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "2"


def gpu_task(vid: str) -> bool:
    if _MODELS_ERROR:
        record(vid, "track", "failed", error=f"loading the tracking models failed: {_MODELS_ERROR}")
        return False
    return run_stage(vid, "track", _MODELS)


def scenes_task(vid: str, keyframes: bool) -> bool:
    return run_stage(vid, "scenes", keyframes)


def post_task(vid: str) -> dict:
    for stage in POST_STAGES:
        st = load_status(vid)
        if stage_state(st, stage) in FINISHED:
            continue
        dep = DEPENDS[stage]
        if stage_state(st, dep) not in FINISHED:
            record(vid, stage, "blocked", error=f"needs {dep}")
            continue
        run_stage(vid, stage)
    return load_status(vid)


# ---------------------------------------------------------------- runner

class Runner:
    """Schedules the stages of every match in `vids` over GPU, scenes and CPU worker pools.

    `wait_for_videos`: videos are being uploaded; a match's video counts as present only once
    downloads/VIDEO_ID.ready exists, and the batch ends after downloads/UPLOADS_DONE appears.
    `max_track_failures` consecutive tracking failures abort the batch (something systematic,
    such as missing weights, would otherwise fail every match while the GPU bill runs).
    `processes=False` runs the workers as threads (tests).
    """

    def __init__(self, vids: list[str], wait_for_videos: bool = False, cpu_workers: int = 2,
                 delete_videos: bool = False, scene_keyframes: bool = False, processes: bool = True,
                 poll_s: float = 5.0, gpu_tasks_per_worker: int = 25, progress_every_s: float = 300.0,
                 max_track_failures: int = 3, gpu_workers: int = 1):
        self.vids = list(vids)
        self.wait = wait_for_videos
        self.cpu_workers = cpu_workers
        self.gpu_workers = gpu_workers
        self.gpu_util: list[float] = []
        self.delete_videos = delete_videos
        self.scene_keyframes = scene_keyframes
        self.processes = processes
        self.poll_s = poll_s
        self.gpu_tasks_per_worker = gpu_tasks_per_worker
        self.progress_every_s = progress_every_s
        self.max_track_failures = max_track_failures
        self.track_failures: list[str] = []
        self.aborted: str | None = None
        self.inflight = {"gpu": {}, "scenes": {}, "cpu": {}}
        self.gpu_busy_s = self.gpu_wait_video_s = self.gpu_wait_other_s = 0.0
        self.started = time.time()
        self.removed = set()

    # pools --------------------------------------------------------------
    def _pool(self, kind: str) -> Executor:
        size = {"gpu": self.gpu_workers, "cpu": self.cpu_workers}.get(kind, 1)
        if not self.processes:
            return ThreadPoolExecutor(size, initializer={"gpu": _gpu_init}.get(kind))
        ctx = mp.get_context("spawn")  # CUDA cannot be re-initialised in a forked child
        if kind == "gpu":
            return ProcessPoolExecutor(size, mp_context=ctx, initializer=_gpu_init,
                                       max_tasks_per_child=self.gpu_tasks_per_worker)
        if kind == "cpu":
            return ProcessPoolExecutor(self.cpu_workers, mp_context=ctx, initializer=_cpu_init, max_tasks_per_child=4)
        return ProcessPoolExecutor(1, mp_context=ctx, max_tasks_per_child=10)

    # match state --------------------------------------------------------
    def video_ready(self, vid: str) -> bool:
        if not video_path(vid).exists():
            return False
        return not self.wait or (DOWNLOADS / f"{vid}.ready").exists()

    def uploads_done(self) -> bool:
        return not self.wait or (DOWNLOADS / UPLOADS_DONE).exists()

    def state(self, vid: str) -> str:
        for kind, label in (("gpu", "tracking"), ("scenes", "scenes"), ("cpu", "post")):
            if vid in self.inflight[kind]:
                return label
        st = load_status(vid)
        if all(stage_state(st, s) in FINISHED for s in ("track", *POST_STAGES)):
            return "done"
        if stage_state(st, "track") == "failed" or stage_state(st, "scenes") == "failed":
            return "failed"
        if stage_state(st, "track") in FINISHED:
            return "failed" if all(stage_state(st, s) is not None for s in POST_STAGES) else "tracked"
        if not self.video_ready(vid):
            return "waiting_video"
        if not (OUTPUTS / vid / "segments.csv").exists():
            return "needs_scenes"
        return "ready"

    def _reset_failures(self):
        """Failed and blocked stages from an earlier run get another try."""
        for vid in self.vids:
            st = load_status(vid)
            stale = [s for s, v in st.items() if v.get("state") in ("failed", "blocked")]
            if stale:
                for s in stale:
                    st.pop(s)
                status_path(vid).write_text(json.dumps(st, indent=1))

    # main loop ----------------------------------------------------------
    def run(self) -> dict:
        self._reset_failures()
        pools = {kind: self._pool(kind) for kind in self.inflight}
        last_tick = last_progress = time.time()
        try:
            while True:
                self._collect(pools)
                states = {v: self.state(v) for v in self.vids}
                self._schedule(pools, states)
                states = {v: self.state(v) for v in self.vids}
                now = time.time()
                self._account(states, now - last_tick)
                last_tick = now
                self._cleanup(states)
                self.write_progress(states)
                if now - last_progress >= self.progress_every_s:
                    print(self.progress_line(states), flush=True)
                    last_progress = now
                if self.aborted or self._finished(states):
                    break
                time.sleep(self.poll_s)
        finally:
            for pool in pools.values():
                pool.shutdown(wait=True, cancel_futures=True)
        if not self.aborted:
            for v, s in {v: self.state(v) for v in self.vids}.items():
                if s == "waiting_video":
                    record(v, "track", "failed", error="video was never uploaded" if self.wait else "video not found")
        self.write_progress({v: self.state(v) for v in self.vids})
        summary = summarize(self.vids)
        if self.aborted:
            summary["aborted"] = self.aborted
        return summary

    def _collect(self, pools: dict):
        for kind, jobs in self.inflight.items():
            for vid, fut in list(jobs.items()):
                if not fut.done():
                    continue
                del jobs[vid]
                err = fut.exception()
                stage = {"gpu": "track", "scenes": "scenes", "cpu": "post"}[kind]
                if err is not None:
                    record(vid, stage, "failed", error=f"worker died: {type(err).__name__}: {err}"[:500])
                    print(f"{vid} {stage} worker died: {err!r}", flush=True)
                    if isinstance(err, BrokenProcessPool):
                        pools[kind].shutdown(wait=False, cancel_futures=True)
                        pools[kind] = self._pool(kind)
                        for other in list(jobs):  # everything else on the broken pool is lost too
                            record(other, stage, "failed", error="worker pool crashed")
                            del jobs[other]
                if kind == "gpu":
                    self._track_outcome(vid)

    def _track_outcome(self, vid: str):
        track = load_status(vid).get("track", {})
        if track.get("state") != "failed":
            self.track_failures = []
            return
        self.track_failures.append(f"{vid}: {track.get('error', '')}")
        if len(self.track_failures) >= self.max_track_failures:
            self.aborted = (f"aborting: {len(self.track_failures)} matches in a row failed tracking; "
                            f"last: {self.track_failures[-1]}")
            print(self.aborted, flush=True)

    def _schedule(self, pools: dict, states: dict):
        for v in self.vids:
            if len(self.inflight["gpu"]) >= self.gpu_workers:
                break
            if states[v] == "ready" and v not in self.inflight["gpu"]:
                self.inflight["gpu"][v] = pools["gpu"].submit(gpu_task, v)
                states[v] = "tracking"
        if not self.inflight["scenes"]:
            nxt = next((v for v in self.vids if states[v] == "needs_scenes"), None)
            if nxt:
                self.inflight["scenes"][nxt] = pools["scenes"].submit(scenes_task, nxt, self.scene_keyframes)
        for v in self.vids:
            if len(self.inflight["cpu"]) >= self.cpu_workers:
                break
            if states[v] == "tracked" and v not in self.inflight["cpu"]:
                self.inflight["cpu"][v] = pools["cpu"].submit(post_task, v)

    def _account(self, states: dict, dt: float):
        if self.inflight["gpu"]:
            self.gpu_busy_s += dt
            util = gpu_utilization()
            if util is not None:
                self.gpu_util.append(util)
        elif any(s in ("ready", "needs_scenes", "scenes") for s in states.values()):
            self.gpu_wait_other_s += dt
        elif any(s == "waiting_video" for s in states.values()):
            self.gpu_wait_video_s += dt

    def _cleanup(self, states: dict):
        if not self.delete_videos:
            return
        for v, s in states.items():
            if s in ("done", "failed") and v not in self.removed:
                remove_video(v)
                self.removed.add(v)

    def _finished(self, states: dict) -> bool:
        if any(self.inflight.values()):
            return False
        runnable = {"ready", "needs_scenes", "tracked"}
        if any(s in runnable for s in states.values()):
            return False
        return all(s in ("done", "failed") for s in states.values()) or self.uploads_done()

    # progress -----------------------------------------------------------
    def progress(self, states: dict) -> dict:
        frames = track_s = 0
        remaining_s = 0.0
        for v in self.vids:
            st = load_status(v)
            tr = st.get("track", {})
            if tr.get("state") == "done":
                frames += tr.get("frames", 0)
                track_s += tr.get("seconds", 0.0)
            elif states[v] not in ("done", "failed"):
                remaining_s += _main_camera_s(v)
        fps = frames / track_s if track_s and frames else None
        return {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_s": round(time.time() - self.started),
                "matches": len(self.vids), "counts": dict(Counter(states.values())), "states": states,
                "frames_tracked": int(frames), "tracking_fps": round(fps, 1) if fps else None,
                "gpu_busy_s": round(self.gpu_busy_s), "gpu_waiting_for_video_s": round(self.gpu_wait_video_s),
                "gpu_util_while_tracking": round(sum(self.gpu_util[-720:]) / len(self.gpu_util[-720:]), 1)
                if self.gpu_util else None,
                "gpu_waiting_for_scenes_s": round(self.gpu_wait_other_s),
                "eta_tracking_s": round(remaining_s * FPS / fps) if fps else None,
                "failures": {v: failures(load_status(v)) for v, s in states.items() if s == "failed"}}

    def write_progress(self, states: dict):
        path = OUTPUTS / "batch_progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.progress(states), indent=1))
        tmp.replace(path)

    def progress_line(self, states: dict) -> str:
        p = self.progress(states)
        eta = f", tracking ETA {p['eta_tracking_s'] / 3600:.1f} h" if p["eta_tracking_s"] else ""
        util = f", {p['gpu_util_while_tracking']:.0f}% utilised" if p["gpu_util_while_tracking"] is not None else ""
        return (f"[batch {p['elapsed_s'] / 3600:.1f} h] {json.dumps(p['counts'])} | {p['tracking_fps']} fps | "
                f"GPU busy {p['gpu_busy_s'] / 60:.0f} min{util}, waiting for video "
                f"{p['gpu_waiting_for_video_s'] / 60:.0f} min{eta}")


def gpu_utilization() -> float | None:
    """Mean utilisation (%) across NVIDIA GPUs, or None without nvidia-smi."""
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        vals = [float(x) for x in out.split()]
    except (subprocess.SubprocessError, ValueError):
        return None
    return sum(vals) / len(vals) if vals else None


def failures(st: dict) -> dict:
    return {s: v.get("error", v["state"]) for s, v in st.items() if v.get("state") in ("failed", "blocked")}


def remove_video(vid: str):
    import shutil

    for name in (f"{vid}.mp4", f"{vid}.f298.mp4", f"{vid}.ready"):
        (DOWNLOADS / name).unlink(missing_ok=True)
    shutil.rmtree(DOWNLOADS / f"{vid}.pack", ignore_errors=True)


def _main_camera_s(vid: str) -> float:
    prep = OUTPUTS / vid / "prep.json"
    if prep.exists():
        return float(json.loads(prep.read_text())["main_camera_s"])
    segs = OUTPUTS / vid / "segments.csv"
    if segs.exists():
        import pandas as pd

        return float(pd.read_csv(segs).duration.sum())
    return 0.0


def summarize(vids: list[str]) -> dict:
    """batch_metrics.csv (one row per match) and batch_summary.json (totals, failures, serve speed)."""
    import pandas as pd

    from .metrics import match_metrics

    rows = []
    for v in vids:
        st = load_status(v)
        row = {"video_id": v, "complete": is_complete(v), "failed_stages": json.dumps(failures(st)) if failures(st) else ""}
        if stage_state(st, "report") == "done":
            try:
                row.update(match_metrics(v))
            except Exception as e:  # metrics are a summary; a gap must not hide the other matches
                row["metrics_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    df = pd.DataFrame(rows)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUTS / "batch_metrics.csv", index=False)
    summary = {"matches": len(vids), "complete": int(df.complete.sum()),
               "failed": {r["video_id"]: json.loads(r["failed_stages"]) for r in rows if r["failed_stages"]}}
    if "aligned_points" in df:
        ok = df[df.aligned_points.notna()]
        w = ok.aligned_points
        summary.update({
            "official_points": int(ok.official_points.sum()), "aligned_points": int(w.sum()),
            "serve_detected_rate": round(float((ok.serve_detected_rate * w).sum() / w.sum()), 3) if w.sum() else None,
            "rally_within_1": round(float((ok.rally_within_1.fillna(0) * w).sum() / w.sum()), 3) if w.sum() else None,
            "main_camera_h_tracked": round(float(ok.main_camera_min_processed.sum()) / 60, 1),
        })
        done = [v for v in ok.video_id if (OUTPUTS / v / "shots.csv").exists()]
        if done:
            from .report import serve_speed_plot

            try:
                summary["serve_speed"] = serve_speed_plot(done)
            except Exception as e:
                summary["serve_speed"] = {"error": f"{type(e).__name__}: {e}"}
    (OUTPUTS / "batch_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ---------------------------------------------------------------- prep and plan (machine with the videos)

def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def prep(vids: list[str], pack: bool = False, keyframes: bool = False, retrain: str | None = None,
         ocr_local: bool = False) -> list[dict]:
    """Scenes, segments, audio onsets and the official-data check for every local video; with
    `pack`, main-camera packs for matches that have official data (or, with `ocr_local`, whose
    score graphics were read here). Cached per match."""
    from . import scenes, video

    local = [v for v in vids if video_path(v).exists() and not video.is_pack(video_path(v))]
    for k, v in enumerate(local):
        if not (OUTPUTS / v / "scene_embeddings.npz").exists():
            t = time.time()
            scenes.embed_video(v, video_path(v), keyframes_only=keyframes)
            print(f"[{k + 1}/{len(local)}] {v} scene embeddings in {time.time() - t:.0f} s", flush=True)
    resegment = False
    if retrain or not scenes.CLASSIFIER_PATH.exists():
        print("scene classifier:", json.dumps(scenes.train_classifier(local, retrain or "auto")), flush=True)
        resegment = True
    recs = []
    for v in vids:
        if v not in local:
            rec = {"video_id": v, "error": "video not found"}
        else:
            try:
                rec = prep_match(v, pack=pack, resegment=resegment, ocr_local=ocr_local)
            except Exception as e:  # e.g. the official-data lookup is offline; the rest still preps
                rec = {"video_id": v, "error": f"{type(e).__name__}: {e}"[:300]}
        recs.append(rec)
        print(json.dumps(rec), flush=True)
    return recs


def prep_match(vid: str, pack: bool = False, resegment: bool = False, ocr_local: bool = False) -> dict:
    from . import audio, scenes, video

    out_dir = OUTPUTS / vid
    src = video_path(vid)
    segs_path = out_dir / "segments.csv"
    if resegment or not segs_path.exists():
        scenes.predict_segments(vid)
    if resegment or not (out_dir / "scene_prob.npz").exists():
        scenes.save_scene_prob(vid)
    import pandas as pd

    segs = pd.read_csv(segs_path)
    audio.onsets(vid, src)
    prev = json.loads((out_dir / "prep.json").read_text()) if (out_dir / "prep.json").exists() else {}
    duration = prev.get("duration_s") or video.probe(src)["duration"]
    rec = {"video_id": vid, "duration_s": round(duration, 1), "segments": int(len(segs)),
           "main_camera_s": round(float(segs.duration.sum()), 1),
           "needs_ocr": prev["needs_ocr"] if "needs_ocr" in prev else _official_missing(vid),
           "segments_digest": _digest(segs_path), "full_bytes": src.stat().st_size}
    rec["main_camera_frac"] = round(rec["main_camera_s"] / max(duration, 1.0), 3)
    rec["check_scenes"] = not 0.2 <= rec["main_camera_frac"] <= 0.8
    ocr_done = (out_dir / "ocr_points.csv").exists()
    if rec["needs_ocr"] and ocr_local and not ocr_done:
        from . import ocr

        summary = ocr.run(vid, src)
        rec["ocr_error"] = summary.get("error")
        ocr_done = (out_dir / "ocr_points.csv").exists()
    pack_dir = DOWNLOADS / f"{vid}.pack"
    if pack and (not rec["needs_ocr"] or ocr_done):
        fresh = prev.get("upload") == "pack" and prev.get("segments_digest") == rec["segments_digest"] \
            and video.is_pack(pack_dir)
        index = json.loads((pack_dir / video.PACK_INDEX).read_text()) if fresh else \
            video.pack_segments(src, list(zip(segs.start, segs.end)), pack_dir)
        rec.update(upload="pack", upload_bytes=int(index["bytes"]))
    else:
        rec.update(upload="full", upload_bytes=rec["full_bytes"])
    (out_dir / "prep.json").write_text(json.dumps(rec, indent=1))
    return rec


def plan(vids: list[str], fps: float = 190.0, price_hourly: float = 1.57, overhead_h: float = 0.5) -> dict:
    """Upload size, GPU hours and cost for the prepped matches in `vids` that are not complete."""
    recs, missing, complete = [], [], []
    for v in vids:
        path = OUTPUTS / v / "prep.json"
        if is_complete(v):
            complete.append(v)
        elif path.exists():
            recs.append(json.loads(path.read_text()))
        else:
            missing.append(v)
    main_s = sum(r["main_camera_s"] for r in recs)
    gpu_h = main_s * FPS / fps / 3600
    upload = sum(r["upload_bytes"] for r in recs)
    return {"matches": len(recs), "complete": len(complete), "not_prepped": missing,
            "video_h": round(sum(r["duration_s"] for r in recs) / 3600, 1), "main_camera_h": round(main_s / 3600, 1),
            "needs_ocr": sum(r["needs_ocr"] for r in recs), "packed": sum(r["upload"] == "pack" for r in recs),
            "check_scenes": [r["video_id"] for r in recs if r.get("check_scenes")],
            "upload_gb": round(upload / 1e9, 1), "full_gb": round(sum(r["full_bytes"] for r in recs) / 1e9, 1),
            "assumed_fps": fps, "gpu_h": round(gpu_h, 1), "price_hourly": price_hourly,
            "cost_usd": round((gpu_h + overhead_h) * price_hourly, 2) if recs else 0.0,
            "upload_mbps_to_keep_gpu_busy": round(upload * 8 / max(gpu_h * 3600, 1) / 1e6, 1)}


def format_plan(p: dict) -> str:
    lines = [f"{p['matches']} matches to run ({p['complete']} already complete, {len(p['not_prepped'])} not prepped)",
             f"  video {p['video_h']} h, main camera {p['main_camera_h']} h; {p['needs_ocr']} without official data "
             f"(score-graphic OCR); {p['packed']} uploaded as main-camera packs",
             f"  upload {p['upload_gb']} GB (full videos would be {p['full_gb']} GB); keeping the GPU busy needs "
             f"{p['upload_mbps_to_keep_gpu_busy']} Mbps upstream",
             f"  tracking {p['gpu_h']} GPU h at {p['assumed_fps']:.0f} fps -> about ${p['cost_usd']} at "
             f"${p['price_hourly']}/h"]
    if p["check_scenes"]:
        lines.append(f"  check the scene filter on {len(p['check_scenes'])} matches whose main-camera share is "
                     f"outside 20-80%: {' '.join(p['check_scenes'][:12])}")
    if p["not_prepped"]:
        lines.append(f"  not prepped (no local video?): {' '.join(p['not_prepped'][:12])}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["prep", "plan", "run", "status", "summary"])
    ap.add_argument("video_ids", nargs="*")
    ap.add_argument("--manifest", help=f"file with one video id per line, or a preset: {', '.join(PRESETS)}")
    ap.add_argument("--pack", action="store_true", help="prep: main-camera packs for matches with official data")
    ap.add_argument("--ocr-local", action="store_true",
                    help="prep: read score graphics here for matches without official data, so they can be packed too")
    ap.add_argument("--scene-keyframes", action="store_true", help="prep/run: keyframe-only scene decoding")
    ap.add_argument("--retrain-scenes", choices=["auto", "vlm", "court"],
                    help="prep: retrain the scene classifier on all these matches with this labeler")
    ap.add_argument("--fps", type=float, default=190.0, help="plan: assumed end-to-end tracking fps")
    ap.add_argument("--price", type=float, default=1.57, help="plan: droplet $/h")
    ap.add_argument("--wait-for-videos", action="store_true", help="run: videos are still being uploaded")
    ap.add_argument("--delete-videos", action="store_true", help="run: delete each video once its match is finished")
    ap.add_argument("--cpu-workers", type=int, default=2)
    ap.add_argument("--gpu-workers", type=int, default=1,
                    help="run: matches tracked at once (try 2 if GPU utilisation stays low with CPUs to spare)")
    args = ap.parse_args()
    vids = load_manifest(args.manifest, args.video_ids)
    if not vids:
        ap.error("no video ids (pass ids or --manifest)")
    if args.action == "prep":
        prep(vids, pack=args.pack, keyframes=args.scene_keyframes, retrain=args.retrain_scenes, ocr_local=args.ocr_local)
        print(format_plan(plan(vids, args.fps, args.price)))
    elif args.action == "plan":
        print(format_plan(plan(vids, args.fps, args.price)))
    elif args.action == "run":
        runner = Runner(vids, wait_for_videos=args.wait_for_videos, cpu_workers=args.cpu_workers,
                        delete_videos=args.delete_videos, scene_keyframes=args.scene_keyframes,
                        gpu_workers=args.gpu_workers)
        summary = runner.run()
        print(json.dumps(summary, indent=2, default=str))
        if summary.get("aborted"):
            raise SystemExit(1)
    elif args.action == "status":
        path = OUTPUTS / "batch_progress.json"
        counts = Counter("complete" if is_complete(v) else ("failed" if failures(load_status(v)) else "pending")
                         for v in vids)
        print(json.dumps({"matches": dict(counts), "progress": json.loads(path.read_text()) if path.exists() else None},
                         indent=2))
    elif args.action == "summary":
        print(json.dumps(summarize(vids), indent=2, default=str))


if __name__ == "__main__":
    main()
