import json
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import pytest

from tennis_pipeline import batch, cli, video


def _video(path, seconds=24, offset=1.4):
    """720p 59.94 fps H.264 with B-frames, irregular keyframes and a non-zero start time."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=60000/1001:d={seconds}",
                    "-c:v", "libx264", "-preset", "veryfast", "-g", "150", "-keyint_min", "30", "-bf", "3",
                    "-pix_fmt", "yuv420p", "-output_ts_offset", str(offset), str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return _video(tmp_path_factory.mktemp("v") / "src.mp4")


def test_pack_decodes_the_same_frames_as_the_source(src, tmp_path):
    pack = tmp_path / "m.pack"
    index = video.pack_segments(src, [(3.1, 6.0), (15.2, 19.5)], pack, merge_gap=4.0)
    assert len(index["parts"]) == 2 and index["bytes"] < index["source_bytes"]
    assert video.is_pack(pack) and video.probe(pack)["duration"] == pytest.approx(video.probe(src)["duration"])
    native = 60000 / 1001
    for start in (3.1, 4.73, 15.2, 18.9):
        a, b = video.read_clip(src, start, 0.5, fps=native), video.read_clip(pack, start, 0.5, fps=native)
        assert len(a) == len(b) > 0
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
    a, b = video.read_clip(src, 16.0, 2.0), video.read_clip(pack, 16.0, 2.0)
    assert len(a) == len(b) == 60
    # At 30 fps the resampler may take the neighbouring source frame in a stretch, never further.
    full = video.read_clip(src, 15.9, 2.3, fps=native)
    idx = lambda f: int(np.argmin([np.abs(f.astype(int) - g.astype(int)).mean() for g in full]))  # noqa: E731
    assert all(abs(idx(x) - idx(y)) <= 1 for x, y in zip(a[::6], b[::6]))
    assert len(video.read_clip(pack, 10.0, 1.0)) == 0  # outside every part
    with pytest.raises(ValueError):
        next(video.iter_frames(pack, 2, (640, 360)))


def test_video_path_falls_back_to_a_pack(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DOWNLOADS", tmp_path)
    (tmp_path / "abc.pack").mkdir()
    (tmp_path / "abc.pack" / video.PACK_INDEX).write_text("{}")
    assert cli.video_path("abc") == tmp_path / "abc.pack"
    (tmp_path / "abc.mp4").write_bytes(b"")
    assert cli.video_path("abc") == tmp_path / "abc.mp4"
    assert cli.video_path("missing") == tmp_path / "missing.mp4"


def test_manifest_preset_and_file(tmp_path):
    ids = batch.load_manifest("usopen")
    assert len(ids) == len(set(ids)) >= 150
    assert {"KCcKkUnjbzA", "Fl33UXv6jKI", "Ce3dRYHWIBI"} <= set(ids)
    f = tmp_path / "m.txt"
    f.write_text("a1  # first\n\n# comment\nb2\na1\n")
    assert batch.load_manifest(str(f), ["z9"]) == ["z9", "a1", "b2"]


# ---------------------------------------------------------------- runner with stand-in stages

@pytest.fixture
def dirs(tmp_path, monkeypatch):
    out, down = tmp_path / "outputs", tmp_path / "downloads"
    out.mkdir()
    down.mkdir()
    monkeypatch.setattr(batch, "OUTPUTS", out)
    monkeypatch.setattr(batch, "DOWNLOADS", down)
    monkeypatch.setattr(batch, "video_path", lambda v: down / f"{v}.mp4")
    return out, down


class Stages:
    """Records calls; `fail` maps (video, stage) to the number of times that stage should fail."""

    def __init__(self, out, fail=None):
        self.out, self.fail, self.calls, self.models = out, dict(fail or {}), [], 0
        self.lock = threading.Lock()

    def install(self, monkeypatch):
        fns = {s: self._stage(s) for s in batch.STAGES}
        fns["load_models"] = self._load
        monkeypatch.setattr(batch, "STAGE_FNS", fns)

    def _load(self):
        with self.lock:
            self.models += 1
        return "models"

    def _stage(self, name):
        def fn(vid, *args):
            with self.lock:
                self.calls.append((vid, name))
                left = self.fail.get((vid, name), 0)
                if left:
                    self.fail[(vid, name)] = left - 1
            if left:
                raise RuntimeError(f"boom {name}")
            if name == "scenes":
                (self.out / vid).mkdir(exist_ok=True)
                pd.DataFrame({"segment_id": [0], "start": [0.0], "end": [30.0], "duration": [30.0]}).to_csv(
                    self.out / vid / "segments.csv", index=False)
            if name == "track":
                assert args == ("models",)
                return {"frames": 900, "fps": 100.0}
            return batch.SKIP if name == "ocr" else None
        return fn


def _prepped(out, down, vids, video=True):
    for v in vids:
        (out / v).mkdir(exist_ok=True)
        pd.DataFrame({"segment_id": [0], "start": [0.0], "end": [30.0], "duration": [30.0]}).to_csv(
            out / v / "segments.csv", index=False)
        if video:
            (down / f"{v}.mp4").write_bytes(b"x")


def test_runner_runs_every_stage_and_isolates_a_failure(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out, fail={("m2", "events"): 5})
    stages.install(monkeypatch)
    _prepped(out, down, ["m1", "m2", "m3"])
    summary = batch.Runner(["m1", "m2", "m3"], processes=False, poll_s=0.01).run()
    assert stages.models == 1  # models are loaded once for the whole batch
    for v in ("m1", "m3"):
        assert batch.is_complete(v)
        assert [s for vv, s in stages.calls if vv == v] == ["track", *batch.POST_STAGES]
    st = batch.load_status("m2")
    assert st["events"]["state"] == "failed" and "boom events" in st["events"]["error"]
    assert st["crops"]["state"] == st["align"]["state"] == "blocked" and st["ocr"]["state"] == "skipped"
    assert summary["complete"] == 2 and set(summary["failed"]) == {"m2"}
    progress = json.loads((out / "batch_progress.json").read_text())
    assert progress["counts"] == {"done": 2, "failed": 1} and progress["frames_tracked"] == 2700
    assert (out / "batch_metrics.csv").exists()


def test_rerun_resumes_and_retries_only_what_failed(dirs, monkeypatch):
    out, down = dirs
    _prepped(out, down, ["m1", "m2"])
    first = Stages(out, fail={("m2", "align"): 1})
    first.install(monkeypatch)
    batch.Runner(["m1", "m2"], processes=False, poll_s=0.01).run()
    assert not batch.is_complete("m2")
    second = Stages(out)
    second.install(monkeypatch)
    batch.Runner(["m1", "m2"], processes=False, poll_s=0.01).run()
    assert batch.is_complete("m1") and batch.is_complete("m2")
    assert second.calls == [("m2", "align"), ("m2", "strokes"), ("m2", "report")]


def test_runner_waits_for_uploaded_videos_and_deletes_finished_ones(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out)
    stages.install(monkeypatch)
    _prepped(out, down, ["m1", "m2", "never"], video=False)
    runner = batch.Runner(["m1", "m2", "never"], wait_for_videos=True, delete_videos=True, processes=False,
                          poll_s=0.01)
    result = {}
    th = threading.Thread(target=lambda: result.update(runner.run()))
    th.start()
    time.sleep(0.2)
    assert stages.calls == []  # no .ready marker yet
    (down / "m1.mp4").write_bytes(b"x")
    time.sleep(0.2)
    assert stages.calls == []  # a video without its marker is still uploading
    (down / "m1.ready").touch()
    (down / "m2.mp4").write_bytes(b"x")
    (down / "m2.ready").touch()
    time.sleep(0.5)
    assert th.is_alive()  # uploads not declared finished
    (down / batch.UPLOADS_DONE).touch()
    th.join(timeout=10)
    assert not th.is_alive()
    assert batch.is_complete("m1") and batch.is_complete("m2")
    assert batch.load_status("never")["track"]["error"] == "video was never uploaded"
    assert not (down / "m1.mp4").exists() and not (down / "m1.ready").exists()
    assert result["complete"] == 2


def test_scenes_run_first_when_a_match_arrives_without_prep(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out)
    stages.install(monkeypatch)
    (down / "raw.mp4").write_bytes(b"x")
    batch.Runner(["raw"], processes=False, poll_s=0.01).run()
    assert [s for _, s in stages.calls][:2] == ["scenes", "track"] and batch.is_complete("raw")


def test_worker_crash_is_recorded_and_the_batch_continues(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out)
    stages.install(monkeypatch)
    _prepped(out, down, ["m1", "m2"])
    real = batch.gpu_task

    def crashing(vid):
        if vid == "m1":
            raise MemoryError("killed")
        return real(vid)

    monkeypatch.setattr(batch, "gpu_task", crashing)
    summary = batch.Runner(["m1", "m2"], processes=False, poll_s=0.01).run()
    assert "worker died: MemoryError" in batch.load_status("m1")["track"]["error"]
    assert summary["complete"] == 1


def test_two_gpu_workers_track_different_matches(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out)
    stages.install(monkeypatch)
    vids = [f"m{i}" for i in range(4)]
    _prepped(out, down, vids)
    summary = batch.Runner(vids, processes=False, poll_s=0.01, gpu_workers=2).run()
    assert summary["complete"] == 4 and 1 <= stages.models <= 2
    assert sorted(v for v, s in stages.calls if s == "track") == vids


def test_systematic_tracking_failure_aborts_the_batch(dirs, monkeypatch):
    out, down = dirs
    stages = Stages(out)
    stages.install(monkeypatch)
    monkeypatch.setitem(batch.STAGE_FNS, "load_models", lambda: (_ for _ in ()).throw(FileNotFoundError("tracknet.pt")))
    monkeypatch.setattr(batch, "_MODELS_ERROR", None)
    vids = [f"m{i}" for i in range(5)]
    _prepped(out, down, vids)
    summary = batch.Runner(vids, processes=False, poll_s=0.01, max_track_failures=3).run()
    assert "3 matches in a row failed tracking" in summary["aborted"]
    assert "FileNotFoundError: tracknet.pt" in batch.load_status("m0")["track"]["error"]
    assert "track" not in batch.load_status("m4")  # never attempted
    assert stages.calls == []


def test_post_stages_run_for_real_on_synthetic_tracks(tmp_path, monkeypatch):
    """events -> crops -> align -> strokes -> report -> summary, with the real stage code."""
    from test_serve import CONTACT, N, make_tracks

    from tennis_pipeline import align, metrics, paths, process, report, strokes

    out, down = tmp_path / "outputs", tmp_path / "downloads"
    for mod in (paths, batch, metrics, report):
        monkeypatch.setattr(mod, "OUTPUTS", out)
    monkeypatch.setattr(batch, "DOWNLOADS", down)
    monkeypatch.setattr(batch, "video_path", lambda v: down / f"{v}.mp4")  # no video: as after deletion
    monkeypatch.setattr(report, "video_path", lambda v: down / f"{v}.mp4")
    monkeypatch.setattr(process, "CROP_DIR", tmp_path / "crops")
    monkeypatch.setattr(strokes, "CACHE", tmp_path)
    monkeypatch.setattr(align, "handedness", lambda: {})
    vid, t0s = "syn1", [100.0, 140.0, 180.0, 220.0]
    d = paths.match_dir(vid)
    (d / "tracks").mkdir()
    for k, t0 in enumerate(t0s):
        tr, b = make_tracks()
        tr.t0, tr.ball, tr.court_ok = t0, b, 1.0
        tr.player_kps = {s: np.full((N, 17, 3), np.nan) for s in tr.players}
        process.save_tracks(d / "tracks" / f"{k:04d}_{int(t0 * 10):06d}.npz", tr)
    pd.DataFrame({"segment_id": range(4), "start": t0s, "end": [t + 10 for t in t0s], "duration": 10.0}).to_csv(
        d / "segments.csv", index=False)
    np.savez_compressed(d / "scene_prob.npz", prob=np.ones(600, np.float32))
    np.savez_compressed(d / "audio_onsets.npz", t=np.array([t + CONTACT / 30 for t in t0s]), strength=np.ones(4))
    (d / "prep.json").write_text(json.dumps({"duration_s": 300.0, "main_camera_s": 40.0, "needs_ocr": False}))
    (d / "track_stats.json").write_text(json.dumps({"fps": 190.0}))
    # Four aces by player 1, serving from the near end, 40 s apart.
    matches = pd.DataFrame({"match_id": ["m"], "player1": ["Ann Near"], "player2": ["Bea Far"]})
    points = pd.DataFrame({"match_id": "m", "ElapsedTime": ["0:00:00", "0:00:40", "0:01:20", "0:02:00"],
                           "SetNo": 1, "GameNo": 1, "PointNumber": [1, 2, 3, 4], "PointWinner": 1, "PointServer": 1,
                           "Speed_KMH": 180, "RallyCount": 1, "ServeNumber": 1, "P1Score": [15, 30, 40, 0],
                           "P2Score": 0})
    files = {"matches": tmp_path / "m.csv", "points": tmp_path / "p.csv"}
    matches.to_csv(files["matches"], index=False)
    points.to_csv(files["points"], index=False)
    monkeypatch.setattr(align, "_fetch", lambda rel: files["matches" if rel.endswith("matches.csv") else "points"])
    monkeypatch.setattr(align, "video_match", lambda v: {"year": "2024", "tournament": "US Open",
                                                         "player1": "Ann Near", "player2": "Bea Far"})
    batch.record(vid, "track", "done", frames=1200, fps=190.0)
    st = batch.post_task(vid)
    assert {s: st[s]["state"] for s in batch.POST_STAGES} == {
        "ocr": "skipped", "events": "done", "crops": "done", "align": "done", "strokes": "done", "report": "done"}
    assert st["events"]["serves"] == 4 and st["align"]["aligned_points"] == 4
    shots = pd.read_csv(d / "shots.csv")
    assert len(shots) == 4 and set(shots.player) == {"Ann Near"} and shots.is_serve.all()
    assert (d / "stroke_report.json").exists() and (d / "court_maps.png").exists()
    summary = batch.summarize([vid])
    assert summary["complete"] == 1 and summary["aligned_points"] == 4 and summary["serve_detected_rate"] == 1.0
    row = pd.read_csv(out / "batch_metrics.csv").iloc[0]
    assert row.video_min == 5.0 and row.rally_within_1 == 1.0


def test_plan_estimates_upload_gpu_hours_and_cost(dirs):
    out, _ = dirs
    for v, main_s, up, ocr in (("a", 3600.0, 2e9, False), ("b", 1800.0, 3e9, True)):
        (out / v).mkdir()
        (out / v / "prep.json").write_text(json.dumps({
            "video_id": v, "duration_s": 2 * main_s, "main_camera_s": main_s, "needs_ocr": ocr,
            "upload": "full" if ocr else "pack", "upload_bytes": up, "full_bytes": 4e9,
            "check_scenes": v == "b"}))
    p = batch.plan(["a", "b", "c"], fps=150.0, price_hourly=2.0, overhead_h=0.5)
    gpu_h = 5400 * 30 / 150 / 3600
    assert p["matches"] == 2 and p["not_prepped"] == ["c"] and p["needs_ocr"] == 1 and p["packed"] == 1
    assert p["gpu_h"] == pytest.approx(round(gpu_h, 1)) and p["cost_usd"] == pytest.approx(round((gpu_h + 0.5) * 2, 2))
    assert p["upload_gb"] == 5.0 and p["full_gb"] == 8.0 and p["check_scenes"] == ["b"]
    assert p["upload_mbps_to_keep_gpu_busy"] == pytest.approx(round(5e9 * 8 / (gpu_h * 3600) / 1e6, 1))
    assert "2 matches to run" in batch.format_plan(p)


def test_prep_match_packs_matches_with_official_data(src, dirs, monkeypatch):
    out, down = dirs
    from tennis_pipeline import audio, scenes

    (down / "p1.mp4").write_bytes(src.read_bytes())
    (down / "o1.mp4").write_bytes(src.read_bytes())

    def segments(vid):
        (out / vid).mkdir(exist_ok=True)
        pd.DataFrame({"segment_id": [0, 1], "start": [2.0, 14.0], "end": [6.5, 19.0], "duration": [4.5, 5.0]}).to_csv(
            out / vid / "segments.csv", index=False)

    monkeypatch.setattr(scenes, "predict_segments", segments)
    monkeypatch.setattr(scenes, "save_scene_prob", lambda vid: None)
    monkeypatch.setattr(audio, "onsets", lambda vid, path: None)
    monkeypatch.setattr(batch, "_official_missing", lambda vid: vid.startswith("o"))
    monkeypatch.setattr(batch, "DOWNLOADS", down)
    rec = batch.prep_match("p1", pack=True)
    assert rec["upload"] == "pack" and video.is_pack(down / "p1.pack")
    assert rec["upload_bytes"] < rec["full_bytes"] and rec["main_camera_s"] == 9.5 and not rec["needs_ocr"]
    again = batch.prep_match("p1", pack=True)  # cached pack is reused
    assert again["upload_bytes"] == rec["upload_bytes"]
    ocr = batch.prep_match("o1", pack=True)  # OCR reads the whole broadcast, so no pack
    assert ocr["upload"] == "full" and ocr["needs_ocr"] and not (down / "o1.pack").exists()
    assert json.loads((out / "p1" / "prep.json").read_text())["upload"] == "pack"

    from tennis_pipeline import ocr as ocr_mod

    (down / "o2.mp4").write_bytes(src.read_bytes())
    monkeypatch.setattr(ocr_mod, "run", lambda vid, path: (out / vid / "ocr_points.csv").write_text("x") and {})
    local = batch.prep_match("o2", pack=True, ocr_local=True)  # read here, so the droplet gets a pack
    assert local["upload"] == "pack" and local["needs_ocr"]
    assert batch._ocr("o2") == {"cached": True}
