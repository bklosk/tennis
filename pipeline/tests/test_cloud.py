import subprocess
import threading
import time

import pytest

from tennis_pipeline import cli, cloud


def test_budget_caps_runtime_and_bills_a_minimum_minute():
    b = cloud.Budget(price_hourly=1.57, cap_usd=4.0, created_at=0.0)
    assert b.max_seconds == pytest.approx(4.0 / 1.57 * 3600)
    assert b.spent(now=10.0) == pytest.approx(60 / 3600 * 1.57)
    assert b.spent(now=3600.0) == pytest.approx(1.57)
    assert b.remaining_seconds(now=b.max_seconds) == pytest.approx(0.0)


def test_self_destruct_script_is_valid_bash_and_targets_this_droplet():
    script = cloud.self_destruct_script("tok123", 900)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "sleep 900" in script
    assert "metadata/v1/id" in script
    assert "Bearer tok123" in script and "X DELETE" in script.replace("-X DELETE", "X DELETE")


class FakeDO:
    def __init__(self, sizes):
        self.token, self._sizes = "tok", sizes

    def size(self, slug):
        return self._sizes[slug]


def test_droplet_refuses_unavailable_region():
    do = FakeDO({"gpu-l40sx1-48gb": {"regions": ["tor1"], "available": True, "price_hourly": 1.57}})
    with pytest.raises(RuntimeError, match="not available"):
        cloud.Droplet(do, "gpu-l40sx1-48gb", "nyc1", cap_usd=1.0)
    d = cloud.Droplet(do, "gpu-l40sx1-48gb", "tor1", cap_usd=1.0)
    assert d.image == cloud.GPU_IMAGE and d.price == 1.57


def test_failed_create_destroys_the_droplet(monkeypatch):
    do = FakeDO({"s-1vcpu-1gb": {"regions": ["tor1"], "available": True, "price_hourly": 0.009}})
    d = cloud.Droplet(do, "s-1vcpu-1gb", "tor1", cap_usd=0.05)
    destroyed = []

    def boom():
        d.id = 42
        raise RuntimeError("ssh never came up")

    monkeypatch.setattr(d, "create", boom)
    monkeypatch.setattr(d, "destroy", lambda: destroyed.append(d.id))
    with pytest.raises(RuntimeError):
        with d:
            pass
    assert destroyed == [42]


def test_batch_job_is_valid_bash_with_a_dead_mans_switch():
    script = cloud.BATCH_JOB.format(remote="/root/tennis", cpu_workers=2, gpu_workers=1, orphan_s=2700, orphan_min=45)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "--wait-for-videos --delete-videos --cpu-workers 2 --gpu-workers 1" in script
    assert '"$age" -gt 2700' in script and "/usr/local/sbin/self-destruct" in script


class FakeDroplet:
    """Droplet stand-in: tracks .ready markers; a consumer thread plays the batch runner."""

    def __init__(self):
        self.ready, self.log, self.max_waiting, self.lock = set(), [], 0, threading.Lock()

    def ssh(self, cmd, check=True, timeout=None, capture=True):
        with self.lock:
            self.log.append(cmd)
            if cmd.startswith("ls ") and ".ready" in cmd:
                return subprocess.CompletedProcess(cmd, 0, f"{len(self.ready)}\n", "")
            if cmd.startswith("touch ") and cmd.endswith(".ready"):
                self.ready.add(cmd.rsplit("/", 1)[1][:-6])
                self.max_waiting = max(self.max_waiting, len(self.ready))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def rsync_up(self, src, dest, excludes=(), compress=True):
        with self.lock:
            self.log.append(("rsync", str(src), dest, compress))
        time.sleep(0.01)


def test_uploader_keeps_at_most_max_ahead_videos_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(cloud, "REPO", tmp_path)
    monkeypatch.setattr(cli, "video_path", lambda v: tmp_path / "downloads" / f"{v}.mp4")
    d = FakeDroplet()
    recs = [{"video_id": f"m{i}", "upload": "full" if i % 2 else "pack", "upload_bytes": 1000} for i in range(6)]
    up = cloud.Uploader(d, recs, max_ahead=2, streams=2)
    monkeypatch.setattr(up.stop, "wait", lambda s: time.sleep(0.01))
    done = threading.Event()

    def runner():  # finishes one waiting match at a time, like the droplet
        while not done.is_set():
            time.sleep(0.05)
            with d.lock:
                if d.ready:
                    d.ready.pop()

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    up.run()
    done.set()
    assert sorted(up.done) == [r["video_id"] for r in recs] and not up.failed
    assert d.max_waiting <= 2
    assert d.log[-1].endswith(f"downloads/{cloud.UPLOADS_DONE}")
    videos = [e for e in d.log if isinstance(e, tuple) and "/downloads" in e[2]]
    assert all(compress is False for *_, compress in videos)
    assert any(src.endswith("m0.pack") for _, src, _, _ in videos) and any(dest.endswith("m1.mp4") for *_, dest, _ in videos)


def test_run_logged_survives_dropped_polls(capsys):
    class D:
        budget = cloud.Budget(1.0, 10.0)
        replies = ["", "one\n\n__RUNNING__\n", "", "", "two\n\n__EXITED__\n"]

        def ssh(self, cmd, check=True, timeout=None, capture=True):
            out = self.replies.pop(0) if "tail -c" in cmd else ""
            return subprocess.CompletedProcess(cmd, 0 if out else 255, out, "")

        def remaining(self):
            return 1e9

    assert cloud.run_logged(D(), "true", "/root/x.log", poll_s=0)
    printed = capsys.readouterr().out
    assert "one" in printed and "two" in printed
