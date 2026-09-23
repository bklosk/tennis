"""Run pipeline work on a DigitalOcean droplet under a hard dollar cap.

Billing is per second while a droplet exists (even powered off), so every droplet gets two
independent kill switches:
  * a self-destruct timer installed through cloud-init that deletes the droplet through the
    API once the budget is spent, even if the machine running this launcher dies, and
  * a local watchdog that stops remote work, syncs results and destroys the droplet before it.

The token is read from DIGITALOCEAN_ACCESS_TOKEN (a scoped token with droplet and ssh_key
access is enough). DIGITALOCEAN_SELF_DESTRUCT_TOKEN, if set, is the one embedded in the
droplet's user data instead.
"""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .batch import UPLOADS_DONE

API = "https://api.digitalocean.com/v2"
GPU_IMAGE = "gpu-h100x1-base"  # NVIDIA AI/ML-ready image (drivers + CUDA); valid on all NVIDIA GPU sizes
CPU_IMAGE = "ubuntu-24-04-x64"
TAG = "tennis-pilot"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=10"]


class DO:
    def __init__(self, token: str | None = None):
        self.token = token or os.environ["DIGITALOCEAN_ACCESS_TOKEN"]

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{API}{path}", data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                if e.code == 404 and method == "DELETE":
                    return {}
                if e.code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"{method} {path}: {e.code} {e.read().decode()[:300]}") from None
        return {}

    def size(self, slug: str) -> dict:
        for s in self.request("GET", "/sizes?per_page=200")["sizes"]:
            if s["slug"] == slug:
                return s
        raise KeyError(slug)

    def tagged_droplets(self) -> list[dict]:
        return self.request("GET", f"/droplets?tag_name={TAG}&per_page=200").get("droplets", [])


@dataclass
class Budget:
    """Dollar cap for one droplet. DigitalOcean bills per second with a 60 s minimum."""
    price_hourly: float
    cap_usd: float
    created_at: float = field(default_factory=time.time)

    @property
    def max_seconds(self) -> float:
        return self.cap_usd / self.price_hourly * 3600

    def spent(self, now: float | None = None) -> float:
        elapsed = max((now or time.time()) - self.created_at, 60.0)
        return elapsed / 3600 * self.price_hourly

    def remaining_seconds(self, now: float | None = None) -> float:
        return self.max_seconds - ((now or time.time()) - self.created_at)


def self_destruct_script(token: str, seconds: int) -> str:
    """cloud-init user data: delete this droplet through the API after `seconds`."""
    return f"""#!/bin/bash
cat > /usr/local/sbin/self-destruct <<'EOS'
#!/bin/bash
ID=$(curl -s http://169.254.169.254/metadata/v1/id)
for i in 1 2 3 4 5; do
  curl -s -X DELETE -H "Authorization: Bearer {token}" https://api.digitalocean.com/v2/droplets/$ID && exit 0
  sleep 20
done
EOS
chmod 700 /usr/local/sbin/self-destruct
nohup bash -c 'sleep {int(seconds)}; /usr/local/sbin/self-destruct' >/var/log/self-destruct.log 2>&1 &
"""


class Droplet:
    """One budget-capped droplet reachable over SSH with an ephemeral key."""

    def __init__(self, do: DO, size: str, region: str, cap_usd: float, name: str = "tennis-pilot",
                 image: str | None = None, grace_s: int = 120):
        self.do, self.size_slug, self.region, self.name = do, size, region, name
        info = do.size(size)
        if region not in info["regions"] or not info["available"]:
            raise RuntimeError(f"{size} is not available in {region} (regions: {info['regions']})")
        self.image = image or (GPU_IMAGE if size.startswith("gpu-") else CPU_IMAGE)
        self.price = float(info["price_hourly"])
        self.cap_usd = cap_usd
        self.grace_s = grace_s
        self.id = self.ip = self.key_id = None
        self.budget: Budget | None = None
        self.tmp = Path(tempfile.mkdtemp(prefix="do-pilot-"))
        self.key_path = self.tmp / "id_ed25519"

    def __enter__(self):
        try:
            self.create()
        except BaseException:
            self.destroy()  # __exit__ does not run when __enter__ raises
            raise
        return self

    def __exit__(self, *exc):
        self.destroy()

    def create(self):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.key_path)], check=True)
        pub = (self.key_path.with_suffix(".pub")).read_text().strip()
        self.key_id = self.do.request("POST", "/account/keys", {"name": f"{self.name}-{int(time.time())}",
                                                                "public_key": pub})["ssh_key"]["id"]
        token = os.environ.get("DIGITALOCEAN_SELF_DESTRUCT_TOKEN", self.do.token)
        # The droplet deletes itself when the cap is reached, whatever happens to this process.
        seconds = max(Budget(self.price, self.cap_usd).max_seconds - 60, 120)
        body = {"name": self.name, "region": self.region, "size": self.size_slug, "image": self.image,
                "ssh_keys": [self.key_id], "tags": [TAG], "user_data": self_destruct_script(token, seconds),
                "monitoring": False, "ipv6": False}
        for attempt in range(6):
            try:
                drop = self.do.request("POST", "/droplets", body)["droplet"]
                break
            except RuntimeError as e:
                # A just-registered SSH key can take a few seconds to become usable.
                if "invalid key identifiers" not in str(e) or attempt == 5:
                    raise
                time.sleep(5)
        self.id = drop["id"]
        self.budget = Budget(self.price, self.cap_usd)
        print(f"created droplet {self.id} ({self.size_slug}, ${self.price}/h, cap ${self.cap_usd:.2f} "
              f"= {self.budget.max_seconds / 60:.0f} min)", flush=True)
        for _ in range(120):
            d = self.do.request("GET", f"/droplets/{self.id}")["droplet"]
            nets = [n["ip_address"] for n in d["networks"].get("v4", []) if n["type"] == "public"]
            if d["status"] == "active" and nets:
                self.ip = nets[0]
                break
            time.sleep(5)
        else:
            raise RuntimeError("droplet did not become active")
        for _ in range(60):
            if self.ssh("true", check=False, timeout=20).returncode == 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError("ssh never came up")
        self._arm_self_destruct(token)
        print(f"droplet {self.id} ready at {self.ip} after {time.time() - self.budget.created_at:.0f} s", flush=True)

    def _arm_self_destruct(self, token: str):
        """cloud-init runs user scripts late in boot, so arm the timer over SSH as well and verify it."""
        seconds = int(max(self.budget.remaining_seconds() - 60, 60))
        script = self_destruct_script(token, seconds)
        self.ssh(f"cat > /root/arm-self-destruct.sh <<'EOF'\n{script}\nEOF\n"
                 "setsid bash /root/arm-self-destruct.sh >/dev/null 2>&1 < /dev/null &", timeout=30)
        # "[s]leep" keeps pgrep from matching this command's own shell.
        ok = self.ssh("pgrep -f '[s]leep [0-9]+; /usr/local/sbin/self-destruct' >/dev/null && echo armed",
                      check=False, timeout=20).stdout.strip()
        if ok != "armed":
            self.destroy()
            raise RuntimeError("could not arm the self-destruct timer; droplet destroyed")
        print(f"self-destruct armed: droplet deletes itself in {seconds / 60:.0f} min", flush=True)

    def ssh(self, cmd: str, check: bool = True, timeout: float | None = None,
            capture: bool = True) -> subprocess.CompletedProcess:
        full = ["ssh", "-i", str(self.key_path), *SSH_OPTS, f"root@{self.ip}", cmd]
        return subprocess.run(full, check=check, timeout=timeout, text=True,
                              capture_output=capture)

    def rsync_up(self, src: str | Path, dest: str, excludes: tuple[str, ...] = (), compress: bool = True):
        ex = sum((["--exclude", e] for e in excludes), [])
        subprocess.run(["rsync", "-az" if compress else "-a", "--partial", *ex, "-e", self._rsh(), str(src),
                        f"root@{self.ip}:{dest}"], check=True)

    def rsync_down(self, src: str, dest: str | Path, excludes: tuple[str, ...] = ()):
        ex = sum((["--exclude", e] for e in excludes), [])
        subprocess.run(["rsync", "-az", "--partial", *ex, "-e", self._rsh(), f"root@{self.ip}:{src}", str(dest)],
                       check=False)

    def _rsh(self) -> str:
        return " ".join(["ssh", "-i", shlex.quote(str(self.key_path)), *SSH_OPTS])

    def remaining(self) -> float:
        return self.budget.remaining_seconds() - self.grace_s if self.budget else 0.0

    def destroy(self):
        if self.id:
            self.do.request("DELETE", f"/droplets/{self.id}")
            spent = self.budget.spent() if self.budget else 0
            print(f"destroyed droplet {self.id}; approx cost ${spent:.2f}", flush=True)
            self.id = None
        if self.key_id:
            self.do.request("DELETE", f"/account/keys/{self.key_id}")
            self.key_id = None


def cleanup_leftovers(do: DO, keep_key: int | None = None) -> list[int]:
    """Destroy droplets (and SSH keys) left behind by an earlier run of this launcher."""
    ids = [d["id"] for d in do.tagged_droplets()]
    for i in ids:
        do.request("DELETE", f"/droplets/{i}")
    for k in do.request("GET", "/account/keys?per_page=200").get("ssh_keys", []):
        if k["name"].startswith("tennis-") and k["id"] != keep_key:
            do.request("DELETE", f"/account/keys/{k['id']}")
    return ids


REPO = Path(__file__).resolve().parents[2]
REMOTE = "/root/tennis"
REPO_EXCLUDES = (".git", ".cache", "outputs", "downloads", "pipeline/.venv", "__pycache__", ".pytest_cache")
SETUP = f"""set -e
export DEBIAN_FRONTEND=noninteractive
# Unattended upgrades hold the apt/dpkg locks for a while after boot: wait for them and retry.
APT="apt-get -qq -o DPkg::Lock::Timeout=600"
for attempt in 1 2 3 4 5; do
  if command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null && command -v rsync >/dev/null; then break; fi
  ($APT update && $APT install -y ffmpeg rsync) >/dev/null 2>&1 || sleep 15
done
command -v ffprobe >/dev/null || {{ echo "ffmpeg install failed" >&2; exit 1; }}
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
mkdir -p {REMOTE}/.cache {REMOTE}/outputs {REMOTE}/downloads
"""
INSTALL = f"""set -e
cd {REMOTE}/pipeline && ~/.local/bin/uv sync --frozen --no-dev -q
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo "hwaccels: $(ffmpeg -hide_banner -hwaccels | tail -n +2 | tr '\\n' ' ')"
"""


def upload_code(d: Droplet):
    d.rsync_up(f"{REPO}/", REMOTE, excludes=REPO_EXCLUDES)
    if (REPO / ".cache" / "weights").exists():
        d.rsync_up(REPO / ".cache" / "weights", f"{REMOTE}/.cache/")


def run_logged(d: Droplet, script: str, log: str, poll_s: float = 15.0, sync=None, sync_every: int = 8,
               max_offline_s: float = 1800.0) -> bool:
    """Run `script` in the background on the droplet, streaming its log until it exits.

    Returns False if the budget ran out first (the job is then killed) or the droplet stayed
    unreachable for `max_offline_s`. `sync` is called every `sync_every` polls so partial results
    survive a forced stop. Every poll refreshes /root/heartbeat, which the batch job watches to
    detect a launcher that has gone away.
    """
    d.ssh(f"touch /root/heartbeat; cat > /root/job.sh <<'EOF'\n{script}\nEOF\n"
          f"setsid nohup bash /root/job.sh > {log} 2>&1 < /dev/null & echo $! > /root/job.pid", timeout=30)
    offset, polls, offline_since = 0, 0, None
    while True:
        try:
            out = d.ssh(f"touch /root/heartbeat; tail -c +{offset + 1} {log}; echo; "
                        "if kill -0 $(cat /root/job.pid) 2>/dev/null; then echo __RUNNING__; else echo __EXITED__; fi",
                        check=False, timeout=120).stdout
        except subprocess.TimeoutExpired:
            out = ""
        marker = next((m for m in ("__RUNNING__", "__EXITED__") if out.rstrip().endswith(m)), None)
        if marker is None:  # ssh failed: a network blip must not end the run
            offline_since = offline_since or time.time()
            if time.time() - offline_since > max_offline_s:
                print(f"droplet unreachable for {max_offline_s / 60:.0f} min; giving up", flush=True)
                return False
            time.sleep(poll_s)
            continue
        offline_since = None
        text = out.rstrip()[: -len(marker)]
        text = text[:-1] if text.endswith("\n") else text
        if text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
            offset += len(text.encode())
        if marker == "__EXITED__":
            return True
        if d.remaining() <= 0:
            print(f"budget reached (${d.budget.spent():.2f}); stopping remote job", flush=True)
            d.ssh("pkill -P $(cat /root/job.pid); kill $(cat /root/job.pid)", check=False, timeout=30)
            return False
        polls += 1
        if sync and polls % sync_every == 0:
            sync()
        time.sleep(poll_s)


def bench(video: Path, budget: float, size: str, region: str, out: Path) -> dict | None:
    do = DO()
    left = cleanup_leftovers(do)
    if left:
        print(f"destroyed leftover pilot droplets: {left}")
    out.unlink(missing_ok=True)  # never report an earlier run's numbers
    with Droplet(do, size, region, budget, name="tennis-gpu-bench") as d:
        t = time.time()
        d.ssh(SETUP, timeout=900)
        upload_code(d)
        d.rsync_up(video, f"{REMOTE}/downloads/bench.mp4")
        r = d.ssh(INSTALL, timeout=1200)
        print(r.stdout.strip(), flush=True)
        print(f"setup took {time.time() - t:.0f} s (${d.budget.spent():.2f} so far)", flush=True)
        script = (f"cd {REMOTE}/pipeline && ~/.local/bin/uv run python -m eval.gpu_bench "
                  f"{REMOTE}/downloads/bench.mp4 --out {REMOTE}/outputs/gpu_bench.json")
        fetch = lambda: d.rsync_down(f"{REMOTE}/outputs/gpu_bench.json", out)  # noqa: E731
        run_logged(d, script, "/root/bench.log", sync=fetch)
        fetch()
        spent = d.budget.spent()
    result = json.loads(out.read_text()) if out.exists() else None
    if result is not None:
        result["droplet"] = {"size": size, "cost_usd": round(spent, 3)}
        out.write_text(json.dumps(result, indent=2))
    return result


PILOT_JOB = """set -e
cd {remote}/pipeline
UV=$HOME/.local/bin/uv
TIMES={remote}/outputs/pilot_stage_times.jsonl
stage() {{ local name=$1; shift; local s=$(date +%s.%N); "$@"; \
  echo "{{\\"stage\\": \\"$name\\", \\"seconds\\": $(python3 -c "print(round($(date +%s.%N)-$s,1))")}}" >> $TIMES; }}
cli() {{ $UV run python -m tennis_pipeline.cli "$@"; }}
NEED=""
for v in {vids}; do [ -f {remote}/outputs/$v/segments.csv ] || NEED="$NEED $v"; done
if [ -n "$NEED" ]; then stage scenes cli scenes $NEED --reuse-classifier {scene_flags}; fi
for v in {vids}; do stage "track:$v" cli track $v; done
for v in {vids}; do
  stage "events:$v" cli events $v
  stage "crops:$v" cli crops $v
  stage "align:$v" cli align $v || echo "align failed for $v"
done
stage strokes cli strokes {vids} || echo "strokes failed"
stage report cli report {vids} || echo "report failed"
PYTHONPATH=. $UV run python eval/pilot_metrics.py {vids} > {remote}/outputs/pilot_metrics.json || echo "metrics failed"
echo PILOT_DONE
"""


def pilot(video_ids: list[str], videos_dir: Path, budget: float, size: str, region: str,
          scene_keyframes: bool = False) -> dict:
    """Full pipeline on a GPU droplet for videos on this machine; results sync to outputs/."""
    missing = [v for v in video_ids if not (videos_dir / f"{v}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"videos not found in {videos_dir}: {missing}")
    do = DO()
    left = cleanup_leftovers(do)
    if left:
        print(f"destroyed leftover pilot droplets: {left}")
    outputs = REPO / "outputs"
    with Droplet(do, size, region, budget, name="tennis-gpu-pilot") as d:
        t = time.time()
        d.ssh(SETUP, timeout=900)
        upload_code(d)
        # Install dependencies on the droplet while the videos upload.
        installer = threading.Thread(target=lambda: print(d.ssh(INSTALL, timeout=1800).stdout.strip(), flush=True))
        installer.start()
        for v in video_ids:
            d.rsync_up(videos_dir / f"{v}.mp4", f"{REMOTE}/downloads/")
            cached = [p for p in ("segments.csv", "scene_embeddings.npz", "audio_onsets.npz")
                      if (outputs / v / p).exists()]
            if cached:
                d.ssh(f"mkdir -p {REMOTE}/outputs/{v}", timeout=30)
                for p in cached:
                    d.rsync_up(outputs / v / p, f"{REMOTE}/outputs/{v}/")
        clf = REPO / ".cache" / "scene_classifier.npz"
        if clf.exists():
            d.rsync_up(clf, f"{REMOTE}/.cache/")
        installer.join()
        print(f"setup + upload took {time.time() - t:.0f} s (${d.budget.spent():.2f} so far)", flush=True)
        script = PILOT_JOB.format(remote=REMOTE, vids=" ".join(video_ids),
                                  scene_flags="--scene-keyframes" if scene_keyframes else "")

        def fetch():
            d.rsync_down(f"{REMOTE}/outputs/", outputs, excludes=("_bench",))
            d.rsync_down(f"{REMOTE}/.cache/hit_crops", REPO / ".cache")

        finished = run_logged(d, script, "/root/pilot.log", sync=fetch)
        fetch()
        spent = d.budget.spent()
    summary = {"videos": video_ids, "finished": finished, "size": size, "cost_usd": round(spent, 3)}
    times = outputs / "pilot_stage_times.jsonl"
    if times.exists():
        summary["stage_seconds"] = [json.loads(line) for line in times.read_text().splitlines()]
    (outputs / "gpu_pilot_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


BATCH_JOB = """set -e
cd {remote}/pipeline
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1
touch /root/heartbeat
# Dead man's switch: if the launcher stops polling (laptop asleep or offline) for {orphan_min} min, delete
# the droplet instead of waiting, at full price, for videos that will not come.
( while sleep 60; do
    age=$(( $(date +%s) - $(stat -c %Y /root/heartbeat) ))
    if [ "$age" -gt {orphan_s} ]; then echo "no launcher heartbeat for $age s; deleting the droplet"; \
/usr/local/sbin/self-destruct; fi
  done ) &
WATCHDOG=$!
trap "kill $WATCHDOG" EXIT
$HOME/.local/bin/uv run python -c "from rapidocr import RapidOCR; RapidOCR()" >/dev/null 2>&1 || echo "rapidocr warm-up failed"
$HOME/.local/bin/uv run python -m tennis_pipeline.batch run --manifest {remote}/outputs/batch_manifest.txt \\
  --wait-for-videos --delete-videos --cpu-workers {cpu_workers} --gpu-workers {gpu_workers}
echo BATCH_DONE
"""


class Uploader(threading.Thread):
    """Streams each match's prep outputs and video (or main-camera pack) to the droplet, in order.

    At most `max_ahead` videos wait on the droplet (the runner deletes a video once its match is
    finished), which bounds droplet disk use. Each finished upload is marked with
    downloads/VIDEO_ID.ready; downloads/UPLOADS_DONE follows the last one.
    """

    def __init__(self, d: Droplet, recs: list[dict], max_ahead: int = 4, streams: int = 2):
        super().__init__(daemon=True)
        self.d, self.recs, self.max_ahead, self.streams = d, recs, max_ahead, streams
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active = 0
        self.bytes = 0
        self.seconds = 0.0
        self.done: list[str] = []
        self.failed: list[str] = []

    def waiting_on_droplet(self) -> int:
        r = self.d.ssh(f"ls {REMOTE}/downloads/*.ready 2>/dev/null | wc -l", check=False, timeout=60)
        return int(r.stdout.strip() or 0) if r.returncode == 0 else self.max_ahead

    def _slot(self) -> bool:
        while not self.stop.is_set():
            waiting = self.waiting_on_droplet()
            with self.lock:
                if self.active < self.streams and waiting + self.active < self.max_ahead:
                    self.active += 1
                    return True
            self.stop.wait(20)
        return False

    def upload(self, rec: dict):
        from . import cli

        vid, d = rec["video_id"], self.d
        local = REPO / "outputs" / vid
        d.ssh(f"mkdir -p {REMOTE}/outputs/{vid} {REMOTE}/.cache/hit_crops", timeout=60)
        # Prep outputs, plus any partial results (tracks, status) when resuming. The embeddings
        # stay local: the droplet only needs the per-sample probabilities in scene_prob.npz.
        d.rsync_up(f"{local}/", f"{REMOTE}/outputs/{vid}/", excludes=("*.mp4", "scene_embeddings.npz"))
        crops = REPO / ".cache" / "hit_crops" / vid
        if crops.exists():
            d.rsync_up(crops, f"{REMOTE}/.cache/hit_crops/")
        if rec["upload"] == "pack":
            src, dest = REPO / "downloads" / f"{vid}.pack", f"{REMOTE}/downloads/"
        else:
            src, dest = cli.video_path(vid), f"{REMOTE}/downloads/{vid}.mp4"
        t = time.time()
        d.rsync_up(src, dest, compress=False)
        dt = time.time() - t
        d.ssh(f"touch {REMOTE}/downloads/{vid}.ready", timeout=60)
        with self.lock:
            self.bytes += rec["upload_bytes"]
            self.seconds += dt
            self.done.append(vid)
        print(f"uploaded {vid} ({rec['upload']}, {rec['upload_bytes'] / 1e9:.2f} GB) in {dt / 60:.1f} min at "
              f"{rec['upload_bytes'] * 8 / max(dt, 1e-6) / 1e6:.0f} Mbps; {len(self.done)}/{len(self.recs)} uploaded",
              flush=True)

    def _worker(self, queue: list[dict]):
        while not self.stop.is_set():
            with self.lock:
                if not queue:
                    return
                rec = queue.pop(0)
            if not self._slot():
                return
            try:
                for attempt in range(3):
                    try:
                        self.upload(rec)
                        break
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                        if attempt == 2:
                            raise
                        print(f"upload of {rec['video_id']} failed ({e}); retrying", flush=True)
                        self.stop.wait(30 * (attempt + 1))
            except Exception as e:
                with self.lock:
                    self.failed.append(rec["video_id"])
                print(f"giving up on uploading {rec['video_id']}: {e}", flush=True)
            finally:
                with self.lock:
                    self.active -= 1

    def run(self):
        queue = list(self.recs)
        workers = [threading.Thread(target=self._worker, args=(queue,), daemon=True) for _ in range(self.streams)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        if not self.stop.is_set():
            self.d.ssh(f"touch {REMOTE}/downloads/{UPLOADS_DONE}", check=False, timeout=60)
            print(f"all uploads finished: {self.bytes / 1e9:.1f} GB at "
                  f"{self.bytes * 8 / max(self.seconds, 1e-6) / 1e6 * self.streams:.0f} Mbps overall", flush=True)


def batch(video_ids: list[str], budget: float, size: str, region: str, pack: bool = True, cpu_workers: int = 2,
          max_ahead: int = 8, upload_streams: int = 2, scene_keyframes: bool = False, plan_only: bool = False,
          assume_fps: float = 190.0, orphan_min: int = 45, ocr_local: bool = False, gpu_workers: int = 1) -> dict:
    """Prep locally, then run the whole batch on one budget-capped GPU droplet while streaming videos.

    Rerunning the same command resumes: matches whose results are already synced are skipped and
    partly processed matches continue from their cached tracks.
    """
    from . import batch as B

    if sys.platform == "darwin":  # keep the Mac awake while this process runs
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    recs = B.prep(video_ids, pack=pack, keyframes=scene_keyframes, ocr_local=ocr_local)
    todo = [r for r in recs if not r.get("error") and not B.is_complete(r["video_id"])]
    do = DO() if not plan_only or os.environ.get("DIGITALOCEAN_ACCESS_TOKEN") else None
    price = float(do.size(size)["price_hourly"]) if do else 1.57
    p = B.plan([r["video_id"] for r in recs], fps=assume_fps, price_hourly=price)
    print(B.format_plan(p), flush=True)
    if plan_only or not todo:
        return p
    if p["cost_usd"] > budget:
        print(f"budget ${budget} is below the estimate; the run stops at the cap and can be resumed", flush=True)
    left = cleanup_leftovers(do)
    if left:
        print(f"destroyed leftover droplets: {left}")
    outputs = REPO / "outputs"
    t_start = time.time()
    with Droplet(do, size, region, budget, name="tennis-batch") as d:
        d.ssh(SETUP, timeout=900)
        upload_code(d)
        for rel in ("scene_classifier.npz", "sackmann"):
            if (REPO / ".cache" / rel).exists():
                d.rsync_up(REPO / ".cache" / rel, f"{REMOTE}/.cache/")
        manifest = d.tmp / "batch_manifest.txt"
        manifest.write_text("\n".join(r["video_id"] for r in todo) + "\n")
        d.rsync_up(manifest, f"{REMOTE}/outputs/batch_manifest.txt")
        uploader = Uploader(d, todo, max_ahead=max_ahead, streams=upload_streams)
        uploader.start()  # uploads overlap the dependency install
        print(d.ssh(INSTALL, timeout=1800).stdout.strip(), flush=True)
        print(f"droplet ready after {(time.time() - t_start) / 60:.1f} min (${d.budget.spent():.2f} so far)", flush=True)

        def fetch():
            d.rsync_down(f"{REMOTE}/outputs/", outputs)
            d.rsync_down(f"{REMOTE}/.cache/hit_crops", REPO / ".cache")

        script = BATCH_JOB.format(remote=REMOTE, cpu_workers=cpu_workers, gpu_workers=gpu_workers,
                                  orphan_s=orphan_min * 60, orphan_min=orphan_min)
        finished = run_logged(d, script, "/root/batch.log", poll_s=20, sync=fetch, sync_every=15)
        uploader.stop.set()
        fetch()
        spent = d.budget.spent()
    summary = {"matches": len(todo), "finished": finished, "size": size, "cost_usd": round(spent, 2),
               "hours": round((time.time() - t_start) / 3600, 2), "uploaded": len(uploader.done),
               "upload_failed": uploader.failed, "upload_gb": round(uploader.bytes / 1e9, 1),
               "complete": sum(B.is_complete(r["video_id"]) for r in todo)}
    (outputs / "batch_run.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Budget-capped DigitalOcean GPU runs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="component/end-to-end throughput on a GPU droplet")
    b.add_argument("video", type=Path)
    b.add_argument("--budget", type=float, default=1.0, help="hard cap in USD for this droplet")
    b.add_argument("--size", default="gpu-l40sx1-48gb")
    b.add_argument("--region", default="tor1")
    b.add_argument("--out", type=Path, default=REPO / "outputs" / "gpu_bench.json")
    p = sub.add_parser("pilot", help="full pipeline on a GPU droplet for local videos")
    p.add_argument("video_ids", nargs="+")
    p.add_argument("--videos-dir", type=Path, default=REPO / "downloads")
    p.add_argument("--budget", type=float, default=4.0, help="hard cap in USD for this droplet")
    p.add_argument("--size", default="gpu-l40sx1-48gb")
    p.add_argument("--region", default="tor1")
    p.add_argument("--scene-keyframes", action="store_true", help="keyframe-only scene decoding (~6x faster)")
    bt = sub.add_parser("batch", help="prep locally, then run every match on one GPU droplet (resumable)")
    bt.add_argument("video_ids", nargs="*")
    bt.add_argument("--manifest", default=None, help="file with one video id per line, or a preset (usopen)")
    bt.add_argument("--budget", type=float, help="hard cap in USD for the droplet (required unless --plan)")
    bt.add_argument("--size", default="gpu-l40sx1-48gb")
    bt.add_argument("--region", default="tor1")
    bt.add_argument("--no-pack", action="store_true", help="upload full videos instead of main-camera packs")
    bt.add_argument("--cpu-workers", type=int, default=2, help="droplet processes for OCR/events/align/report")
    bt.add_argument("--gpu-workers", type=int, default=1, help="matches tracked at once on the droplet")
    bt.add_argument("--max-ahead", type=int, default=8, help="videos allowed to wait on the droplet")
    bt.add_argument("--ocr-local", action="store_true",
                    help="prep: OCR matches without official data on this machine, so they upload as packs too")
    bt.add_argument("--upload-streams", type=int, default=2, help="parallel uploads")
    bt.add_argument("--scene-keyframes", action="store_true", help="prep: keyframe-only scene decoding")
    bt.add_argument("--assume-fps", type=float, default=190.0, help="tracking fps for the cost estimate")
    bt.add_argument("--plan", action="store_true", help="prep and print the plan; no droplet")
    sub.add_parser("cleanup", help="destroy droplets left by earlier runs")
    args = ap.parse_args()
    if args.cmd == "bench":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(bench(args.video, args.budget, args.size, args.region, args.out), indent=2))
    elif args.cmd == "pilot":
        print(json.dumps(pilot(args.video_ids, args.videos_dir, args.budget, args.size, args.region,
                               args.scene_keyframes), indent=2))
    elif args.cmd == "batch":
        from .batch import load_manifest

        vids = load_manifest(args.manifest, args.video_ids)
        if not vids:
            ap.error("batch needs video ids or --manifest")
        if args.budget is None and not args.plan:
            ap.error("batch needs --budget (or --plan to only prep and estimate)")
        print(json.dumps(batch(vids, args.budget, args.size, args.region, pack=not args.no_pack,
                               cpu_workers=args.cpu_workers, max_ahead=args.max_ahead,
                               upload_streams=args.upload_streams, scene_keyframes=args.scene_keyframes,
                               plan_only=args.plan, assume_fps=args.assume_fps, ocr_local=args.ocr_local,
                               gpu_workers=args.gpu_workers), indent=2))
    elif args.cmd == "cleanup":
        print(cleanup_leftovers(DO()))


if __name__ == "__main__":
    main()
