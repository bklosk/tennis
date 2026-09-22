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
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

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

    def rsync_up(self, src: str | Path, dest: str, excludes: tuple[str, ...] = ()):
        ex = sum((["--exclude", e] for e in excludes), [])
        subprocess.run(["rsync", "-az", "--partial", *ex, "-e", self._rsh(), str(src), f"root@{self.ip}:{dest}"],
                       check=True)

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
for i in $(seq 60); do fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break; sleep 5; done
if ! command -v ffmpeg >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update -qq && apt-get install -y -qq ffmpeg rsync >/dev/null
fi
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


def run_logged(d: Droplet, script: str, log: str, poll_s: float = 15.0, sync=None) -> bool:
    """Run `script` in the background on the droplet, streaming its log until it exits.

    Returns False if the budget ran out first (the job is then killed). `sync` is called
    every few polls so partial results survive a forced stop.
    """
    d.ssh(f"cat > /root/job.sh <<'EOF'\n{script}\nEOF\n"
          f"setsid nohup bash /root/job.sh > {log} 2>&1 < /dev/null & echo $! > /root/job.pid", timeout=30)
    offset, polls = 0, 0
    while True:
        out = d.ssh(f"tail -c +{offset + 1} {log}; echo; kill -0 $(cat /root/job.pid) 2>/dev/null && echo __RUNNING__",
                    check=False, timeout=60).stdout
        running = out.rstrip().endswith("__RUNNING__")
        text = out.rstrip()[: -len("__RUNNING__")] if running else out
        text = text[:-1] if text.endswith("\n") else text
        if text:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)
            offset += len(text.encode())
        if not running:
            return True
        if d.remaining() <= 0:
            print(f"budget reached (${d.budget.spent():.2f}); stopping remote job", flush=True)
            d.ssh("pkill -P $(cat /root/job.pid); kill $(cat /root/job.pid)", check=False, timeout=30)
            return False
        polls += 1
        if sync and polls % 8 == 0:
            sync()
        time.sleep(poll_s)


def bench(video: Path, budget: float, size: str, region: str, out: Path) -> dict | None:
    do = DO()
    left = cleanup_leftovers(do)
    if left:
        print(f"destroyed leftover pilot droplets: {left}")
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
    sub.add_parser("cleanup", help="destroy droplets left by earlier runs")
    args = ap.parse_args()
    if args.cmd == "bench":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(bench(args.video, args.budget, args.size, args.region, args.out), indent=2))
    elif args.cmd == "cleanup":
        print(cleanup_leftovers(DO()))


if __name__ == "__main__":
    main()
