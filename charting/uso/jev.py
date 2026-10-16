"""TypeSafe Jev client for image questions, with an on-disk cache and a hard spend cap.

Same contract as `uso.decisions`: every request is cached by a hash of its body, so re-running a
stage never pays twice; every paid request is appended to its own ledger; before each request the
ledger total plus a worst-case reservation is checked against the cap, and the client refuses to
send anything that could cross it. Images and Box questions are a preview feature and need the
`TypeSafe-Danger-Zone-Images` header.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from uso.paths import OUTPUTS, REPO

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"  # pinned: Box needs 1.14, and the alias can move under us
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000  # output tokens are free
# The user's limit for the Jev benchmark is $1.50.
HARD_CAP_USD = 1.50
RETRY = {429, 500, 502, 503, 504, 529}

JEV_DIR = OUTPUTS / "jev"
LEDGER = JEV_DIR / "ledger.jsonl"
CACHE = JEV_DIR / "cache"


class BudgetExceeded(RuntimeError):
    pass


def _load_key() -> str:
    for name in ("TYPESAFE_API_KEY", "JEV_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("TYPESAFE_API_KEY", "JEV_KEY") and v.strip():
                return v.strip().strip('"').strip("'")
    raise RuntimeError("TYPESAFE_API_KEY is not set and .env has no key")


def image_ref(jpeg: bytes) -> str:
    """Deterministic reference so identical requests hash (and cache) identically."""
    return "img-" + hashlib.sha256(jpeg).hexdigest()[:20]


def questions_from_decisions(qs: list[dict]) -> dict:
    """Decisions API questions (list, `choices`, `predicate`) as Jev questions (map, `criteria`, `noul`)."""
    out = {}
    for q in qs:
        if q["type"] == "choice":
            out[q["name"]] = {"type": "choice", "instructions": q["instructions"],
                              "criteria": {c["value"]: c["description"] for c in q["choices"]}}
        elif q["type"] == "predicate":
            out[q["name"]] = {"type": "noul", "instructions": f"Is this true? {q['instructions']}"}
        else:
            raise ValueError(q["type"])
    return out


@dataclass
class Request:
    """state: any JSON, with image references where the images go; images: {reference: jpeg}."""
    state: object
    questions: dict
    images: dict[str, bytes] = field(default_factory=dict)

    def body(self) -> bytes:
        assets = {ref: {"type": "image", "format": "jpg", "data_base64": base64.b64encode(im).decode()}
                  for ref, im in self.images.items()}
        return json.dumps({"model": MODEL, "state": self.state, "assets": assets, "questions": self.questions},
                          sort_keys=True).encode()

    def key(self) -> str:
        return hashlib.sha256(self.body()).hexdigest()


_TALLY = {"offset": 0, "usd": 0.0, "tokens": 0, "n": 0}
_TALLY_LOCK = threading.Lock()


def _tally() -> dict:
    with _TALLY_LOCK:
        if LEDGER.exists():
            with LEDGER.open("rb") as f:
                f.seek(_TALLY["offset"])
                data = f.read()
            end = data.rfind(b"\n") + 1
            for line in data[:end].splitlines():
                if line.strip():
                    r = json.loads(line)
                    _TALLY["usd"] += r["usd"]
                    _TALLY["tokens"] += r["input_tokens"]
                    _TALLY["n"] += 1
            _TALLY["offset"] += end
        return dict(_TALLY)


class JevClient:
    def __init__(self, cap_usd: float = HARD_CAP_USD, max_tokens_per_request: int = 20000):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.cap = cap_usd
        self.max_tokens = max_tokens_per_request
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._http = None

    @staticmethod
    def spent() -> float:
        return _tally()["usd"]

    @staticmethod
    def tally() -> dict:
        return _tally()

    def check_estimate(self, n_requests: int, tokens_per_request: float, label: str = "") -> float:
        est = n_requests * tokens_per_request * USD_PER_INPUT_TOKEN
        have = self.spent()
        print(f"[jev] {label} estimate: {n_requests} requests x {tokens_per_request:.0f} tok = ${est:.3f}; "
              f"spent so far ${have:.3f}; cap ${self.cap:.2f}", flush=True)
        if have + est > self.cap:
            raise BudgetExceeded(f"estimated ${have + est:.2f} would exceed cap ${self.cap:.2f}")
        return est

    def _reserve(self) -> float:
        worst = self.max_tokens * USD_PER_INPUT_TOKEN
        with self._lock:
            if self.spent() + self._reserved + worst > self.cap:
                raise BudgetExceeded(f"cap ${self.cap:.2f} reached (spent ${self.spent():.4f})")
            self._reserved += worst
        return worst

    def _record(self, reserved: float, key: str, tokens: int, label: str) -> None:
        usd = tokens * USD_PER_INPUT_TOKEN
        with self._lock:
            self._reserved -= reserved
            with LEDGER.open("a") as f:
                f.write(json.dumps({"t": time.time(), "key": key, "input_tokens": tokens, "usd": usd,
                                    "label": label}) + "\n")

    @property
    def http(self):
        if self._http is None:
            import httpx

            self._http = httpx.Client(timeout=120, headers={
                "Authorization": f"Bearer {_load_key()}", "Content-Type": "application/json",
                "TypeSafe-Danger-Zone-Images": "true"})
        return self._http

    def ask(self, req: Request, label: str = "", cached_only: bool = False) -> dict | None:
        """Raw response ({'answers': {...}, 'usage': {...}, 'model': ...}). Cached responses are free."""
        body = req.body()
        key = hashlib.sha256(body).hexdigest()
        path = CACHE / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        if cached_only:
            return None
        reserved = self._reserve()
        delay = 2.0
        try:
            for attempt in range(8):
                try:
                    r = self.http.post(ENDPOINT, content=body)
                except Exception as e:  # connection resets and timeouts
                    if attempt == 7:
                        raise
                    print(f"[jev] {type(e).__name__}, retrying", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                if r.status_code == 200:
                    break
                if r.status_code in RETRY and attempt < 7:
                    wait = float(r.headers.get("retry-after") or delay)
                    time.sleep(wait)
                    delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        except BaseException:
            with self._lock:
                self._reserved -= reserved
            raise
        raw = r.json()
        tokens = int((raw.get("usage") or {}).get("input_tokens") or self.max_tokens)
        self._record(reserved, key, tokens, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
        return raw

    def ask_many(self, reqs: list[Request], label: str = "", workers: int = 16, cached_only: bool = False) -> list:
        def one(r):
            try:
                return self.ask(r, label=label, cached_only=cached_only)
            except BudgetExceeded:
                raise
            except Exception as e:  # keep the batch going; the caller sees None
                print(f"[jev] request failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                return None

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, reqs))


def choice_probs(raw: dict | None, name: str) -> dict[str, float]:
    a = ((raw or {}).get("answers") or {}).get(name)
    if not a or a.get("type") != "choice":
        return {}
    return {k: float(v) for k, v in a.get("probabilities", {}).items()}


def noul(raw: dict | None, name: str) -> float:
    a = ((raw or {}).get("answers") or {}).get(name)
    return float(a["noul"]) if a and a.get("type") == "noul" else float("nan")


def box(raw: dict | None, name: str) -> dict | None:
    a = ((raw or {}).get("answers") or {}).get(name)
    if not a or a.get("type") != "box":
        return None
    return {**a["box"], "confidence": float(a.get("confidence", float("nan")))}
