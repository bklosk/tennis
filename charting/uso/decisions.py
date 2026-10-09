"""OpenAI Decisions API client with an on-disk cache and a hard spend cap.

Every request is cached by a hash of (model, text, image bytes, questions), so re-running a
stage never pays twice. Every paid request is appended to a ledger; before each request the
ledger total plus a worst-case estimate is checked against the cap, and the client refuses to
send anything that could cross it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from uso.paths import OUTPUTS, REPO

MODEL = "gpt-6-luna"
USD_PER_INPUT_TOKEN = 0.10 / 1_000_000
# The user's limit is $15. Stop short of it so price multipliers we can't see stay inside it.
HARD_CAP_USD = 14.0

DECISIONS_DIR = OUTPUTS / "decisions"
LEDGER = DECISIONS_DIR / "ledger.jsonl"
CACHE = DECISIONS_DIR / "cache"


class BudgetExceeded(RuntimeError):
    pass


def _load_key() -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "OPENAI_API_KEY":
                return v.strip()
    raise RuntimeError("OPENAI_API_KEY is not set and .env has no key")


def image_part(jpeg: bytes) -> dict:
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}


@dataclass
class Request:
    text: str
    images: list[bytes]
    questions: list[dict]

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(MODEL.encode())
        h.update(self.text.encode())
        for im in self.images:
            h.update(hashlib.sha256(im).digest())
        h.update(json.dumps(self.questions, sort_keys=True).encode())
        return h.hexdigest()


class DecisionClient:
    def __init__(self, cap_usd: float = HARD_CAP_USD, max_tokens_per_request: int = 8000):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.cap = cap_usd
        self.max_tokens = max_tokens_per_request
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._client = None

    # -- spend accounting -------------------------------------------------------------
    @staticmethod
    def spent() -> float:
        if not LEDGER.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in LEDGER.read_text().splitlines() if line.strip())

    @staticmethod
    def tokens_spent() -> int:
        if not LEDGER.exists():
            return 0
        return sum(json.loads(line)["input_tokens"] for line in LEDGER.read_text().splitlines() if line.strip())

    def check_estimate(self, n_requests: int, tokens_per_request: float, label: str = "") -> float:
        """Raise before a bulk run whose estimated cost would cross the cap."""
        est = n_requests * tokens_per_request * USD_PER_INPUT_TOKEN
        have = self.spent()
        print(f"[decisions] {label} estimate: {n_requests} requests x {tokens_per_request:.0f} tok = ${est:.3f}; "
              f"spent so far ${have:.3f}; cap ${self.cap:.2f}")
        if have + est > self.cap:
            raise BudgetExceeded(f"estimated ${have + est:.2f} would exceed cap ${self.cap:.2f}")
        return est

    def _reserve(self) -> float:
        worst = self.max_tokens * USD_PER_INPUT_TOKEN
        with self._lock:
            if self.spent() + self._reserved + worst > self.cap:
                raise BudgetExceeded(f"cap ${self.cap:.2f} reached (spent ${self.spent():.3f})")
            self._reserved += worst
        return worst

    def _record(self, reserved: float, key: str, tokens: int, label: str) -> None:
        usd = tokens * USD_PER_INPUT_TOKEN
        with self._lock:
            self._reserved -= reserved
            with LEDGER.open("a") as f:
                f.write(json.dumps({"t": time.time(), "key": key, "input_tokens": tokens, "usd": usd, "label": label}) + "\n")

    # -- requests ---------------------------------------------------------------------
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=_load_key(), max_retries=0, timeout=120)
        return self._client

    def ask(self, req: Request, label: str = "", cached_only: bool = False) -> dict | None:
        """Answers as {name: answer_dict}, plus '_usage'. Cached responses are free."""
        key = req.key()
        path = CACHE / key[:2] / f"{key}.json"
        if path.exists():
            return _answers(json.loads(path.read_text()))
        if cached_only:
            return None
        content = [{"type": "input_text", "text": req.text}] + [image_part(im) for im in req.images]
        reserved = self._reserve()
        delay = 2.0
        try:
            for attempt in range(8):
                try:
                    resp = self.client.decisions.create(
                        model=MODEL, input=[{"role": "user", "content": content}], questions=req.questions
                    )
                    break
                except Exception as e:  # rate limits and transient server errors
                    status = getattr(e, "status_code", None)
                    if status in (429, 500, 502, 503, 504) or "timeout" in type(e).__name__.lower():
                        time.sleep(delay)
                        delay = min(delay * 2, 60)
                        continue
                    raise
            else:
                raise RuntimeError("Decisions API kept failing")
        except BaseException:
            with self._lock:
                self._reserved -= reserved
            raise
        raw = resp.model_dump()
        tokens = int((raw.get("usage") or {}).get("input_tokens") or self.max_tokens)
        self._record(reserved, key, tokens, label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
        return _answers(raw)

    def ask_many(self, reqs: list[Request], label: str = "", workers: int = 8, cached_only: bool = False) -> list:
        def one(r):
            try:
                return self.ask(r, label=label, cached_only=cached_only)
            except BudgetExceeded:
                raise
            except Exception as e:  # keep the batch going; the caller sees None
                print(f"[decisions] request failed: {type(e).__name__}: {str(e)[:200]}")
                return None

        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(one, reqs))


def _answers(raw: dict) -> dict:
    out = {a["name"]: a for a in raw.get("answers", [])}
    out["_usage"] = raw.get("usage") or {}
    return out


def choice_probs(answer: dict | None) -> dict[str, float]:
    if not answer or answer.get("type") != "choice":
        return {}
    return {p["value"]: float(p["probability"]) for p in answer.get("probabilities", [])}
