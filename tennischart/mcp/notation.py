"""Tokenizer, parser and serializer for Match Charting Project point strings.

A point is one or two serve attempts (the MCP `1st` and `2nd` columns). Each attempt is

    c*  serve  rally-shot*

where `c` is a let, the serve is a direction digit (4 wide, 5 body, 6 T, 0 unknown) with an
optional `+` (serve and volley) followed by a fault letter, `*` (ace), `#` (unreturnable) or
nothing (the rally continues). A rally shot is

    letter  modifiers  direction?  modifiers  depth?  modifiers  error?  outcome?

Whole-attempt codes: S (point to server), R (point to returner), P / Q (penalty points),
V (time violation, a fault). `C` closes a point interrupted by a challenge or replay.
Hitters are implied by alternation starting from the server; strings never name players.

The serializer reproduces the input exactly for every string the parser accepts, which is what
the round-trip tests check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

SERVE_DIRECTIONS = {"4": "wide", "5": "body", "6": "T", "0": "unknown"}
RALLY_DIRECTIONS = "0123"
DEPTHS = "789"
SHOT_LETTERS = "fbrsvzopuylmhijktq"
MODIFIERS = "+-=;^"
RALLY_ERRORS = "nwdx!e"
SERVE_FAULTS = "nwdxge!"
OUTCOMES = {"*": "winner", "#": "forced_error", "@": "unforced_error", "C": "interrupted"}
ATTEMPT_CODES = {"S", "R", "P", "Q", "V"}

FOREHAND_LETTERS = set("frvoulhj")
BACKHAND_LETTERS = set("bszpymik")

FAMILY_OF_LETTER = {
    "f": "groundstroke", "b": "groundstroke",
    "r": "slice", "s": "slice",
    "v": "volley", "z": "volley",
    "o": "overhead", "p": "overhead",
    "u": "drop_shot", "y": "drop_shot",
    "l": "lob", "m": "lob",
    "h": "half_volley", "i": "half_volley",
    "j": "swinging_volley", "k": "swinging_volley",
    "t": "trick", "q": "unknown",
}
LETTER_OF = {(fam, "forehand"): l for l, fam in FAMILY_OF_LETTER.items() if l in FOREHAND_LETTERS}
LETTER_OF.update({(fam, "backhand"): l for l, fam in FAMILY_OF_LETTER.items() if l in BACKHAND_LETTERS})
LETTER_OF[("trick", "forehand")] = LETTER_OF[("trick", "backhand")] = "t"

STROKE_FAMILIES = [
    "serve", "groundstroke", "slice", "volley", "half_volley", "swinging_volley",
    "overhead", "drop_shot", "lob", "trick",
]


def stroke_side(letter: str) -> str | None:
    """'forehand', 'backhand', 'serve' or None (trick / unknown)."""
    if letter == "serve":
        return "serve"
    if letter in FOREHAND_LETTERS:
        return "forehand"
    if letter in BACKHAND_LETTERS:
        return "backhand"
    return None


def stroke_family(letter: str) -> str:
    if letter == "serve":
        return "serve"
    return FAMILY_OF_LETTER.get(letter, "unknown")


def letter_for(family: str, side: str | None) -> str:
    """MCP shot letter for a (family, side) pair; 'q' when it cannot be expressed."""
    if family == "serve":
        return "serve"
    if side not in ("forehand", "backhand"):
        return "t" if family == "trick" else "q"
    return LETTER_OF.get((family, side), "q")


@dataclass
class Shot:
    index: int                 # 0 = serve, 1 = return, 2 = serve+1, ...
    stroke: str                # "serve" or an MCP shot letter
    direction: int | None      # serve: 4/5/6/0; rally shot: 0-3
    depth: int | None = None   # 7/8/9
    pre: str = ""              # modifiers directly after the letter (serve: '+' = serve and volley)
    mid: str = ""              # modifiers between direction and depth
    post: str = ""             # modifiers after depth
    error: str | None = None   # n w d x ! e (+ g for serve faults)
    outcome: str | None = None  # '*', '#', '@', 'C'
    raw_tail: str | None = None  # set when the token had typos; reproduced verbatim

    @property
    def irregular(self) -> bool:
        return self.raw_tail is not None

    @property
    def hitter(self) -> str:
        return "server" if self.index % 2 == 0 else "returner"

    @property
    def is_serve(self) -> bool:
        return self.stroke == "serve"

    @property
    def side(self) -> str | None:
        return stroke_side(self.stroke)

    @property
    def family(self) -> str:
        return stroke_family(self.stroke)

    @property
    def modifiers(self) -> str:
        return self.pre + self.mid + self.post

    @property
    def approach(self) -> bool:
        return not self.is_serve and "+" in self.modifiers

    @property
    def serve_and_volley(self) -> bool:
        return self.is_serve and "+" in self.modifiers

    @property
    def at_net(self) -> bool:
        return "-" in self.modifiers

    @property
    def at_baseline(self) -> bool:
        return "=" in self.modifiers

    @property
    def net_cord(self) -> bool:
        return ";" in self.modifiers

    def text(self) -> str:
        if self.is_serve:
            if self.raw_tail is not None:
                return self.raw_tail
            d = "" if self.direction is None else str(self.direction)
            return d + self.pre + (self.error or "") + (self.outcome or "")
        if self.raw_tail is not None:
            return self.stroke + self.raw_tail
        out = self.stroke + self.pre
        out += "" if self.direction is None else str(self.direction)
        out += self.mid
        out += "" if self.depth is None else str(self.depth)
        out += self.post + (self.error or "") + (self.outcome or "")
        return out


@dataclass
class ServeAttempt:
    raw: str
    lets: int = 0
    shots: list[Shot] = field(default_factory=list)
    code: str | None = None     # whole-attempt code: S R P Q V
    ok: bool = True
    problem: str | None = None
    embedded: "ServeAttempt | None" = None  # a second serve typed into the same cell

    @property
    def irregular(self) -> bool:
        return any(s.irregular for s in self.shots) or self.embedded is not None

    @property
    def serve(self) -> Shot | None:
        return self.shots[0] if self.shots else None

    @property
    def is_fault(self) -> bool:
        if self.code == "V":
            return True
        s = self.serve
        return bool(s and s.error and len(self.shots) == 1)

    @property
    def ended(self) -> str | None:
        """How this attempt ended, from the string alone."""
        if self.code:
            return {"S": "server_point", "R": "returner_point", "P": "penalty", "Q": "penalty",
                    "V": "fault"}[self.code]
        if not self.shots:
            return None
        if self.is_fault:
            return "fault"
        last = self.shots[-1]
        if last.is_serve:
            return {"*": "ace", "#": "unreturnable", "C": "interrupted"}.get(last.outcome or "")
        return OUTCOMES.get(last.outcome or "")

    @property
    def rally_length(self) -> int:
        """Contacts in this attempt, serve included (an ace or a fault is 1)."""
        return len(self.shots)

    def winner(self) -> str | None:
        """'server' / 'returner' implied by the string, None if it does not say."""
        e = self.ended
        if e in ("server_point", "ace", "unreturnable"):
            return "server"
        if e == "returner_point":
            return "returner"
        if e in ("winner", "forced_error", "unforced_error"):
            hitter = self.shots[-1].hitter
            if e == "winner":
                return hitter
            return "returner" if hitter == "server" else "server"
        return None


@dataclass
class PointParse:
    attempts: list[ServeAttempt]

    @property
    def ok(self) -> bool:
        return all(a.ok for a in self.attempts) and bool(self.attempts)

    @property
    def final(self) -> ServeAttempt:
        return self.attempts[-1]

    @property
    def serve_number(self) -> int:
        return len(self.attempts)

    @property
    def double_fault(self) -> bool:
        return len(self.attempts) == 2 and self.attempts[1].is_fault

    @property
    def rally_length(self) -> int:
        return self.final.rally_length

    @property
    def shots(self) -> list[Shot]:
        return self.final.shots

    def winner(self) -> str | None:
        if self.double_fault:
            return "returner"
        return self.final.winner()


class ParseError(ValueError):
    pass


_TAIL_CHARS = set("0123456789") | set(MODIFIERS) | set(RALLY_ERRORS) | set(OUTCOMES)
_CANONICAL_TAIL = re.compile(r"^([+\-=;^]*)([0-3]?)([+\-=;^]*)([7-9]?)([+\-=;^]*)([nwdx!e]?)([*#@C]?)$")
_SERVE_TAIL_CHARS = set(MODIFIERS) | set(SERVE_FAULTS) | set("*#@C")
_CANONICAL_SERVE_TAIL = re.compile(r"^(\+*)(?:([nwdxge!])|([*#C]))?$")


def _scan_shot(s: str, i: int, index: int) -> tuple[Shot, int]:
    n = len(s)
    letter = s[i]
    i += 1
    j = i
    while i < n and s[i] in _TAIL_CHARS:
        i += 1
    tail = s[j:i]
    shot = Shot(index=index, stroke=letter, direction=None)
    m = _CANONICAL_TAIL.match(tail)
    if m:
        pre, d, mid, dep, post, err, out = m.groups()
        shot.pre, shot.mid, shot.post = pre, mid, post
        shot.direction = int(d) if d else None
        shot.depth = int(dep) if dep else None
        shot.error = err or None
        shot.outcome = out or None
        return shot, i
    # Typos: extra or swapped digits, doubled error letters, outcome before error.
    shot.raw_tail = tail
    shot.direction = next((int(c) for c in tail if c in RALLY_DIRECTIONS), None)
    shot.depth = next((int(c) for c in tail if c in DEPTHS), None)
    shot.error = next((c for c in tail if c in RALLY_ERRORS), None)
    shot.outcome = next((c for c in tail if c in OUTCOMES), None)
    mods = "".join(c for c in tail if c in MODIFIERS)
    shot.pre = mods
    if shot.outcome and tail[-1] not in OUTCOMES and any(c in "0123456789" for c in tail[tail.index(shot.outcome):]):
        raise ParseError(f"rally continues after outcome in {letter + tail!r}")
    return shot, i


def parse_attempt(raw: str) -> ServeAttempt:
    """Parse one MCP serve-attempt string (a `1st` or `2nd` cell). Never raises."""
    s = (raw or "").strip()
    att = ServeAttempt(raw=s)
    try:
        _parse_into(att, s)
    except ParseError as exc:
        att.ok = False
        att.problem = str(exc)
    return att


def _parse_into(att: ServeAttempt, s: str) -> None:
    n = len(s)
    i = 0
    while i < n and s[i] == "c":
        att.lets += 1
        i += 1
    if i >= n:
        raise ParseError("empty attempt")
    if s[i:] in ATTEMPT_CODES:
        att.code = s[i:]
        return
    serve = Shot(index=0, stroke="serve", direction=None)
    start = i
    if s[i] in SERVE_DIRECTIONS:
        serve.direction = int(s[i])
        i += 1
    elif s[i] not in SERVE_FAULTS:
        raise ParseError(f"expected serve direction at {i}")
    j = i
    while i < n and s[i] in _SERVE_TAIL_CHARS:
        i += 1
    tail = s[j:i]
    att.shots.append(serve)
    m = _CANONICAL_SERVE_TAIL.match(tail)
    if m:
        serve.pre = m.group(1)
        serve.error = m.group(2)
        serve.outcome = m.group(3)
    else:
        serve.raw_tail = s[start:i]
        serve.pre = "".join(c for c in tail if c == "+")
        serve.error = next((c for c in tail if c in SERVE_FAULTS), None)
        if serve.error is None:
            serve.outcome = next((c for c in tail if c in "*#C"), None)
    if serve.error:
        rest = s[i:]
        if rest and len(rest) == 1 and rest in SERVE_DIRECTIONS and serve.direction is None:
            serve.direction = int(rest)          # "n4" for "4n"
            serve.raw_tail = s[start:]
            return
        if rest:
            k = 0
            while k < len(rest) and rest[k] == "c":
                k += 1
            if k < len(rest) and rest[k] in SERVE_DIRECTIONS:
                att.embedded = parse_attempt(rest)
                if not att.embedded.ok:
                    raise ParseError(f"bad embedded second serve: {att.embedded.problem}")
                return
            raise ParseError(f"trailing text after fault at {i}")
        return
    if serve.outcome:
        if i != n:
            raise ParseError(f"trailing text after serve outcome at {i}")
        return
    if serve.direction is None:
        raise ParseError("serve without direction")
    index = 1
    while i < n:
        if s[i] not in SHOT_LETTERS:
            raise ParseError(f"unexpected {s[i]!r} at {i}")
        shot, i = _scan_shot(s, i, index)
        att.shots.append(shot)
        index += 1
        if shot.outcome and i != n:
            raise ParseError(f"trailing text after outcome at {i}")


def parse_point(first: str, second: str = "") -> PointParse:
    a1 = parse_attempt(first)
    attempts = [a1]
    second = (second or "").strip()
    if a1.ok and a1.embedded is not None and not second:
        attempts.append(a1.embedded)
    elif a1.ok and a1.is_fault:
        if second:
            attempts.append(parse_attempt(second))
        else:
            a2 = ServeAttempt(raw="", ok=False, problem="first serve fault without a second serve")
            attempts.append(a2)
    return PointParse(attempts)


def serialize_attempt(att: ServeAttempt) -> str:
    out = "c" * att.lets
    if att.code:
        return out + att.code
    out += "".join(s.text() for s in att.shots)
    if att.embedded is not None:
        out += serialize_attempt(att.embedded)
    return out


def serialize_point(p: PointParse) -> tuple[str, str]:
    first = serialize_attempt(p.attempts[0])
    second = serialize_attempt(p.attempts[1]) if len(p.attempts) > 1 else ""
    return first, second


def build_attempt(shots: list[dict], lets: int = 0) -> ServeAttempt:
    """Build an attempt from simple dicts (used by the exporter).

    Each dict needs `stroke` ("serve" or a letter) and may carry direction, depth, pre, mid,
    post, error and outcome.
    """
    att = ServeAttempt(raw="", lets=lets)
    for k, d in enumerate(shots):
        att.shots.append(Shot(index=k, stroke=d["stroke"], direction=d.get("direction"),
                              depth=d.get("depth"), pre=d.get("pre", ""), mid=d.get("mid", ""),
                              post=d.get("post", ""), error=d.get("error"),
                              outcome=d.get("outcome")))
    att.raw = serialize_attempt(att)
    return att
