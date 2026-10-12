"""Match Charting Project (MCP) data: loaders, a grammar for point strings, tables.

The MCP point files describe each point with up to two *attempt strings*, the
``1st`` and ``2nd`` columns.  ``1st`` holds the first serve and, when it lands,
the whole rally; ``2nd`` holds the second serve and its rally after a fault.

Grammar (per attempt string)
----------------------------
Parsing is two-stage: :func:`lex` classifies every character into a lexeme
kind, and a recursive-descent parser assembles lexemes into :class:`Serve` and
:class:`Shot` tokens::

    attempt  := CODE                         S R P Q -- whole point, no rally
              | 'V'                          serve forfeited (time violation)
              | LET* serve shot* 'C'?
    serve    := SDIR? smod* ( fault ';'? '@'?  |  '*'  |  '#' )?
    SDIR     := 4 wide | 5 body | 6 T | 0 unknown
    smod     := '+' serve-and-volley | '^' underarm | ';' net cord
    fault    := [nwdxge!]{1,2}               n net  w wide  d deep  x wide+deep
                                             g foot fault  e unknown  ! shank
    shot     := LETTER mod* DIR? DEPTH? ERR{0,2} OUTCOME?
    LETTER   := f b r s v z o p u y l m h i j k t q
    mod      := '+' approach | '-' at net | '=' at baseline | ';' net cord
              | '^' drop/stop variant | '!' shank
    DIR      := 1 2 3 0                      (relative to a right-handed receiver)
    DEPTH    := 7 8 9                        shallow .. deep
    ERR      := n w d x ! e
    OUTCOME  := '*' winner | '#' forced error | '@' unforced error

Findings from the local AO + US Open files (521,347 non-empty strings after
de-duplication), where the data refines or contradicts the implementation guide
(section 4.1):

* ``V`` is *not* a whole-point code.  All 11 occurrences sit in ``1st`` with a
  non-empty ``2nd``: the first serve was forfeited for a time violation and the
  point continued on a second serve.  It is parsed as a non-contact fault.
* ``S``/``Q`` always go to the server and ``R``/``P`` always to the returner
  (checked against ``PtWinner``): ``P`` is a penalty against the server and
  ``Q`` a penalty against the returner (or an interrupted point awarded to the
  server).
* ``^`` after a serve digit marks an underarm serve (25 serves, all in matches
  of known underarm servers: Kyrgios, Bublik, Moutet, Dellien...).  On rally
  shots 99.4% of the 3,869 occurrences are on v/z/h/i/u/y: a drop (stop) volley
  or touch variant.
* ``;`` (net cord) on a serve sits *after* the fault letter (``4x;``, 517x);
  ``4;w`` (19x) is accepted and normalized.  An in-play serve never has ``;``
  (a net-cord serve that lands in is a let, ``c``).
* ``!`` appears both as an error letter after the direction (``f1!#``) and in
  the modifier slot before it (``f!3#``, ``f!2d#``); both are kept as written.
* ``c`` (let) only occurs at the start of an attempt string (one junk ``6cd``).
* ``C`` always ends the string: an unsuccessful challenge that stopped play.
  The last recorded hitter wins the point (verified against ``PtWinner``); the
  challenger sometimes touched the next ball without it being recorded.
* Fault letters may stand alone (``g`` 246x, ``n`` 42x, ``e`` 40x): a fault with
  no recorded direction.  Two fault letters occur (``4!n``, ``4wd``).  A
  redundant ``@`` sometimes follows a (double) fault (``4n@``).
* ``0`` as a serve direction is rare (565 strings).

Accepted variants that serialize to a different (canonical) spelling; the
point's ``canonical`` flag is False and ``warnings`` says what was normalized:
depth written before direction (``b72`` -> ``b27``; one charter's convention in
Nadal's AO 2014 matches), a modifier after the direction (``f1+`` -> ``f+1``),
an outcome before the error letter (``f@d`` -> ``fd@``), a net cord before a
serve fault letter (``4;w`` -> ``4w;``), a bare fault letter before its serve
digit (``n4`` -> ``4n``, ``e0`` -> ``0e``) and serve modifiers out of order
(``4+^`` -> ``4^+``).

Loader quirks (:func:`load_points`): rows of some matches are stored in shuffled
blocks (sorted by ``Pt`` here), and four matches are stored twice -- one exact
copy and three near-copies that differ in a few points (including the 2001 US
Open women's final); one whole copy is kept.

Everything else is an error (``ok=False``): unknown characters, a missing or
invalid serve digit, text after a fault or ace, two directions or two depths in
one shot, an outcome or error letter on a non-final shot, and inconsistent
columns (``2nd`` present after an in-play first serve, etc.).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_DATA_DIR",
    "Lexeme",
    "Serve",
    "Shot",
    "ParsedPoint",
    "lex",
    "parse_point",
    "parse_points",
    "serialize_point",
    "stroke_side",
    "stroke_family",
    "load_matches",
    "load_points",
    "shots_table",
    "points_table",
]

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "USO_MCP_DIR",
        Path(__file__).resolve().parents[2] / "data" / "match-charting",
    )
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SERVE_DIRECTIONS = {"4": "wide", "5": "body", "6": "T", "0": "unknown"}
SHOT_DIRECTIONS = {"1": "forehand_side", "2": "middle", "3": "backhand_side", "0": "unknown"}
DEPTHS = {"7": "shallow", "8": "middle", "9": "deep"}
SHOT_LETTERS = "fbrsvzopuylmhijktq"
FOREHAND_LETTERS = frozenset("frvoulhj")
BACKHAND_LETTERS = frozenset("bszpymik")
STROKE_FAMILIES = {
    "f": "groundstroke", "b": "groundstroke",
    "r": "slice", "s": "slice",
    "v": "volley", "z": "volley",
    "h": "half_volley", "i": "half_volley",
    "j": "swinging_volley", "k": "swinging_volley",
    "o": "overhead", "p": "overhead",
    "u": "drop", "y": "drop",
    "l": "lob", "m": "lob",
    "t": "trick",
    "q": "unknown",
}
FAULT_TYPES = {
    "n": "net", "w": "wide", "d": "deep", "x": "wide_and_deep",
    "g": "foot_fault", "e": "unknown", "!": "shank", "V": "time_violation",
}
ERROR_TYPES = {"n": "net", "w": "wide", "d": "deep", "x": "wide_and_deep", "!": "shank", "e": "unknown"}
OUTCOME_NAMES = {"*": "winner", "#": "forced_error", "@": "unforced_error"}
MODIFIER_NAMES = {"+": "approach", "-": "at_net", "=": "at_baseline", ";": "net_cord", "^": "drop", "!": "shank"}
# Whole-point codes and which role wins the point (verified against PtWinner).
POINT_CODES = {"S": "server", "R": "returner", "P": "returner", "Q": "server"}
POINT_OUTCOMES = ("ace", "service_winner", "winner", "forced_error", "unforced_error", "double_fault", "other")

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

LET, SERVE_DIR, ZERO, SHOT_DIR, DEPTH, SHOT, MOD, SHANK, ERR, FOOT, OUTCOME, CODE, VIOLATION, CHALLENGE, JUNK = (
    "let", "serve_dir", "zero", "shot_dir", "depth", "shot", "mod", "shank", "err", "foot_fault",
    "outcome", "code", "violation", "challenge", "junk",
)

_KIND: dict[str, str] = {"c": LET, "0": ZERO, "!": SHANK, "g": FOOT, "V": VIOLATION, "C": CHALLENGE}
_KIND.update({ch: SERVE_DIR for ch in "456"})
_KIND.update({ch: SHOT_DIR for ch in "123"})
_KIND.update({ch: DEPTH for ch in "789"})
_KIND.update({ch: SHOT for ch in SHOT_LETTERS})
_KIND.update({ch: MOD for ch in "+-=;^"})
_KIND.update({ch: ERR for ch in "nwdxe"})
_KIND.update({ch: OUTCOME for ch in "*#@"})
_KIND.update({ch: CODE for ch in POINT_CODES})


class Lexeme(NamedTuple):
    kind: str
    char: str
    pos: int


def lex(s: str) -> list[Lexeme]:
    """Classify every character of an attempt string (context-free)."""
    return [Lexeme(_KIND.get(ch, JUNK), ch, i) for i, ch in enumerate(s)]


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def stroke_side(letter: str) -> str:
    """'F' forehand, 'B' backhand, 'U' unknown (t, q, or anything else)."""
    if letter in FOREHAND_LETTERS:
        return "F"
    if letter in BACKHAND_LETTERS:
        return "B"
    return "U"


def stroke_family(letter: str) -> str:
    """groundstroke, slice, volley, half_volley, swinging_volley, overhead, drop, lob, trick or unknown."""
    return STROKE_FAMILIES.get(letter, "unknown")


@dataclass(frozen=True, slots=True)
class Serve:
    """One serve attempt.

    ``fault`` holds the fault letters as written (``'n'``, ``'g'``, ``'!n'``, ``'wd'``)
    or ``'V'`` for a serve forfeited to a time violation (no contact).  ``outcome``
    is ``'*'`` (ace) or ``'#'`` (unreturnable) on an in-play serve; on a fault it can
    be a redundant ``'@'``.  ``direction`` is ``'4'``/``'5'``/``'6'``/``'0'`` or None
    when no digit was written.  ``raw`` is the text as written (lets included); it
    is not part of equality, so a normalized spelling compares equal.
    """

    raw: str = field(compare=False)
    lets: int = 0
    direction: str | None = None
    serve_and_volley: bool = False
    fault: str | None = None
    outcome: str | None = None
    underarm: bool = False
    net_cord: bool = False

    @property
    def is_fault(self) -> bool:
        return self.fault is not None

    @property
    def is_violation(self) -> bool:
        return self.fault == "V"

    @property
    def is_contact(self) -> bool:
        """False only for a time-violation 'serve', which was never struck."""
        return self.fault != "V"

    @property
    def is_ace(self) -> bool:
        return self.fault is None and self.outcome == "*"

    @property
    def is_unreturnable(self) -> bool:
        return self.fault is None and self.outcome == "#"

    @property
    def direction_name(self) -> str | None:
        return SERVE_DIRECTIONS.get(self.direction) if self.direction else None

    @property
    def fault_type(self) -> str | None:
        """Primary fault type; a shank with a landing letter (``'!n'``) reports the landing."""
        if self.fault is None:
            return None
        letters = [ch for ch in self.fault if ch != "!"] or ["!"]
        if len(letters) > 1 and set(letters) == {"w", "d"}:
            return "wide_and_deep"
        return FAULT_TYPES.get(letters[0], "unknown")

    @property
    def modifiers(self) -> str:
        return ("^" if self.underarm else "") + ("+" if self.serve_and_volley else "") + (";" if self.net_cord else "")


@dataclass(frozen=True, slots=True)
class Shot:
    """One rally shot (any contact after the serve).

    ``modifiers`` keeps the characters written between the letter and the
    direction, in order (``'+'``, ``'-'``, ``'='``, ``';'``, ``'^'``, ``'!'``).
    ``error`` keeps the error letters as written (usually one; ``'w!'`` occurs).
    ``raw`` is the text as written and is not part of equality.
    """

    raw: str = field(compare=False)
    letter: str
    modifiers: str = ""
    direction: str | None = None
    depth: str | None = None
    error: str | None = None
    outcome: str | None = None

    @property
    def side(self) -> str:
        return stroke_side(self.letter)

    @property
    def family(self) -> str:
        return stroke_family(self.letter)

    @property
    def approach(self) -> bool:
        return "+" in self.modifiers

    @property
    def at_net(self) -> bool:
        return "-" in self.modifiers

    @property
    def at_baseline(self) -> bool:
        return "=" in self.modifiers

    @property
    def net_cord(self) -> bool:
        return ";" in self.modifiers

    @property
    def drop(self) -> bool:
        return "^" in self.modifiers

    @property
    def shank(self) -> bool:
        return "!" in self.modifiers or (self.error is not None and "!" in self.error)

    @property
    def error_type(self) -> str | None:
        """Landing error type; a shank written only as a modifier counts as 'shank'."""
        if self.error:
            letters = [ch for ch in self.error if ch != "!"] or ["!"]
            return ERROR_TYPES.get(letters[0], "unknown")
        if "!" in self.modifiers and self.outcome in ("#", "@"):
            return "shank"
        return None

    @property
    def direction_name(self) -> str | None:
        return SHOT_DIRECTIONS.get(self.direction) if self.direction else None

    @property
    def outcome_name(self) -> str | None:
        return OUTCOME_NAMES.get(self.outcome) if self.outcome else None


def _other(role: str) -> str:
    return "returner" if role == "server" else "server"


@dataclass(slots=True)
class ParsedPoint:
    """A parsed point: serve attempts plus the rally that followed the in-play serve.

    ``shots`` excludes the serve.  Hitters alternate: the in-play serve is the
    server's, ``shots[0]`` (the return) the returner's, ``shots[1]`` the server's...
    ``ok`` is False when any grammar error was found (see ``errors``, prefixed
    with ``1st:``, ``2nd:`` or ``point:``).  ``canonical`` is False when an accepted
    variant spelling was normalized (``serialize_point`` then differs from input).
    """

    serves: list[Serve] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    special: str | None = None
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    challenge: bool = False
    canonical: bool = True

    # -- serve structure ---------------------------------------------------
    @property
    def n_serve_attempts(self) -> int:
        """Serve attempts recorded (a time-violation forfeit counts as an attempt)."""
        return len(self.serves)

    @property
    def in_serve_index(self) -> int | None:
        """1-based number (1 or 2) of the attempt that landed in play.

        None for a double fault, a whole-point code, or an unparsed point.
        """
        if not self.ok or self.special:
            return None
        for k, sv in enumerate(self.serves, 1):
            if not sv.is_fault:
                return k
        return None

    @property
    def in_serve(self) -> Serve | None:
        k = self.in_serve_index
        return self.serves[k - 1] if k else None

    @property
    def fault_serves(self) -> list[Serve]:
        """Fault attempts that were actually struck (time violations excluded)."""
        return [sv for sv in self.serves if sv.is_fault and sv.is_contact]

    @property
    def is_double_fault(self) -> bool:
        return self.ok and not self.special and len(self.serves) == 2 and all(sv.is_fault for sv in self.serves)

    # -- rally structure ---------------------------------------------------
    @property
    def n_contacts(self) -> int | None:
        """Contacts in the final rally including the in-play serve.

        ace / unreturnable serve -> 1, return error -> 2, double fault -> 0.
        None when unknown (unparsed point or a whole-point code S/R/P/Q).
        An unreturnable serve (``#``) counts 1 by MCP convention even though the
        returner usually touched the ball.
        """
        if not self.ok or self.special:
            return None
        if self.is_double_fault:
            return 0
        return 1 + len(self.shots)

    @property
    def contact_roles(self) -> list[str]:
        """Hitter role of each contact in the final rally (serve first)."""
        n = self.n_contacts or 0
        return ["server" if k % 2 == 0 else "returner" for k in range(n)]

    @property
    def last_hitter_role(self) -> str | None:
        """Role of the last player to strike the ball (a double fault -> 'server')."""
        if not self.ok or self.special:
            return None
        if self.is_double_fault:
            return "server"
        return "server" if len(self.shots) % 2 == 0 else "returner"

    @property
    def outcome(self) -> str:
        """ace, service_winner, winner, forced_error, unforced_error, double_fault or other."""
        if not self.ok or self.special:
            return "other"
        if self.is_double_fault:
            return "double_fault"
        if self.challenge:
            return "other"
        if not self.shots:
            sv = self.in_serve
            if sv is not None and sv.outcome == "*":
                return "ace"
            if sv is not None and sv.outcome == "#":
                return "service_winner"
            return "other"
        last = self.shots[-1].outcome
        return {"*": "winner", "#": "forced_error", "@": "unforced_error"}.get(last or "", "other")

    @property
    def winner_role(self) -> str | None:
        """Point winner implied by the notation alone (None when it cannot be told)."""
        if self.special:
            return POINT_CODES[self.special]
        if not self.ok:
            return None
        if self.is_double_fault:
            return "returner"
        last = self.last_hitter_role
        out = self.outcome
        if out in ("ace", "service_winner", "winner") or self.challenge:
            return last
        if out in ("forced_error", "unforced_error"):
            return _other(last)
        if self.shots and self.shots[-1].error:  # error letter without outcome symbol
            return _other(last)
        return None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Attempt:
    special: str | None = None
    serve: Serve | None = None
    shots: tuple[Shot, ...] = ()
    challenge: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    canonical: bool = True


class _AttemptParser:
    """Recursive-descent parser over the lexemes of one attempt string."""

    def __init__(self, s: str):
        self.s = s
        self.lx = lex(s)
        self.n = len(self.lx)
        self.i = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.canonical = True

    # helpers
    def kind(self) -> str | None:
        return self.lx[self.i].kind if self.i < self.n else None

    def char(self) -> str:
        return self.lx[self.i].char if self.i < self.n else ""

    def error(self, msg: str) -> None:
        self.errors.append(f"{msg} at {self.i} in {self.s!r}")

    def normalize(self, msg: str) -> None:
        self.warnings.append(f"normalized: {msg}")
        self.canonical = False

    # grammar
    def parse(self) -> _Attempt:
        s = self.s
        if not s:
            return _Attempt(errors=("empty attempt string",))
        if s in POINT_CODES:
            return _Attempt(special=s)
        if s == "V":
            return _Attempt(serve=Serve(raw="V", fault="V"))
        serve = self.serve()
        if serve is None:
            return self.result(None, ())
        shots: list[Shot] = []
        challenge = False
        terminal = serve.is_fault or serve.outcome in ("*", "#")
        if terminal and self.i < self.n:
            self.error("unexpected text after a fault/ace/unreturnable serve")
            return self.result(serve, ())
        while self.i < self.n:
            k = self.kind()
            if k == CHALLENGE:
                if self.i == self.n - 1:
                    challenge = True
                    self.i += 1
                    break
                self.error("'C' not at end")
                break
            if k != SHOT:
                self.error(f"expected a shot letter, got {self.char()!r}")
                break
            shot = self.shot()
            if shot is None:
                break
            shots.append(shot)
        if not self.errors:
            for j, sh in enumerate(shots[:-1]):
                if sh.outcome:
                    self.errors.append(f"outcome on non-final shot {j + 1} ({sh.raw!r}) in {s!r}")
                elif sh.error:
                    self.errors.append(f"error letter on non-final shot {j + 1} ({sh.raw!r}) in {s!r}")
            if shots and shots[-1].outcome is None and not challenge:
                self.warnings.append("incomplete: final shot has no outcome")
            if not shots and not serve.is_fault and serve.outcome is None and not challenge:
                self.warnings.append("incomplete: in-play serve with no rally")
        return self.result(serve, tuple(shots), challenge)

    def result(self, serve: Serve | None, shots: tuple[Shot, ...], challenge: bool = False) -> _Attempt:
        return _Attempt(
            serve=serve,
            shots=shots,
            challenge=challenge,
            errors=tuple(self.errors),
            warnings=tuple(self.warnings),
            canonical=self.canonical,
        )

    def serve(self) -> Serve | None:
        lets = 0
        while self.kind() == LET:
            lets += 1
            self.i += 1
        direction = None
        if self.kind() in (SERVE_DIR, ZERO):
            direction = self.char()
            self.i += 1
        elif self.kind() in (ERR, FOOT) and self.i + 2 == self.n and self.lx[self.i + 1].kind in (SERVE_DIR, ZERO):
            # 'n4', 'e0': a bare fault written before its serve digit
            fault, direction = self.char(), self.lx[self.i + 1].char
            self.i += 2
            self.normalize("fault letter written before the serve digit")
            return Serve(raw=self.s, lets=lets, direction=direction, fault=fault)
        elif self.kind() not in (ERR, FOOT, SHANK):
            self.error(f"expected a serve direction (4/5/6/0), got {self.char()!r}")
            return None
        mods = ""
        while self.kind() == MOD and self.char() in "+^;":
            mods += self.char()
            self.i += 1
        sv_and_volley, underarm, net_cord = "+" in mods, "^" in mods, ";" in mods
        canonical_mods = ("^" if underarm else "") + ("+" if sv_and_volley else "")
        fault = ""
        while self.kind() in (ERR, FOOT, SHANK):
            fault += self.char()
            self.i += 1
        outcome = None
        if fault:
            if len(fault) > 2:
                self.error(f"too many fault letters {fault!r}")
                return None
            if ";" in mods:
                self.normalize("net cord before the serve fault letter")
            elif mods != canonical_mods:
                self.normalize("serve modifiers reordered")
            if self.kind() == MOD and self.char() == ";":
                net_cord = True
                self.i += 1
            if self.kind() == OUTCOME and self.char() == "@":
                outcome = "@"
                self.i += 1
        else:
            if mods != canonical_mods + (";" if net_cord else ""):
                self.normalize("serve modifiers reordered")
            if direction is None:
                self.error("serve has neither direction nor fault")
                return None
            if self.kind() == OUTCOME:
                if self.char() == "@":
                    self.error("'@' on an in-play serve")
                    return None
                outcome = self.char()
                self.i += 1
        return Serve(
            raw=self.s[:self.i],
            lets=lets,
            direction=direction,
            serve_and_volley=sv_and_volley,
            fault=fault or None,
            outcome=outcome,
            underarm=underarm,
            net_cord=net_cord,
        )

    def shot(self) -> Shot | None:
        start = self.i
        letter = self.char()
        self.i += 1
        mods = ""
        while self.kind() in (MOD, SHANK):
            mods += self.char()
            self.i += 1
        direction = depth = None
        depth_first = False
        while self.kind() in (SHOT_DIR, ZERO, DEPTH, SERVE_DIR):
            k, ch = self.kind(), self.char()
            if k == SERVE_DIR:
                self.error(f"serve digit {ch!r} inside a rally shot")
                return None
            if k == DEPTH:
                if depth is not None:
                    self.error("two depth digits in one shot")
                    return None
                depth = ch
                if direction is None:
                    depth_first = True
            else:
                if direction is not None:
                    self.error("two direction digits in one shot")
                    return None
                direction = ch
            self.i += 1
        if depth_first and direction is not None:
            self.normalize("depth written before direction")
        post = ""
        while self.kind() == MOD:
            post += self.char()
            self.i += 1
        if post:  # only reachable after a digit: the first loop consumed leading modifiers
            mods += post
            self.normalize("modifier written after direction/depth")
        if self.kind() in (SHOT_DIR, ZERO, DEPTH, SERVE_DIR):
            self.error("digit after a modifier that followed the direction")
            return None
        error = ""
        while self.kind() in (ERR, SHANK):
            error += self.char()
            self.i += 1
        outcome = None
        if self.kind() == OUTCOME:
            outcome = self.char()
            self.i += 1
            if self.kind() in (ERR, SHANK):
                if error:
                    self.error("error letters on both sides of the outcome")
                    return None
                while self.kind() in (ERR, SHANK):
                    error += self.char()
                    self.i += 1
                self.normalize("outcome written before the error letter")
        if len(error) > 2:
            self.error(f"too many error letters {error!r}")
            return None
        return Shot(
            raw=self.s[start:self.i],
            letter=letter,
            modifiers=mods,
            direction=direction,
            depth=depth,
            error=error or None,
            outcome=outcome,
        )


@lru_cache(maxsize=1 << 16)
def _parse_attempt(s: str) -> _Attempt:
    return _AttemptParser(s).parse()


def _clean(s: object) -> str:
    if s is None or s is pd.NA or (isinstance(s, float) and np.isnan(s)):
        return ""
    return str(s).strip()


def parse_point(first: str, second: str | None = None) -> ParsedPoint:
    """Parse one point from its ``1st`` and ``2nd`` strings (whitespace is stripped)."""
    f, s2 = _clean(first), _clean(second)
    a1 = _parse_attempt(f)
    errors = [f"1st: {e}" for e in a1.errors]
    warnings = [f"1st: {w}" for w in a1.warnings]
    canonical = a1.canonical
    if a1.special:
        if s2:
            errors.append("point: 2nd column present after a whole-point code")
        return ParsedPoint(special=a1.special, ok=not errors, errors=errors, warnings=warnings, canonical=canonical)
    serves = [a1.serve] if a1.serve is not None else []
    shots = list(a1.shots)
    challenge = a1.challenge
    first_failed_or_fault = a1.serve is None or a1.serve.is_fault or bool(a1.errors)
    if first_failed_or_fault and s2:
        a2 = _parse_attempt(s2)
        errors += [f"2nd: {e}" for e in a2.errors]
        warnings += [f"2nd: {w}" for w in a2.warnings]
        canonical = canonical and a2.canonical
        if a2.special:
            errors.append("2nd: whole-point code in the 2nd column")
        if a2.serve is not None:
            serves.append(a2.serve)
            shots = list(a2.shots)
            challenge = a2.challenge
    elif s2:
        errors.append(f"point: 2nd column {s2!r} present but the first serve was in play")
    elif a1.serve is not None and a1.serve.is_fault:
        errors.append("point: first serve was a fault but the 2nd column is empty")
    return ParsedPoint(
        serves=serves,
        shots=shots,
        special=None,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        challenge=challenge,
        canonical=canonical,
    )


def parse_points(points_df: pd.DataFrame) -> list[ParsedPoint]:
    """Parse every row of a points frame (``1st``/``2nd`` columns), in row order."""
    return [parse_point(a, b) for a, b in zip(points_df["1st"], points_df["2nd"])]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_serve(sv: Serve) -> str:
    if sv.fault == "V":
        return "V"
    out = "c" * sv.lets + (sv.direction or "")
    out += ("^" if sv.underarm else "") + ("+" if sv.serve_and_volley else "")
    if sv.fault:
        out += sv.fault + (";" if sv.net_cord else "")
    elif sv.net_cord:
        out += ";"
    return out + (sv.outcome or "")


def _serialize_shot(sh: Shot) -> str:
    return (
        sh.letter
        + sh.modifiers
        + (sh.direction or "")
        + (sh.depth or "")
        + (sh.error or "")
        + (sh.outcome or "")
    )


def serialize_point(point: ParsedPoint) -> tuple[str, str]:
    """Write a point back to MCP ``(1st, 2nd)`` strings in canonical spelling.

    ``serialize_point(parse_point(a, b)) == (a.strip(), b.strip())`` for every
    canonical input; normalized variants come back in canonical form.
    """
    if point.special:
        return point.special, ""
    strings: list[str] = []
    for sv in point.serves:
        s = _serialize_serve(sv)
        if not sv.is_fault:
            s += "".join(_serialize_shot(sh) for sh in point.shots) + ("C" if point.challenge else "")
            strings.append(s)
            break
        strings.append(s)
    first = strings[0] if strings else ""
    second = strings[1] if len(strings) > 1 else ""
    return first, second


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

POINT_FILE_PATTERNS = (
    "charting-{g}-points-to-2009.csv",
    "charting-{g}-points-2010s.csv",
    "charting-{g}-points-2020s.csv",
)
_POINT_INT_COLUMNS = ("Pt", "Set1", "Set2", "Gm1", "Gm2", "Gm#", "Svr", "PtWinner")


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def load_matches(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Both genders' match lists, text stripped, plus ``gender``, ``year``, ``date``, ``best_of``."""
    data_dir = Path(data_dir)
    frames = []
    for g in ("m", "w"):
        path = data_dir / f"charting-{g}-matches.csv"
        if path.exists():
            df = _read_csv(path)
            df.insert(1, "gender", g)
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no charting-*-matches.csv in {data_dir}")
    m = pd.concat(frames, ignore_index=True)
    m["year"] = pd.to_numeric(m["Date"].str[:4], errors="coerce").astype("Int64")
    m["date"] = pd.to_datetime(m["Date"], format="%Y%m%d", errors="coerce")
    m["best_of"] = pd.to_numeric(m["Best of"], errors="coerce").astype("Int64")
    return m


def _copy_score(df: pd.DataFrame) -> tuple[int, int]:
    """Information content of one copy of a duplicated match (higher is better)."""
    first = df["1st"]
    n_special = int(first.isin(list(POINT_CODES)).sum() + (first == "e").sum())
    n_chars = int(first.str.len().sum() + df["2nd"].str.len().sum())
    return (-n_special, n_chars)


def load_points(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    match_ids: Iterable[str] | None = None,
    dedupe: bool = True,
) -> pd.DataFrame:
    """All point rows (both genders, all decades), text stripped, integer columns typed.

    Rows are sorted by ``(match_id, Pt)``: some matches have their point blocks
    out of order in the files.  With ``dedupe`` (default) exact duplicate rows are
    dropped, and for the few matches stored twice with small differences the more
    informative copy is kept (fewer S/R/'e' placeholders, then more characters).
    What was dropped is recorded in ``df.attrs['dropped_copies']``.
    """
    data_dir = Path(data_dir)
    wanted = set(match_ids) if match_ids is not None else None
    frames = []
    for g in ("m", "w"):
        for pat in POINT_FILE_PATTERNS:
            path = data_dir / pat.format(g=g)
            if not path.exists():
                continue
            df = _read_csv(path)
            if wanted is not None:
                df = df[df["match_id"].isin(wanted)]
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no charting-*-points-*.csv in {data_dir}")
    p = pd.concat(frames, ignore_index=True)
    for c in _POINT_INT_COLUMNS:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("Int64")
    p["TbSet"] = p["TbSet"].map({"True": True, "False": False}).astype("boolean")
    dropped: dict[str, int] = {}
    if dedupe:
        # A match stored twice: the k-th occurrence (in file order) of each Pt belongs
        # to copy k.  Keep one whole copy -- never mix rows from different copies.
        p = p.reset_index(drop=True)
        dup_ids = p.loc[p.duplicated(["match_id", "Pt"]), "match_id"].unique()
        keep = np.ones(len(p), dtype=bool)
        for mid in dup_ids:
            idx = np.flatnonzero((p["match_id"] == mid).to_numpy())
            sub = p.iloc[idx]
            copy_no = sub.groupby("Pt").cumcount().to_numpy()
            best = max(range(copy_no.max() + 1), key=lambda k: (_copy_score(sub[copy_no == k]), -k))
            keep[idx[copy_no != best]] = False
            dropped[mid] = int((copy_no != best).sum())
        p = p[keep]
        n0 = len(p)
        p = p.drop_duplicates()
        p.attrs["dropped_exact_duplicates"] = n0 - len(p)
    p = p.sort_values(["match_id", "Pt"], kind="stable").reset_index(drop=True)
    p.attrs["dropped_copies"] = dropped
    return p


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _match_lookup(matches_df: pd.DataFrame) -> dict[str, tuple[str, str, str, str]]:
    def hand(h: str) -> str:
        return h if h in ("R", "L") else "U"

    return {
        r[0]: (r[1], r[2], hand(r[3]), hand(r[4]))
        for r in matches_df[["match_id", "Player 1", "Player 2", "Pl 1 hand", "Pl 2 hand"]].itertuples(index=False)
    }


SHOT_COLUMNS = [
    "match_id", "Pt", "shot_no", "is_serve", "is_fault_serve", "serve_attempt", "hitter_role", "hitter",
    "hitter_name", "hitter_hand", "letter", "side", "family", "direction", "depth", "modifiers", "error",
    "outcome", "is_last_contact", "n_contacts", "raw",
]


def shots_table(points_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """One row per contact in each point's final rally, plus one row per struck fault.

    The in-play serve is ``shot_no`` 1 (``letter`` 'serve', ``family`` 'serve',
    ``side`` None, ``direction`` its serve digit); rally shots follow as 2, 3, ...
    Fault serves are extra rows with ``shot_no`` 0 and ``is_fault_serve`` True
    (``error`` holds the fault letters) so contact models can learn them too;
    time-violation forfeits (``V``) are not contacts and get no row.  Points that
    fail to parse and whole-point codes (S/R/P/Q) contribute no rows.
    ``hitter`` is 1/2 for MCP Player 1/2, derived from ``Svr`` and alternation.
    """
    lookup = _match_lookup(matches_df)
    rows: list[tuple] = []
    for mid, pt, svr, a, b in zip(
        points_df["match_id"], points_df["Pt"], points_df["Svr"], points_df["1st"], points_df["2nd"]
    ):
        pp = parse_point(a, b)
        if not pp.ok or pp.special or pd.isna(svr):
            continue
        svr = int(svr)
        ret = 3 - svr
        names = lookup.get(mid, (None, None, "U", "U"))
        sname, rname = names[svr - 1], names[ret - 1]
        shand, rhand = names[svr + 1], names[ret + 1]
        n_contacts = pp.n_contacts
        for k, sv in enumerate(pp.serves, 1):
            if not sv.is_fault:
                continue
            if not sv.is_contact:
                continue
            rows.append((mid, pt, 0, True, True, k, "server", svr, sname, shand, "serve", None, "serve",
                         sv.direction, None, sv.modifiers, sv.fault, sv.outcome, False, n_contacts, sv.raw))
        k_in = pp.in_serve_index
        if k_in is None:
            continue
        sv = pp.serves[k_in - 1]
        last = len(pp.shots)
        rows.append((mid, pt, 1, True, False, k_in, "server", svr, sname, shand, "serve", None, "serve",
                     sv.direction, None, sv.modifiers, None, sv.outcome, last == 0, n_contacts, sv.raw))
        for j, sh in enumerate(pp.shots):
            is_ret = j % 2 == 0
            rows.append((
                mid, pt, j + 2, False, False, None,
                "returner" if is_ret else "server",
                ret if is_ret else svr,
                rname if is_ret else sname,
                rhand if is_ret else shand,
                sh.letter, stroke_side(sh.letter), stroke_family(sh.letter),
                sh.direction, sh.depth, sh.modifiers, sh.error, sh.outcome, j == last - 1, n_contacts, sh.raw,
            ))
    out = pd.DataFrame.from_records(rows, columns=SHOT_COLUMNS)
    out["serve_attempt"] = out["serve_attempt"].astype("Int64")
    out["n_contacts"] = out["n_contacts"].astype("Int64")
    return out


POINT_COLUMNS = [
    "match_id", "Pt", "server", "returner", "server_name", "returner_name",
    "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", "TbSet",
    "n_serve_attempts", "in_serve", "in_serve_direction", "is_double_fault", "n_contacts", "outcome",
    "last_hitter_role", "special", "challenge", "winner_role", "PtWinner", "notation_winner_agrees",
    "parse_ok", "canonical", "parse_errors",
]


def points_table(points_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """One row per point: server/returner, MCP score columns and parsed point summary.

    ``winner_role`` is the winner implied by the notation alone; ``notation_winner_agrees``
    compares it with ``PtWinner`` (NA when the notation does not determine it).
    """
    lookup = _match_lookup(matches_df)
    rows: list[tuple] = []
    cols = ["match_id", "Pt", "Svr", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", "TbSet", "1st", "2nd", "PtWinner"]
    for mid, pt, svr, s1, s2, g1, g2, pts, gno, tb, a, b, win in zip(*(points_df[c] for c in cols)):
        pp = parse_point(a, b)
        names = lookup.get(mid, (None, None, "U", "U"))
        if pd.isna(svr):
            server = returner = None
            sname = rname = None
        else:
            server, returner = int(svr), 3 - int(svr)
            sname, rname = names[server - 1], names[returner - 1]
        wr = pp.winner_role
        if wr is None or server is None or pd.isna(win):
            agrees = None
        else:
            agrees = (server if wr == "server" else returner) == int(win)
        sv = pp.in_serve
        rows.append((
            mid, pt, server, returner, sname, rname, s1, s2, g1, g2, pts, gno, tb,
            pp.n_serve_attempts, pp.in_serve_index, sv.direction if sv else None, pp.is_double_fault,
            pp.n_contacts, pp.outcome, pp.last_hitter_role, pp.special, pp.challenge, wr, win, agrees,
            pp.ok, pp.canonical, "; ".join(pp.errors),
        ))
    out = pd.DataFrame.from_records(rows, columns=POINT_COLUMNS)
    for c in ("server", "returner", "in_serve", "n_contacts", "Set1", "Set2", "Gm1", "Gm2", "Gm#", "PtWinner"):
        out[c] = pd.array(out[c], dtype="Int64")
    for c in ("TbSet", "notation_winner_agrees"):
        out[c] = pd.array(out[c], dtype="boolean")
    return out
