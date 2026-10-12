"""Deterministic tennis scoring: games, sets, tiebreaks, server, side and ends.

The :class:`Scorer` state machine is driven by the sequence of point winners.
Before every point it reports who serves, from which court ('deuce'/'ad'), the
score, and whether the players are at their starting ends.

Rules implemented (ITF), with event-specific final sets (:func:`final_set_rule`):

* Games: 0, 15, 30, 40, deuce/advantage.  Sets: first to 6 games by two, with
  a tiebreak at 6-6 unless the set is an advantage set.
* Serving: alternates every game.  Within a game the first point is served from
  the deuce court and the court alternates every point.  In a (standard)
  tiebreak the player due to serve serves point 1 (deuce court), then players
  alternate every two points, each serving first from the ad court.  The
  tiebreak counts as one game for the alternation, so the player who received
  first in the tiebreak serves the first game of the next set.
* Ends: change after every game whose index within the set is odd (1, 3, 5,
  ...).  This also gives the set-end rule (change at set end iff the set had an
  odd number of games; a tiebreak counts as one game, so 7-6 changes).  Within
  a tiebreak, change after every 6 points while the tiebreak continues.

Event history (verified against the MCP score columns, see :func:`validate_all`):

* US Open: no tiebreaks before 1970; 1970-74 a 9-point "sudden death"
  tiebreak (first to 5, serve rotating in pairs from the first point: A A B B A
  A B B ..., visible in the 1971/1972 finals); 7-point tiebreaks in every set
  from 1975 to 2021; from 2022 a 10-point tiebreak at 6-6 in the final set.
* Australian Open: 7-point tiebreaks in non-final sets (from 1975 in the data);
  final set an advantage set before 2019; 10-point tiebreak at 6-6 from 2019
  (the AO's own rule 2019-21, the common Grand Slam rule from 2022 -- same
  format).

MCP column conventions (verified over all local matches): ``Pts`` is
server-first ('15-0' after the server wins the first point; tiebreak points are
plain counts, also server-first; deuce is '40-40', advantages 'AD-40'/'40-AD');
``Set1``/``Set2`` and ``Gm1``/``Gm2`` are MCP Player 1/2 sets and games in the
current set, before the point; ``Gm#`` is the 1-based game number within the
match (a tiebreak is one game); ``TbSet`` is True when the current set is played
with a tiebreak at 6-6.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "SetFormat",
    "MatchFormat",
    "ADVANTAGE_SET",
    "TIEBREAK_7",
    "TIEBREAK_10",
    "SUDDEN_DEATH_9",
    "final_set_rule",
    "regular_set_rule",
    "match_format",
    "MatchOver",
    "Scorer",
    "annotate_points",
    "mcp_state",
    "validate_match",
    "validate_all",
    "grand_slam_best_of",
    "format_pts",
]

# ---------------------------------------------------------------------------
# Formats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetFormat:
    """How a set is decided.

    ``tiebreak_at=None`` is an advantage set.  Otherwise a tiebreak is played at
    ``tiebreak_at``-all, won by the first to ``tiebreak_points`` with a margin of
    ``tiebreak_margin`` (1 = sudden death).  ``tiebreak_first_block`` is how many
    points the first tiebreak server serves before the pairs start (1 standard,
    2 for the 1970s sudden-death tiebreak); ends change every
    ``tiebreak_change_every`` tiebreak points.
    """

    games: int = 6
    tiebreak_at: int | None = 6
    tiebreak_points: int = 7
    tiebreak_margin: int = 2
    tiebreak_first_block: int = 1
    tiebreak_change_every: int = 6

    @property
    def has_tiebreak(self) -> bool:
        return self.tiebreak_at is not None


ADVANTAGE_SET = SetFormat(tiebreak_at=None)
TIEBREAK_7 = SetFormat()
TIEBREAK_10 = SetFormat(tiebreak_points=10)
# Van Alen's 9-point tiebreak (US Open 1970-74).  The ends rule is not visible in
# MCP columns; 4 points follows the original description and is unverified.
SUDDEN_DEATH_9 = SetFormat(tiebreak_points=5, tiebreak_margin=1, tiebreak_first_block=2, tiebreak_change_every=4)


def _event(tournament: str) -> str:
    t = str(tournament).strip().lower().replace("_", " ").replace("-", " ")
    if t in ("us open", "uso", "usopen", "u.s. open"):
        return "uso"
    if t in ("australian open", "ao", "australianopen"):
        return "ao"
    raise ValueError(f"unknown tournament {tournament!r} (expected 'US Open' or 'Australian Open')")


def final_set_rule(tournament: str, year: int) -> SetFormat:
    """Format of the deciding set (5th for best-of-5, 3rd for best-of-3)."""
    ev, year = _event(tournament), int(year)
    if ev == "uso":
        if year < 1970:
            return ADVANTAGE_SET
        if year < 1975:
            return SUDDEN_DEATH_9
        return TIEBREAK_7 if year < 2022 else TIEBREAK_10
    # Australian Open
    return ADVANTAGE_SET if year < 2019 else TIEBREAK_10


def regular_set_rule(tournament: str, year: int) -> SetFormat:
    """Format of every non-deciding set."""
    ev, year = _event(tournament), int(year)
    if ev == "uso":
        if year < 1970:
            return ADVANTAGE_SET
        return SUDDEN_DEATH_9 if year < 1975 else TIEBREAK_7
    return ADVANTAGE_SET if year < 1971 else TIEBREAK_7


@dataclass(frozen=True)
class MatchFormat:
    best_of: int = 3
    regular: SetFormat = TIEBREAK_7
    final: SetFormat = TIEBREAK_7

    def __post_init__(self) -> None:
        if self.best_of not in (1, 3, 5):
            raise ValueError(f"best_of must be 1, 3 or 5, got {self.best_of}")

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1

    def set_format(self, set_no: int) -> SetFormat:
        return self.final if set_no == self.best_of else self.regular


def match_format(tournament: str, year: int, best_of: int) -> MatchFormat:
    return MatchFormat(int(best_of), regular_set_rule(tournament, year), final_set_rule(tournament, year))


QUALIFYING_ROUNDS = ("Q1", "Q2", "Q3", "Q4")


def grand_slam_best_of(gender: str, round_: str) -> int:
    """Best-of for Grand Slam singles: men's main draw 5, men's qualifying and women 3.

    The MCP match file's ``Best of`` disagrees with this for 22 men's main-draw
    matches listed as best of 3 (e.g. the 2011 US Open final) and one women's
    match listed as best of 5; the points themselves follow this rule.
    """
    g = str(gender).strip().lower()[:1]
    if g == "m" and str(round_).strip().upper() not in QUALIFYING_ROUNDS:
        return 5
    return 3


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

_GAME_POINT_NAMES = ("0", "15", "30", "40")


def format_pts(a: int, b: int, in_tiebreak: bool) -> str:
    """Game score string, MCP style, with ``a`` the points of the player written first."""
    if in_tiebreak:
        return f"{a}-{b}"
    if a >= 3 and b >= 3:
        if a == b:
            return "40-40"
        return "AD-40" if a > b else "40-AD"
    return f"{_GAME_POINT_NAMES[a]}-{_GAME_POINT_NAMES[b]}"


class MatchOver(RuntimeError):
    pass


class Scorer:
    """Tennis scoring state machine.  Players are 1 and 2.

    Read the properties for the state *before* the next point, then call
    :meth:`point` with its winner.
    """

    def __init__(self, fmt: MatchFormat, first_server: int):
        if first_server not in (1, 2):
            raise ValueError("first_server must be 1 or 2")
        self.fmt = fmt
        self.sets = [0, 0]
        self.games = [0, 0]
        self.points = [0, 0]  # points in the current game or tiebreak
        self.game_server = first_server  # server of the current game (first server of a tiebreak)
        self.in_tiebreak = False
        self.set_no = 1
        self.game_no_in_set = 1
        self.game_no_in_match = 1
        self.ends_swapped = 0  # 0: both players at their starting ends
        self.winner: int | None = None
        self.n_points = 0

    # -- derived state --------------------------------------------------------
    @property
    def set_format(self) -> SetFormat:
        return self.fmt.set_format(self.set_no)

    @property
    def points_played_in_game(self) -> int:
        return self.points[0] + self.points[1]

    @property
    def point_no_in_game(self) -> int:
        return self.points_played_in_game + 1

    @property
    def server(self) -> int:
        if not self.in_tiebreak:
            return self.game_server
        k = self.points_played_in_game
        block = self.set_format.tiebreak_first_block
        # standard: A | B B | A A | ...  (block 1);  sudden death: A A | B B | ... (block 2)
        first_has_it = (k < block) or ((k - block) // 2) % 2 == 1
        return self.game_server if first_has_it else 3 - self.game_server

    @property
    def returner(self) -> int:
        return 3 - self.server

    @property
    def side(self) -> str:
        return "deuce" if self.points_played_in_game % 2 == 0 else "ad"

    @property
    def is_final_set(self) -> bool:
        return self.set_no == self.fmt.best_of

    @property
    def match_over(self) -> bool:
        return self.winner is not None

    def pts_string(self, order: str = "server") -> str:
        """Game score in MCP style; ``order`` 'server' (MCP ``Pts``) or 'p1'."""
        first = self.server if order == "server" else 1
        return format_pts(self.points[first - 1], self.points[2 - first], self.in_tiebreak)

    def snapshot(self) -> dict:
        server = self.server
        return {
            "set_no": self.set_no,
            "game_no_in_set": self.game_no_in_set,
            "game_no_in_match": self.game_no_in_match,
            "point_no_in_game": self.point_no_in_game,
            "in_tiebreak": self.in_tiebreak,
            "tb_set": self.set_format.has_tiebreak,
            "final_set": self.is_final_set,
            "server": server,
            "returner": 3 - server,
            "side": self.side,
            "sets_p1": self.sets[0],
            "sets_p2": self.sets[1],
            "games_p1": self.games[0],
            "games_p2": self.games[1],
            "points_p1": self.points[0],
            "points_p2": self.points[1],
            "pts": self.pts_string("server"),
            "game_score": f"{self.games[0]}-{self.games[1]}",
            "set_score": f"{self.sets[0]}-{self.sets[1]}",
            "ends_swapped_before_point": self.ends_swapped,
        }

    # -- transitions ----------------------------------------------------------
    def point(self, winner: int) -> str | None:
        """Record a point.  Returns 'game', 'set', 'match' or None."""
        if self.winner is not None:
            raise MatchOver(f"match already won by player {self.winner}")
        if winner not in (1, 2):
            raise ValueError("winner must be 1 or 2")
        self.n_points += 1
        w = winner - 1
        self.points[w] += 1
        a, b = self.points[w], self.points[1 - w]
        fmt = self.set_format
        if self.in_tiebreak:
            if a >= fmt.tiebreak_points and a - b >= fmt.tiebreak_margin:
                return self._game_won(winner)
            if self.points_played_in_game % fmt.tiebreak_change_every == 0:
                self.ends_swapped ^= 1
            return None
        if a >= 4 and a - b >= 2:
            return self._game_won(winner)
        return None

    def _game_won(self, winner: int) -> str:
        w = winner - 1
        fmt = self.set_format
        was_tiebreak = self.in_tiebreak
        self.games[w] += 1
        if self.game_no_in_set % 2 == 1:
            self.ends_swapped ^= 1
        self.points = [0, 0]
        self.game_server = 3 - self.game_server
        self.game_no_in_match += 1
        self.in_tiebreak = False
        a, b = self.games[w], self.games[1 - w]
        if was_tiebreak or (a >= fmt.games and a - b >= 2):
            self.sets[w] += 1
            self.games = [0, 0]
            self.set_no += 1
            self.game_no_in_set = 1
            if self.sets[w] >= self.fmt.sets_to_win:
                self.winner = winner
                return "match"
            return "set"
        self.game_no_in_set += 1
        if fmt.has_tiebreak and self.games[0] == self.games[1] == fmt.tiebreak_at:
            self.in_tiebreak = True
        return "game"

    # -- construction from a known state --------------------------------------
    @classmethod
    def from_state(
        cls,
        fmt: MatchFormat,
        sets: tuple[int, int],
        games: tuple[int, int],
        points: tuple[int, int],
        server: int,
        game_no_in_match: int | None = None,
        ends_swapped: int = 0,
    ) -> "Scorer":
        """A scorer positioned before a point with the given (player 1, player 2) score.

        ``server`` is the server of that point; inside a tiebreak the first
        tiebreak server is recovered from it.  Ends parity cannot be recovered
        and defaults to 0.
        """
        sc = cls(fmt, server)
        sc.sets = list(sets)
        sc.games = list(games)
        sc.points = list(points)
        sc.set_no = sets[0] + sets[1] + 1
        sc.game_no_in_set = games[0] + games[1] + 1
        sc.game_no_in_match = game_no_in_match if game_no_in_match is not None else sc.game_no_in_set
        f = sc.set_format
        sc.in_tiebreak = f.has_tiebreak and games[0] == games[1] == f.tiebreak_at
        sc.game_server = server
        if sc.in_tiebreak and sc.server != server:
            sc.game_server = 3 - server
        sc.ends_swapped = ends_swapped
        return sc


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

ANNOTATION_COLUMNS = [
    "Pt", "set_no", "game_no_in_set", "game_no_in_match", "point_no_in_game", "in_tiebreak", "tb_set",
    "final_set", "server", "returner", "side", "sets_p1", "sets_p2", "games_p1", "games_p2", "points_p1",
    "points_p2", "pts", "game_score", "set_score", "ends_swapped_before_point", "after_match_end",
]


def annotate_points(
    points_df_for_one_match: pd.DataFrame,
    first_server: int,
    best_of: int,
    tournament: str,
    year: int,
    winner_col: str = "PtWinner",
) -> pd.DataFrame:
    """Score state before every point, driven by the point-winner sequence.

    Returns one row per input row (same index) with set/game/point numbers,
    ``in_tiebreak``, ``server``/``returner`` (1/2), ``side`` ('deuce'/'ad'),
    score strings (``pts`` server-first as in MCP, ``game_score`` and
    ``set_score`` player-1-first) and ``ends_swapped_before_point`` (0 = players
    at their starting ends for this point, 1 = swapped).  Rows after the match is
    decided (or after a missing winner) are flagged ``after_match_end`` and
    carry the final state.
    """
    sc = Scorer(match_format(tournament, year, best_of), int(first_server))
    rows = []
    stopped = False
    for pt, win in zip(points_df_for_one_match["Pt"], points_df_for_one_match[winner_col]):
        snap = sc.snapshot()
        snap["Pt"] = pt
        snap["after_match_end"] = stopped or sc.match_over
        rows.append(snap)
        if snap["after_match_end"] or pd.isna(win):
            stopped = True
            continue
        sc.point(int(win))
    return pd.DataFrame(rows, columns=ANNOTATION_COLUMNS, index=points_df_for_one_match.index)


# ---------------------------------------------------------------------------
# Validation against MCP's own columns
# ---------------------------------------------------------------------------

_POINT_VALUE = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}


def _parse_pts(pts: str, in_tiebreak: bool) -> tuple[int, int] | None:
    """MCP ``Pts`` (server-first) -> (server points, returner points)."""
    try:
        a, b = str(pts).split("-")
    except ValueError:
        return None
    if in_tiebreak:
        if a.isdigit() and b.isdigit():
            return int(a), int(b)
        return None
    if a not in _POINT_VALUE or b not in _POINT_VALUE:
        return None
    x, y = _POINT_VALUE[a], _POINT_VALUE[b]
    if x == 4 and y != 3 or y == 4 and x != 3:
        return None
    return x, y


def mcp_state(row, fmt: MatchFormat) -> dict | None:
    """MCP columns of one point row as a state dict comparable to :meth:`Scorer.snapshot`."""
    try:
        s1, s2, g1, g2, svr = (int(row[c]) for c in ("Set1", "Set2", "Gm1", "Gm2", "Svr"))
    except (TypeError, ValueError):
        return None
    set_no = s1 + s2 + 1
    f = fmt.set_format(set_no)
    in_tb = f.has_tiebreak and g1 == g2 == f.tiebreak_at
    p = _parse_pts(row["Pts"], in_tb)
    if p is None:
        return None
    ps, pr = p
    points = (ps, pr) if svr == 1 else (pr, ps)
    return {"sets": (s1, s2), "games": (g1, g2), "points": points, "server": svr, "in_tiebreak": in_tb}


COMPARE_FIELDS = ("Svr", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", "TbSet")
CORE_FIELDS = ("Svr", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#")
_SNAP_FOR = {
    "Svr": "server", "Set1": "sets_p1", "Set2": "sets_p2", "Gm1": "games_p1", "Gm2": "games_p2",
    "Pts": "pts", "Gm#": "game_no_in_match", "TbSet": "tb_set",
}


def _equal(mcp_value, ours) -> bool:
    if pd.isna(mcp_value):
        return False
    if isinstance(ours, bool):
        return bool(mcp_value) == ours
    if isinstance(ours, str):
        return str(mcp_value) == ours
    return int(mcp_value) == int(ours)


def _step_check(prev_row, row, fmt: MatchFormat) -> bool:
    """Is MCP row ``row`` one legal point after MCP row ``prev_row`` (given its PtWinner)?"""
    st = mcp_state(prev_row, fmt)
    if st is None or pd.isna(prev_row["PtWinner"]):
        return False
    try:
        sc = Scorer.from_state(fmt, st["sets"], st["games"], st["points"], st["server"],
                               int(prev_row["Gm#"]) if not pd.isna(prev_row["Gm#"]) else None)
        sc.point(int(prev_row["PtWinner"]))
    except (MatchOver, ValueError):
        return False
    snap = sc.snapshot()
    return all(_equal(row[c], snap[_SNAP_FOR[c]]) for c in ("Svr", "Set1", "Set2", "Gm1", "Gm2", "Pts"))


def _categorize(
    per: pd.DataFrame, df: pd.DataFrame, starts_at_zero: bool, ann: pd.DataFrame, fmt: MatchFormat
) -> tuple[str, int | None]:
    """Category of a match's disagreement and the row index of the first core mismatch."""
    core_ok = per[[f"ok_{c}" for c in CORE_FIELDS]].all(axis=1).to_numpy()
    if not starts_at_zero:
        return "partial_start", None
    if core_ok.all():
        return ("tbset_flag_only" if (~per["ok_TbSet"]).any() else "clean"), None
    i = int((~core_ok).argmax())
    if bool(ann["after_match_end"].iloc[i]):
        return "rows_after_match_end", i
    row = df.iloc[i]
    if any(pd.isna(row[c]) for c in CORE_FIELDS):
        return "mcp_missing_value", i
    return "mcp_inconsistent" if (i > 0 and not _step_check(df.iloc[i - 1], row, fmt)) else "diverged", i


def validate_match(
    points_df_for_one_match: pd.DataFrame,
    best_of: int,
    tournament: str,
    year: int,
) -> tuple[pd.DataFrame, dict]:
    """Run the state machine from 0-0 with MCP's first server and compare every point.

    Returns ``(per_point, summary)``.  ``per_point`` is the annotation plus one
    boolean ``ok_<field>`` column per compared MCP field.  ``summary['category']``:

    * ``clean`` -- every compared field agrees on every point;
    * ``tbset_flag_only`` -- only MCP's ``TbSet`` flag disagrees (always in a
      deciding set that never reached 6-6, so play cannot contradict the rule);
    * ``partial_start`` -- the first charted point is not at 0-0 (charting starts
      mid-match), so a replay from 0-0 is meaningless;
    * ``rows_after_match_end`` -- the match is decided by the rules before the
      rows end (wrong ``Best of`` in the match file, or extra rows);
    * ``mcp_missing_value`` -- the first disagreement is an empty MCP cell;
    * ``mcp_inconsistent`` -- at the first disagreement MCP's own row is not one
      legal point after its previous row (a wrong ``PtWinner``, a missing point,
      or a score typo);
    * ``diverged`` -- anything else (would indicate a rule error).

    ``finished`` is False when the last point does not end the match
    (retirement, default, or charting stopped early).  ``step_ok/step_total``
    count rows that are one legal point after MCP's previous row (independent of
    earlier errors).  ``pts_asym_*`` count points served by player 2 at an
    asymmetric score, where server-first and player-1-first ``Pts`` differ.
    """
    df = points_df_for_one_match
    fmt = match_format(tournament, year, best_of)
    first = df.iloc[0]
    starts_at_zero = (
        int(first["Set1"]) == 0 and int(first["Set2"]) == 0 and int(first["Gm1"]) == 0
        and int(first["Gm2"]) == 0 and str(first["Pts"]) == "0-0"
    )
    ann = annotate_points(df, int(first["Svr"]), best_of, tournament, year)
    per = ann.copy()
    for c in COMPARE_FIELDS:
        snapc = _SNAP_FOR[c]
        per[f"ok_{c}"] = [_equal(m, o) for m, o in zip(df[c], ann[snapc])]
    ok_cols = [f"ok_{c}" for c in COMPARE_FIELDS]
    all_ok = per[ok_cols].all(axis=1)
    category, i_bad = _categorize(per, df, starts_at_zero, ann, fmt)
    finished = (not ann["after_match_end"].any()) and _ends_match(df, fmt)
    p1_first = [
        format_pts(a, b, tb) for a, b, tb in zip(ann["points_p1"], ann["points_p2"], ann["in_tiebreak"])
    ]
    asym = (ann["server"] == 2) & (ann["points_p1"] != ann["points_p2"]) & ~ann["after_match_end"]
    mcp_pts = df["Pts"].astype(str)
    summary = {
        "n_points": len(df),
        "starts_at_zero": starts_at_zero,
        "finished": bool(finished),
        "rows_after_match_end": int(ann["after_match_end"].sum()),
        "all_fields_ok_points": int(all_ok.sum()),
        "category": category,
    }
    for c in COMPARE_FIELDS:
        summary[f"mismatch_{c}"] = int((~per[f"ok_{c}"]).sum())
    if i_bad is not None:
        summary["first_bad_pt"] = int(df["Pt"].iloc[i_bad])
        summary["first_bad_fields"] = ",".join(c for c in COMPARE_FIELDS if not per[f"ok_{c}"].iloc[i_bad])
    else:
        summary["first_bad_pt"] = None
        summary["first_bad_fields"] = ""
    recs = df[list(COMPARE_FIELDS) + ["PtWinner"]].to_dict("records")
    steps = [_step_check(recs[k - 1], recs[k], fmt) for k in range(1, len(recs))]
    summary["step_ok"] = int(sum(steps))
    summary["step_total"] = len(steps)
    summary["pts_asym_n"] = int(asym.sum())
    summary["pts_asym_server_first_ok"] = int((asym & (mcp_pts == ann["pts"])).sum())
    summary["pts_asym_p1_first_ok"] = int((asym & (mcp_pts == pd.Series(p1_first, index=df.index))).sum())
    return per, summary


def _ends_match(df: pd.DataFrame, fmt: MatchFormat) -> bool:
    sc = Scorer(fmt, int(df["Svr"].iloc[0]))
    for w in df["PtWinner"]:
        if pd.isna(w) or sc.match_over:
            return False
        sc.point(int(w))
    return sc.match_over


def validate_all(points_df: pd.DataFrame, matches_df: pd.DataFrame, best_of_source: str = "file") -> pd.DataFrame:
    """:func:`validate_match` for every match; one summary row per match.

    ``best_of_source='file'`` uses the match file's ``Best of``; ``'rule'`` uses
    :func:`grand_slam_best_of` (men's main draw 5, otherwise 3).  Both values are
    reported (``best_of_file``, ``best_of_rule``) whichever is used.
    """
    if best_of_source not in ("file", "rule"):
        raise ValueError("best_of_source must be 'file' or 'rule'")
    meta = matches_df.set_index("match_id")
    out = []
    for mid, df in points_df.groupby("match_id", sort=False):
        m = meta.loc[mid]
        bo_file = int(m["best_of"])
        bo_rule = grand_slam_best_of(m["gender"], m["Round"])
        bo = bo_file if best_of_source == "file" else bo_rule
        _, s = validate_match(df, bo, m["Tournament"], int(m["year"]))
        s.update(match_id=mid, tournament=m["Tournament"], year=int(m["year"]), gender=m["gender"],
                 round=m["Round"], best_of_file=bo_file, best_of_rule=bo_rule)
        out.append(s)
    cols = ["match_id", "tournament", "year", "gender", "round", "best_of_file", "best_of_rule"]
    res = pd.DataFrame(out)
    return res[cols + [c for c in res.columns if c not in cols]]
