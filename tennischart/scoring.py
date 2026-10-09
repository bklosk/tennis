"""Deterministic tennis scoring state machine: score, server, deuce/ad side and ends.

Players are 1 and 2. `near` is the player at the camera end. Rules (ITF):

* games: deuce/advantage; sets to 6 with a two-game margin; tiebreak at 6-6
* tiebreak: first to 7 (10 in a 10-point tiebreak), win by two; first point from the deuce
  court, then players alternate every two points; ends change every six points
* serve alternates every game (a tiebreak counts as one game served by the player due)
* ends change whenever the number of games played in the set is odd, including at set end
* final-set format depends on tournament and year (see `format_for`)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

POINT_NAMES = ["0", "15", "30", "40"]


@dataclass(frozen=True)
class MatchFormat:
    best_of: int = 3
    final_set: str = "tb7"   # "tb7" | "tb10" | "adv"
    set_tiebreak_at: int = 6

    @property
    def sets_to_win(self) -> int:
        return self.best_of // 2 + 1


def format_for(tournament: str, year: int, gender: str, best_of: int | None = None) -> MatchFormat:
    """Grand Slam singles format by tournament and year."""
    t = (tournament or "").strip().lower()
    bo = best_of or (5 if gender.upper().startswith("M") else 3)
    if "us open" in t:
        final = "tb10" if year >= 2022 else "tb7"
    elif "australian" in t:
        final = "tb10" if year >= 2019 else "adv"
    elif "wimbledon" in t or "roland" in t or "french" in t:
        final = "tb10" if year >= 2022 else "adv"   # Wimbledon 2019-21 (12-12 TB) not modelled
    else:
        final = "tb7"
    return MatchFormat(best_of=bo, final_set=final)


def format_from_mcp(final_tb: str, best_of: int | str | None, gender: str = "M") -> MatchFormat:
    """MCP `Final TB?` column: 0 = advantage, 1 = 7-point, A = 10-point at 6-6."""
    try:
        bo = int(best_of)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        bo = 5 if gender.upper().startswith("M") else 3
    final = {"0": "adv", "1": "tb7", "A": "tb10"}.get((final_tb or "").strip().upper(), "tb7")
    return MatchFormat(best_of=bo, final_set=final)


@dataclass(frozen=True)
class State:
    fmt: MatchFormat
    sets: tuple[int, int] = (0, 0)
    games: tuple[int, int] = (0, 0)
    points: tuple[int, int] = (0, 0)
    server: int = 1                 # serves the current point
    game_server: int = 1            # serves the current game (tiebreak: player due)
    near: int = 1                   # player at the near (camera) end
    tiebreak: bool = False
    winner: int | None = None
    set_history: tuple[tuple[int, int], ...] = ()

    # ---- derived ----
    @property
    def done(self) -> bool:
        return self.winner is not None

    @property
    def set_number(self) -> int:
        return sum(self.sets) + 1

    @property
    def final_set(self) -> bool:
        return self.set_number == self.fmt.best_of

    @property
    def set_has_tiebreak(self) -> bool:
        return not (self.final_set and self.fmt.final_set == "adv")

    @property
    def tiebreak_target(self) -> int:
        return 10 if (self.final_set and self.fmt.final_set == "tb10") else 7

    @property
    def receiver(self) -> int:
        return 3 - self.server

    @property
    def side(self) -> str:
        """Court the current point is served from: 'deuce' or 'ad'."""
        return "deuce" if sum(self.points) % 2 == 0 else "ad"

    @property
    def server_end(self) -> str:
        return "near" if self.server == self.near else "far"

    def score_string(self, order: str = "server") -> str:
        """Game score in MCP style ('15-0', '40-AD', tiebreak '3-2'), server first by default."""
        a, b = self.points
        if order == "server" and self.server == 2:
            a, b = b, a
        if self.tiebreak:
            return f"{a}-{b}"
        if a >= 3 and b >= 3:
            if a == b:
                return "40-40"
            return "AD-40" if a > b else "40-AD"
        return f"{POINT_NAMES[min(a, 3)]}-{POINT_NAMES[min(b, 3)]}"

    # ---- transitions ----
    def point(self, won_by: int) -> "State":
        """State before the next point, after `won_by` (1 or 2) wins this one."""
        if self.done:
            raise ValueError("match is over")
        if won_by not in (1, 2):
            raise ValueError("won_by must be 1 or 2")
        p = list(self.points)
        p[won_by - 1] += 1
        a, b = p
        st = self
        if self.tiebreak:
            target = self.tiebreak_target
            if max(a, b) >= target and abs(a - b) >= 2:
                return st._game_won(won_by)
            total = a + b
            # serve changes after the first point, then every two points
            server = self.game_server if (total % 4 in (0, 3)) else 3 - self.game_server
            near = 3 - self.near if total % 6 == 0 else self.near
            return replace(st, points=(a, b), server=server, near=near)
        if max(a, b) >= 4 and abs(a - b) >= 2:
            return st._game_won(won_by)
        return replace(st, points=(a, b))

    def _game_won(self, won_by: int) -> "State":
        g = list(self.games)
        g[won_by - 1] += 1
        ga, gb = g
        games_in_set = ga + gb
        set_won = False
        if self.tiebreak:
            set_won = True
        elif max(ga, gb) >= 6 and abs(ga - gb) >= 2:
            set_won = True
        near = 3 - self.near if games_in_set % 2 == 1 else self.near
        next_server = 3 - self.game_server
        if set_won:
            s = list(self.sets)
            s[won_by - 1] += 1
            hist = self.set_history + ((ga, gb),)
            winner = won_by if s[won_by - 1] >= self.fmt.sets_to_win else None
            return replace(self, sets=(s[0], s[1]), games=(0, 0), points=(0, 0),
                           server=next_server, game_server=next_server, near=near,
                           tiebreak=False, winner=winner, set_history=hist)
        st = replace(self, games=(ga, gb), points=(0, 0), server=next_server,
                     game_server=next_server, near=near, tiebreak=False)
        t = self.fmt.set_tiebreak_at
        if ga == t and gb == t and st.set_has_tiebreak:
            st = replace(st, tiebreak=True)
        return st


def new_match(fmt: MatchFormat, first_server: int, near: int = 1) -> State:
    return State(fmt=fmt, server=first_server, game_server=first_server, near=near)


def replay(fmt: MatchFormat, first_server: int, winners: list[int], near: int = 1) -> list[State]:
    """States before each point (len == len(winners)) for a sequence of point winners."""
    st = new_match(fmt, first_server, near)
    out = []
    for w in winners:
        out.append(st)
        if st.done:
            break
        st = st.point(w)
    return out
