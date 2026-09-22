"""Tennis scoring rules: turn a timeline of on-screen scores into a point-by-point table.

Used for matches without official point-by-point data (everything before 2011, plus gaps such
as AO 2024 and 2025+). The output mirrors the official columns `align` consumes; the server
column is given for the hypothesis "player 1 served first", and `align` tries both.
"""
from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

DISPLAY = {0: "0", 1: "15", 2: "30", 3: "40"}
PARSE = {"0": 0, "00": 0, "O": 0, "LOVE": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4, "ADV": 4}


@dataclass(frozen=True)
class Rules:
    best_of: int = 3
    final_set: str = "tb7"  # "tb7", "tb10" (first to 10 at 6-6) or "adv" (no final-set tiebreak)


def rules_for(tournament: str, year: int, draw: str) -> Rules:
    best_of = 5 if draw == "men" else 3
    if year >= 2022:
        return Rules(best_of, "tb10")
    if tournament == "Australian Open":
        return Rules(best_of, "tb10" if year >= 2019 else "adv")
    return Rules(best_of, "tb7")


@dataclass(frozen=True)
class State:
    """Games per set (current set last) and points in the current game or tiebreak."""
    games1: tuple = (0,)
    games2: tuple = (0,)
    pts1: int | None = 0
    pts2: int | None = 0

    @property
    def set_no(self) -> int:
        return len(self.games1)

    def in_tiebreak(self, rules: Rules) -> bool:
        g1, g2 = self.games1[-1], self.games2[-1]
        final = self.set_no == rules.best_of
        return g1 == 6 and g2 == 6 and not (final and rules.final_set == "adv")

    def games_played(self) -> int:
        return sum(self.games1) + sum(self.games2)

    def display(self, rules: Rules) -> tuple[str, str]:
        if self.pts1 is None:
            return "", ""
        if self.in_tiebreak(rules):
            return str(self.pts1), str(self.pts2)
        a, b = self.pts1, self.pts2
        if a >= 3 and b >= 3:
            return ("AD", "40") if a > b else ("40", "AD") if b > a else ("40", "40")
        return DISPLAY[min(a, 3)], DISPLAY[min(b, 3)]

    def key(self, rules: Rules):
        """Identity as shown on screen (deuce/advantage states collapse)."""
        return self.games1, self.games2, self.display(rules)


def set_complete(x: int, y: int, final_adv: bool) -> bool:
    return (max(x, y) >= 6 and abs(x - y) >= 2) or (not final_adv and max(x, y) == 7)


def sets_won(s: State, rules: Rules) -> tuple[int, int]:
    w1 = w2 = 0
    for k, (a, b) in enumerate(zip(s.games1, s.games2)):
        if set_complete(a, b, k + 1 == rules.best_of and rules.final_set == "adv"):
            w1 += a > b
            w2 += b > a
    return w1, w2


def match_over(s: State, rules: Rules) -> bool:
    need = rules.best_of // 2 + 1
    return max(sets_won(s, rules)) >= need


def server(s: State, rules: Rules, first: int = 1) -> int:
    """Who serves the next point, if player `first` served the first game of the match."""
    g = s.games_played()
    game_server = first if g % 2 == 0 else 3 - first
    if s.in_tiebreak(rules) and s.pts1 is not None:
        k = s.pts1 + s.pts2
        return game_server if ((k + 1) // 2) % 2 == 0 else 3 - game_server
    return game_server


def win_point(s: State, winner: int, rules: Rules) -> State:
    a, b = (s.pts1 or 0), (s.pts2 or 0)
    a, b = (a + 1, b) if winner == 1 else (a, b + 1)
    g1, g2 = list(s.games1), list(s.games2)
    if s.in_tiebreak(rules):
        target = 10 if (s.set_no == rules.best_of and rules.final_set == "tb10") else 7
        done = max(a, b) >= target and abs(a - b) >= 2
    else:
        done = max(a, b) >= 4 and abs(a - b) >= 2
    if not done:
        if not s.in_tiebreak(rules) and a >= 3 and b >= 3:
            a, b = 3 + (a > b), 3 + (b > a)  # keep deuce states canonical
        return replace(s, pts1=a, pts2=b)
    if winner == 1:
        g1[-1] += 1
    else:
        g2[-1] += 1
    if set_complete(g1[-1], g2[-1], s.set_no == rules.best_of and rules.final_set == "adv"):
        new = State(tuple(g1), tuple(g2), 0, 0)
        if not match_over(new, rules):
            new = State(tuple(g1) + (0,), tuple(g2) + (0,), 0, 0)
        return new
    return State(tuple(g1), tuple(g2), 0, 0)


def matches(s: State, target: State, rules: Rules) -> bool:
    if s.games1 != target.games1 or s.games2 != target.games2:
        return False
    if target.pts1 is None:  # games-only graphic: shown between games
        return s.pts1 == 0 and s.pts2 == 0
    return s.display(rules) == target.display(rules)


def path_between(a: State, b: State, rules: Rules, max_points: int = 14) -> list[int] | None:
    """Shortest sequence of point winners taking score `a` to score `b` (None if unreachable)."""
    if matches(a, b, rules):
        return []
    start = replace(a, pts1=a.pts1 or 0, pts2=a.pts2 or 0)
    seen = {start.key(rules)}
    q = deque([(start, [])])
    while q:
        s, path = q.popleft()
        if len(path) >= max_points or match_over(s, rules):
            continue
        for w in (1, 2):
            n = win_point(s, w, rules)
            p = path + [w]
            if matches(n, b, rules):
                return p
            k = n.key(rules)
            if k not in seen:
                seen.add(k)
                q.append((n, p))
    return None


def parse_state(games1, games2, pts1, pts2, rules: Rules) -> State | None:
    """Build a State from parsed graphic fields; None if implausible."""
    if not games1 or len(games1) != len(games2) or len(games1) > rules.best_of:
        return None
    final_adv = len(games1) == rules.best_of and rules.final_set == "adv"
    if any(g < 0 or (g > 7 and not final_adv) for g in (*games1, *games2)):
        return None
    if pts1 is None or pts2 is None:
        return State(tuple(games1), tuple(games2), None, None)
    if State(tuple(games1), tuple(games2)).in_tiebreak(rules):
        try:
            a, b = int(pts1), int(pts2)
        except ValueError:
            return None
        return State(tuple(games1), tuple(games2), a, b)
    if pts1 not in PARSE or pts2 not in PARSE:
        return None
    a, b = PARSE[pts1], PARSE[pts2]
    if a == 4 and b == 4:
        return None
    if a == 4 or b == 4:
        a, b = (4, 3) if a == 4 else (3, 4)
    return State(tuple(games1), tuple(games2), a, b)


def _progress(s: State) -> tuple:
    a, b = s.pts1 or 0, s.pts2 or 0
    if a >= 3 and b >= 3 and abs(a - b) <= 1:  # deuce <-> advantage is not a step backwards
        return s.set_no, s.games_played(), 6
    return s.set_no, s.games_played(), a + b


def stable_states(reads: pd.DataFrame, rules: Rules, min_repeat: int = 2) -> list[dict]:
    """Debounce per-sample score reads into a sequence of distinct on-screen states.

    `reads` has columns t and state (State or None), in time order. A state is accepted after
    `min_repeat` consecutive identical reads; a state that goes backwards in match order is
    treated as a misread unless it persists for twice as long.
    """
    out = []
    run_state, run_t0, run_t1, run_n = None, None, None, 0

    def flush():
        if run_state is None:
            return
        need = min_repeat
        if out and _progress(run_state) < _progress(out[-1]["state"]):
            need = 2 * min_repeat
        if run_n < need:
            return
        if out and out[-1]["state"].key(rules) == run_state.key(rules):
            out[-1]["last_seen"] = run_t1
            out[-1]["n_reads"] += run_n
        else:
            out.append({"state": run_state, "first_seen": run_t0, "last_seen": run_t1, "n_reads": run_n})

    for r in reads.itertuples():
        s = r.state
        if s is None:
            continue
        if run_state is not None and s.key(rules) == run_state.key(rules):
            run_t1, run_n = r.t, run_n + 1
            continue
        flush()
        run_state, run_t0, run_t1, run_n = s, r.t, r.t, 1
    flush()
    if not out:
        return out
    # Drop isolated regressions that survived debouncing (e.g. a replay graphic of an old score).
    cleaned = [out[0]]
    for rec in out[1:]:
        if _progress(rec["state"]) < _progress(cleaned[-1]["state"]) and rec["n_reads"] < 3 * min_repeat:
            continue
        cleaned.append(rec)
    return cleaned


def points_from_states(states: list[dict], rules: Rules, match_id: str) -> pd.DataFrame:
    """Expand consecutive on-screen states into points (server given for player 1 serving first)."""
    if not states:
        return pd.DataFrame()
    # (score before, winners, last time the old score was seen, first time the new one was seen)
    steps = []
    prefix = path_between(State(), states[0]["state"], rules)
    if prefix:
        steps.append((State(), prefix, None, states[0]["first_seen"]))
    breaks = 0
    for a, b in zip(states[:-1], states[1:]):
        path = path_between(a["state"], b["state"], rules)
        if path is None:  # too many points unseen, or a misread that survived debouncing
            breaks += 1
            continue
        steps.append((a["state"], path, a["last_seen"], b["first_seen"]))
    rows = []
    for start, path, t_lo, t_hi in steps:
        cur = replace(start, pts1=start.pts1 or 0, pts2=start.pts2 or 0)
        n = len(path)
        for k, w in enumerate(path):
            srv = server(cur, rules, first=1)
            nxt = win_point(cur, w, rules)
            p1, p2 = nxt.display(rules)
            game_no = cur.games1[-1] + cur.games2[-1] + 1
            est = t_hi if t_lo is None or n == 1 else t_lo + (t_hi - t_lo) * (k + 1) / n
            rows.append({"match_id": match_id, "SetNo": cur.set_no, "GameNo": game_no, "PointWinner": w,
                         "server_p1_first": srv, "P1Score": p1, "P2Score": p2,
                         "video_t_lo": t_lo, "video_t_hi": t_hi, "elapsed_s": float(est),
                         "inferred": n > 1, "tiebreak": cur.in_tiebreak(rules)})
            cur = nxt
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(1, "PointNumber", np.arange(1, len(df) + 1))
    df.attrs["breaks"] = breaks
    return df
