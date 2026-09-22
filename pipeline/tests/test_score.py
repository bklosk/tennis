import numpy as np
import pandas as pd

from tennis_pipeline import score
from tennis_pipeline.score import Rules, State

BO3 = Rules(3, "tb7")


def play(winners, rules=BO3, s=None):
    s = s or State()
    for w in winners:
        s = score.win_point(s, w, rules)
    return s


def test_game_deuce_and_display():
    s = play([1, 1, 1, 2, 2, 2])
    assert s.display(BO3) == ("40", "40")
    s = score.win_point(s, 2, BO3)
    assert s.display(BO3) == ("40", "AD")
    s = score.win_point(s, 1, BO3)
    assert s.display(BO3) == ("40", "40")
    s = play([1, 1], s=s)
    assert s.games1 == (1,) and s.games2 == (0,) and (s.pts1, s.pts2) == (0, 0)


def test_tiebreak_and_next_set():
    s = State((6,), (6,), 0, 0)
    assert s.in_tiebreak(BO3)
    s = play([1] * 7, s=s)
    assert s.games1 == (7, 0) and s.games2 == (6, 0)


def test_final_set_advantage_has_no_tiebreak():
    rules = Rules(3, "adv")
    s = State((6, 3, 6), (4, 6, 6), 0, 0)
    assert not s.in_tiebreak(rules)
    s = play([1] * 4 + [2] * 4 + [1] * 4, rules, s)
    assert (s.games1[-1], s.games2[-1]) == (8, 7) and not score.match_over(s, rules)
    s = play([1] * 4, rules, s)
    assert s.games1[-1] == 9 and score.match_over(s, rules)


def test_final_set_ten_point_tiebreak():
    rules = Rules(3, "tb10")
    s = State((6, 3, 6), (4, 6, 6), 0, 0)
    s7 = play([1] * 7, rules, s)
    assert not score.match_over(s7, rules)
    assert score.match_over(play([1] * 3, rules, s7), rules)


def test_match_over_best_of_three():
    s = play([1] * 4 * 6)
    assert s.games1 == (6, 0)
    s = play([1] * 4 * 6, s=s)
    assert score.match_over(s, BO3) and s.games1 == (6, 6)


def test_server_rotation_with_tiebreak():
    assert score.server(State(), BO3, first=1) == 1
    assert score.server(State((1,), (0,)), BO3, first=1) == 2
    tb = State((6,), (6,), 0, 0)
    order = []
    for w in [1, 2, 1, 2, 1, 2]:
        order.append(score.server(tb, BO3, first=1))
        tb = score.win_point(tb, w, BO3)
    assert order == [1, 2, 2, 1, 1, 2]
    # The player who received first in the tiebreak serves the next set's first game.
    assert score.server(State((7, 0), (6, 0)), BO3, first=1) == 2


def test_path_between():
    a = State((0,), (0,), 1, 0)
    b = State((0,), (0,), 3, 1)
    p = score.path_between(a, b, BO3)
    assert sorted(p) == [1, 1, 2]
    across = score.path_between(State((1,), (0,), 3, 1), State((2,), (0,), 0, 0), BO3)
    assert across == [1]
    games_only = score.path_between(State((1,), (0,), 1, 1), State((2,), (0,), None, None), BO3)
    assert games_only == [1, 1, 1]
    assert score.path_between(State((0,), (0,)), State((5,), (5,)), BO3) is None


def test_parse_state():
    assert score.parse_state([6, 2], [4, 3], "AD", "40", BO3) == State((6, 2), (4, 3), 4, 3)
    assert score.parse_state([6, 6], [4, 6], "5", "3", BO3).pts1 == 5
    assert score.parse_state([6], [4, 2], "0", "0", BO3) is None
    assert score.parse_state([9], [4], "0", "0", BO3) is None


def simulate(n_points=160, seed=0, rules=BO3, first=2):
    rng = np.random.default_rng(seed)
    s, t, truth, snaps = State(), 0.0, [], []
    for _ in range(n_points):
        if score.match_over(s, rules):
            break
        w = int(rng.choice([1, 2], p=[0.55, 0.45]))
        truth.append({"server": score.server(s, rules, first), "winner": w})
        s = score.win_point(s, w, rules)
        t += rng.uniform(20, 45)
        snaps.append((t, s))
    return truth, snaps


def test_timeline_to_points_recovers_match():
    truth, snaps = simulate()
    rows = []
    for t, s in snaps:
        for k in range(3):  # three 1 s samples of each score
            rows.append({"t": t + k, "state": s})
        rows.append({"t": t + 3, "state": State((9,), (9,))})  # a one-off misread
    reads = pd.DataFrame(rows)
    reads.loc[reads.index[::7], "state"] = None
    stable = score.stable_states(reads, BO3)
    pts = score.points_from_states(stable, BO3, "m")
    assert len(pts) == len(truth)
    assert (pts.PointWinner.to_numpy() == [t["winner"] for t in truth]).all()
    servers_p2_first = 3 - pts.server_p1_first.to_numpy()
    assert (servers_p2_first == [t["server"] for t in truth]).all()


def test_missing_scores_become_inferred_points():
    truth, snaps = simulate(60, seed=3)
    kept = [x for k, x in enumerate(snaps) if k % 4 != 1]
    reads = pd.DataFrame([{"t": t + k, "state": s} for t, s in kept for k in range(2)])
    pts = score.points_from_states(score.stable_states(reads, BO3), BO3, "m")
    assert len(pts) == len(truth)
    assert pts.inferred.sum() > 0
