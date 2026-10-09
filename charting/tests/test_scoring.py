"""Tests for the scoring state machine and its agreement with MCP's score columns."""

from __future__ import annotations

import pandas as pd
import pytest

from uso import mcp, scoring
from uso.scoring import (
    ADVANTAGE_SET,
    TIEBREAK_7,
    TIEBREAK_10,
    MatchFormat,
    MatchOver,
    Scorer,
    annotate_points,
    final_set_rule,
    grand_slam_best_of,
    match_format,
)

HAVE_DATA = (mcp.DEFAULT_DATA_DIR / "charting-m-matches.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason=f"MCP data not found in {mcp.DEFAULT_DATA_DIR}")

BO3 = MatchFormat(3, TIEBREAK_7, TIEBREAK_7)


def hold(sc: Scorer) -> None:
    """The current server wins the game to love."""
    s = sc.server
    for _ in range(4):
        sc.point(s)


def win_set(sc: Scorer, player: int) -> None:
    set_no = sc.set_no
    while sc.set_no == set_no:
        sc.point(player)


def to_six_all(sc: Scorer) -> None:
    for _ in range(12):
        hold(sc)
    assert sc.games == [6, 6]


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


def test_game_scoring_deuce_and_advantage():
    sc = Scorer(BO3, first_server=1)
    seen = []
    for w in (1, 1, 1, 2, 2, 2, 1, 2, 2):
        seen.append((sc.pts_string(), sc.side, sc.point_no_in_game))
        sc.point(w)
    assert seen == [
        ("0-0", "deuce", 1), ("15-0", "ad", 2), ("30-0", "deuce", 3), ("40-0", "ad", 4),
        ("40-15", "deuce", 5), ("40-30", "ad", 6), ("40-40", "deuce", 7), ("AD-40", "ad", 8),
        ("40-40", "deuce", 9),
    ]
    assert sc.pts_string() == "40-AD" and sc.pts_string("p1") == "40-AD"
    assert sc.point(2) == "game"
    assert sc.games == [0, 1] and sc.server == 2 and sc.pts_string() == "0-0" and sc.side == "deuce"


def test_pts_is_written_server_first():
    sc = Scorer(BO3, first_server=1)
    hold(sc)  # player 2 now serves
    sc.point(1)
    assert sc.server == 2 and sc.pts_string() == "0-15" and sc.pts_string("p1") == "15-0"


def test_serve_alternates_each_game_and_sets():
    sc = Scorer(BO3, first_server=2)
    servers = []
    for _ in range(6):
        servers.append(sc.server)
        sc.point(1), sc.point(1), sc.point(1), sc.point(1)
    assert servers == [2, 1, 2, 1, 2, 1]
    assert sc.sets == [1, 0] and sc.games == [0, 0] and sc.set_no == 2
    assert sc.server == 2  # 6 games played, so the alternation continues across the set boundary


def test_set_needs_two_game_margin():
    sc = Scorer(BO3, first_server=1)
    for _ in range(10):
        hold(sc)  # 5-5
    hold(sc)  # 6-5
    assert sc.set_no == 1 and sc.games == [6, 5]
    for _ in range(4):
        sc.point(1)  # player 1 breaks: 7-5
    assert sc.sets == [1, 0] and sc.games == [0, 0]


# ---------------------------------------------------------------------------
# Tiebreaks
# ---------------------------------------------------------------------------


def test_tiebreak_serving_order_and_sides():
    sc = Scorer(BO3, first_server=1)
    to_six_all(sc)
    assert sc.in_tiebreak and sc.server == 1  # player due to serve game 13
    servers, sides, pts = [], [], []
    for w in (1, 2) * 6:  # 6-6 in the tiebreak
        servers.append(sc.server)
        sides.append(sc.side)
        pts.append(sc.pts_string())
        sc.point(w)
    assert servers == [1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1]
    assert sides == ["deuce", "ad"] * 6
    assert pts[:4] == ["0-0", "0-1", "1-1", "2-1"]  # server-first counts
    assert sc.in_tiebreak and sc.pts_string() == "6-6"
    sc.point(2), sc.point(2)  # 6-8: player 2 wins the set 7-6
    assert sc.sets == [0, 1] and not sc.in_tiebreak
    assert sc.server == 2  # the player who received first in the tiebreak serves next


def test_tiebreak_needs_two_point_margin():
    sc = Scorer(BO3, first_server=1)
    to_six_all(sc)
    for w in (1, 2) * 6 + (1,):
        sc.point(w)
    assert sc.in_tiebreak and sc.points == [7, 6]
    sc.point(1)
    assert sc.sets == [1, 0]


def test_ends_change_after_odd_games():
    sc = Scorer(BO3, first_server=1)
    parity = []
    for _ in range(5):
        parity.append(sc.ends_swapped)
        hold(sc)
    assert parity == [0, 1, 1, 0, 0]


def test_set_end_parity_rule():
    # 6-0: six games (even) -> no change at set end, change after game 1 of set 2
    sc = Scorer(BO3, first_server=1)
    win_set(sc, 1)
    before_set2 = sc.ends_swapped
    hold(sc)
    assert sc.ends_swapped != before_set2
    # 6-3: nine games (odd) -> change at set end
    sc = Scorer(BO3, first_server=1)
    parity_at_game_start = []
    for g in range(9):
        parity_at_game_start.append(sc.ends_swapped)
        if g < 6:
            hold(sc) if sc.server == 1 else [sc.point(1) for _ in range(4)]
        else:
            hold(sc) if sc.server == 2 else [sc.point(1) for _ in range(4)]
    assert sc.sets == [1, 0]
    # flips after games 1, 3, 5, 7, 9 -> five flips -> swapped entering set 2
    assert parity_at_game_start == [0, 1, 1, 0, 0, 1, 1, 0, 0]
    assert sc.ends_swapped == 1


def test_tiebreak_end_changes_every_six_points_and_after_the_set():
    sc = Scorer(BO3, first_server=1)
    to_six_all(sc)
    start = sc.ends_swapped
    assert start == 0  # flips after games 1,3,5,7,9,11
    parity = []
    for w in (1, 2) * 9:  # 9-9 after 18 points
        parity.append(sc.ends_swapped ^ start)
        sc.point(w)
    assert parity == [0] * 6 + [1] * 6 + [0] * 6
    assert sc.ends_swapped ^ start == 1  # changed after point 18 as the tiebreak continues
    sc.point(1), sc.point(1)  # 11-9: tiebreak (game 13, odd) ends the set -> change ends
    assert sc.sets == [1, 0] and sc.ends_swapped ^ start == 0


def test_tiebreak_ending_on_a_multiple_of_six_changes_once():
    sc = Scorer(BO3, first_server=1)
    to_six_all(sc)
    for w in (1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1):  # 7-5 after 12 points
        sc.point(w)
    assert sc.sets == [1, 0]
    # changes: after tiebreak point 6, then the set-end change (13 games); not again at point 12
    assert sc.ends_swapped == 0


# ---------------------------------------------------------------------------
# Final-set rules
# ---------------------------------------------------------------------------


def test_final_set_rule_table():
    assert final_set_rule("US Open", 2001) == TIEBREAK_7
    assert final_set_rule("US Open", 2021) == TIEBREAK_7
    assert final_set_rule("US Open", 2022) == TIEBREAK_10
    assert final_set_rule("US Open", 2025) == TIEBREAK_10
    assert final_set_rule("Australian Open", 2018) == ADVANTAGE_SET
    assert final_set_rule("Australian Open", 2019) == TIEBREAK_10
    assert final_set_rule("Australian Open", 2021) == TIEBREAK_10
    assert final_set_rule("Australian Open", 2023) == TIEBREAK_10
    assert final_set_rule("US Open", 1969) == ADVANTAGE_SET
    assert final_set_rule("US Open", 1972).tiebreak_points == 5
    assert scoring.regular_set_rule("Australian Open", 2010) == TIEBREAK_7
    with pytest.raises(ValueError):
        final_set_rule("Wimbledon", 2020)


def _to_final_set_six_all(fmt: MatchFormat) -> Scorer:
    sc = Scorer(fmt, first_server=1)
    for k in range(fmt.best_of - 1):
        win_set(sc, 1 + k % 2)
    assert sc.is_final_set
    to_six_all(sc)
    return sc


def test_us_open_2022_final_set_ten_point_tiebreak():
    sc = _to_final_set_six_all(match_format("US Open", 2022, 3))
    assert sc.in_tiebreak
    for w in (1,) * 7 + (2,) * 5:
        sc.point(w)
    assert not sc.match_over and sc.pts_string("p1") == "7-5"  # a 7-point tiebreak would be over
    for w in (1, 2, 2, 1, 2):
        sc.point(w)
    assert not sc.match_over and sc.points == [9, 8]
    assert sc.point(1) == "match" and sc.winner == 1


def test_us_open_2021_final_set_seven_point_tiebreak():
    sc = _to_final_set_six_all(match_format("US Open", 2021, 5))
    for w in (1,) * 6 + (2,) * 5:
        sc.point(w)
    assert sc.point(1) == "match" and sc.sets == [3, 2]


def test_australian_open_2018_advantage_final_set():
    sc = _to_final_set_six_all(match_format("Australian Open", 2018, 5))
    assert not sc.in_tiebreak and sc.pts_string() == "0-0"
    hold(sc)
    assert not sc.match_over and sc.games == [7, 6]
    hold(sc)
    hold(sc)
    assert sc.games == [8, 7] and not sc.match_over
    for _ in range(4):
        sc.point(1)  # break for 9-7
    assert sc.match_over and sc.sets == [3, 2]


def test_australian_open_2019_final_set_ten_point_tiebreak():
    sc = _to_final_set_six_all(match_format("Australian Open", 2019, 3))
    assert sc.in_tiebreak
    for _ in range(9):
        sc.point(2)
    assert not sc.match_over
    assert sc.point(2) == "match" and sc.winner == 2


def test_sudden_death_tiebreak_1970s():
    sc = _to_final_set_six_all(match_format("US Open", 1972, 5))
    servers = []
    for w in (1, 2) * 4:  # 4-4
        servers.append(sc.server)
        sc.point(w)
    assert servers == [1, 1, 2, 2, 1, 1, 2, 2]  # pairs from the first point (MCP 1971/1972 finals)
    assert sc.point(2) == "match"  # sudden death at 4-4


def test_match_over_and_best_of():
    sc = Scorer(BO3, first_server=1)
    win_set(sc, 2)
    win_set(sc, 2)
    assert sc.match_over and sc.winner == 2
    with pytest.raises(MatchOver):
        sc.point(1)
    assert grand_slam_best_of("m", "QF") == 5
    assert grand_slam_best_of("m", "Q2") == 3
    assert grand_slam_best_of("w", "F") == 3


def test_from_state_recovers_tiebreak_rotation():
    fmt = BO3
    sc = Scorer.from_state(fmt, sets=(1, 0), games=(6, 6), points=(2, 1), server=2)
    # 3 points played; point 4 (and 5) belong to the first tiebreak server -> player 2 opened it
    assert sc.in_tiebreak and sc.server == 2 and sc.game_server == 2 and sc.side == "ad"
    sc.point(2)
    assert sc.server == 2 and sc.side == "deuce"
    sc.point(2)
    assert sc.server == 1 and sc.side == "ad"


def test_annotate_points_synthetic():
    winners = [1] * 48  # player 1 wins 6-0 6-0
    df = pd.DataFrame({"Pt": range(1, 50), "PtWinner": winners + [2]})
    ann = annotate_points(df, first_server=2, best_of=3, tournament="US Open", year=2024)
    assert list(ann.columns[:6]) == ["Pt", "set_no", "game_no_in_set", "game_no_in_match", "point_no_in_game",
                                     "in_tiebreak"]
    assert ann["server"].iloc[0] == 2 and ann["server"].iloc[4] == 1 and ann["side"].iloc[1] == "ad"
    assert ann["pts"].iloc[1] == "0-15"  # player 1 (returner) won point 1; server-first
    assert ann["ends_swapped_before_point"].iloc[3] == 0 and ann["ends_swapped_before_point"].iloc[4] == 1
    assert ann["ends_swapped_before_point"].iloc[8] == 1 and ann["ends_swapped_before_point"].iloc[12] == 0
    assert ann["set_no"].iloc[24] == 2 and ann["set_score"].iloc[24] == "1-0"
    assert ann["after_match_end"].iloc[48] and not ann["after_match_end"].iloc[47]


# ---------------------------------------------------------------------------
# Agreement with MCP's own columns
# ---------------------------------------------------------------------------

ALLOWED = {"clean", "tbset_flag_only", "mcp_inconsistent", "mcp_missing_value", "rows_after_match_end"}


@pytest.fixture(scope="module")
def us_open_validation():
    from uso.targets import us_open_targets

    m = mcp.load_matches()
    t = us_open_targets(matches_df=m)
    p = mcp.load_points(match_ids=t["match_id"])
    return scoring.validate_all(p, m, best_of_source="rule")


@needs_data
def test_agreement_on_us_open_targets(us_open_validation):
    v = us_open_validation
    assert len(v) == 179
    assert set(v["category"]) <= ALLOWED - {"rows_after_match_end"}
    assert (v["category"] == "clean").sum() >= 175
    tot = v["n_points"].sum()
    for c in scoring.CORE_FIELDS:
        assert 1 - v[f"mismatch_{c}"].sum() / tot > 0.999, c
    assert v["step_ok"].sum() / v["step_total"].sum() > 0.9995
    # the two charted retirements are detected as unfinished
    unfinished = set(v.loc[~v["finished"], "match_id"])
    assert unfinished == {
        "20180907-M-US_Open-SF-Rafael_Nadal-Juan_Martin_Del_Potro",
        "20240903-M-US_Open-QF-Grigor_Dimitrov-Frances_Tiafoe",
    }


@needs_data
def test_pts_column_is_server_first(us_open_validation):
    v = us_open_validation
    n = v["pts_asym_n"].sum()
    assert n > 10_000
    assert v["pts_asym_server_first_ok"].sum() / n > 0.999
    assert v["pts_asym_p1_first_ok"].sum() / n < 0.01


@needs_data
def test_no_rule_divergence_in_any_local_match():
    m = mcp.load_matches()
    p = mcp.load_points()
    v = scoring.validate_all(p, m, best_of_source="rule")
    assert set(v["category"]) <= ALLOWED, v["category"].value_counts()
    assert (v["category"].isin(["clean", "tbset_flag_only"])).mean() > 0.99
    assert v["step_ok"].sum() / v["step_total"].sum() > 0.9999
    # with the match file's Best of, the mislabelled men's matches end too early
    vf = scoring.validate_all(p, m, best_of_source="file")
    conflicts = vf["best_of_file"] != vf["best_of_rule"]
    assert (vf.loc[conflicts, "category"] == "rows_after_match_end").sum() >= 20
