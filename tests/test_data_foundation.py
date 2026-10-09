"""M0 checks against the full MCP extract in data/ (parser coverage, state machine, coverage)."""

import pytest

from tennischart.mcp.load import load_matches, load_points
from tennischart.mcp.notation import parse_attempt, serialize_attempt
from tennischart.scoring import format_for, new_match
from tennischart.targets import best_of_for, coverage


@pytest.fixture(scope="module")
def points():
    return load_points()


def test_parser_handles_999_permille(points):
    cells = list(points["1st"]) + [s for s in points["2nd"] if s]
    parsed = [parse_attempt(s) for s in cells]
    ok = sum(a.ok for a in parsed)
    assert ok / len(cells) >= 0.999
    assert all(serialize_attempt(a) == a.raw for a in parsed if a.ok)


def test_state_machine_replays_mcp(points):
    matches = load_matches().set_index("match_id")
    good = total = 0
    for mid, g in points.groupby("match_id", sort=False):
        mr = matches.loc[mid]
        fmt = format_for(mr["Tournament"], int(mr["year"]), mr["gender"],
                         best_of_for(mr["gender"], mr["Round"]))
        st = new_match(fmt, int(g["Svr"].iloc[0]))
        total += 1
        ok = True
        for pts, svr, won in zip(g["Pts"], g["Svr"], g["PtWinner"]):
            if st.done or st.score_string() != pts or str(st.server) != svr:
                ok = False
                break
            st = st.point(int(won))
        good += ok
    assert good / total >= 0.99


def test_us_open_coverage_matches_guide():
    c = coverage("US Open", 2001, 2025)
    assert int(c.loc["M", "charted"]) == 120
    assert int(c.loc["W", "charted"]) == 59
    assert int(c["targets"].sum()) == 350
    assert int(c["played"].sum()) == 349
