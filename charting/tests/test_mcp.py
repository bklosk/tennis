"""Tests for the MCP tokenizer/grammar, serializer, helpers and tables."""

from __future__ import annotations

import pytest

from uso import mcp
from uso.mcp import parse_point, serialize_point, stroke_family, stroke_side

HAVE_DATA = (mcp.DEFAULT_DATA_DIR / "charting-m-matches.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason=f"MCP data not found in {mcp.DEFAULT_DATA_DIR}")


def roundtrip(first, second=""):
    pp = parse_point(first, second)
    assert pp.ok, pp.errors
    assert serialize_point(pp) == (first.strip(), (second or "").strip())
    return pp


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


def test_lexer_kinds():
    kinds = [lx.kind for lx in mcp.lex("c4+f;37n#C")]
    assert kinds == ["let", "serve_dir", "mod", "shot", "mod", "shot_dir", "depth", "err", "outcome", "challenge"]
    assert mcp.lex("0")[0].kind == "zero"
    assert mcp.lex("g")[0].kind == "foot_fault"
    assert mcp.lex("!")[0].kind == "shank"
    assert [lx.kind for lx in mcp.lex("SRPQV")] == ["code"] * 4 + ["violation"]
    assert mcp.lex("&")[0].kind == "junk"


# ---------------------------------------------------------------------------
# Serves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("digit,name", [("4", "wide"), ("5", "body"), ("6", "T"), ("0", "unknown")])
def test_serve_directions(digit, name):
    pp = roundtrip(digit + "*")
    assert pp.serves[0].direction == digit
    assert pp.serves[0].direction_name == name
    assert pp.outcome == "ace" and pp.n_contacts == 1 and pp.in_serve_index == 1


def test_lets_and_serve_and_volley():
    pp = roundtrip("cc4+f1v2*")
    sv = pp.serves[0]
    assert sv.lets == 2 and sv.serve_and_volley and sv.direction == "4"
    assert [s.letter for s in pp.shots] == ["f", "v"]
    assert pp.outcome == "winner" and pp.last_hitter_role == "server" and pp.winner_role == "server"


def test_underarm_serve():
    pp = roundtrip("4^s-3b19*")
    assert pp.serves[0].underarm and not pp.serves[0].serve_and_volley
    assert pp.shots[0].at_net


@pytest.mark.parametrize(
    "fault,ftype",
    [("n", "net"), ("w", "wide"), ("d", "deep"), ("x", "wide_and_deep"), ("g", "foot_fault"),
     ("e", "unknown"), ("!", "shank")],
)
def test_fault_letters(fault, ftype):
    pp = roundtrip("6" + fault, "5f2n@")
    sv = pp.serves[0]
    assert sv.is_fault and sv.fault == fault and sv.fault_type == ftype
    assert pp.in_serve_index == 2 and pp.n_serve_attempts == 2
    assert pp.n_contacts == 2 and pp.outcome == "unforced_error" and pp.winner_role == "server"


def test_fault_variants():
    assert roundtrip("g", "4f1*").serves[0].direction is None  # foot fault, no direction
    assert roundtrip("e", "4n").is_double_fault
    pp = roundtrip("4x;", "5b28f2*")  # net cord on a fault sits after the letter
    assert pp.serves[0].net_cord and pp.serves[0].fault == "x"
    assert roundtrip("4!n", "5f1*").serves[0].fault_type == "net"  # shank into the net
    assert roundtrip("4wd", "5f1*").serves[0].fault_type == "wide_and_deep"
    pp = roundtrip("4n", "6d@")  # redundant '@' on a double fault
    assert pp.is_double_fault and pp.serves[1].outcome == "@"
    assert roundtrip("4+g", "5f1*").serves[0].serve_and_volley


def test_unreturnable_and_double_fault():
    pp = roundtrip("5#")
    assert pp.outcome == "service_winner" and pp.n_contacts == 1 and pp.serves[0].is_unreturnable
    pp = roundtrip("4n", "6d")
    assert pp.is_double_fault and pp.outcome == "double_fault"
    assert pp.n_contacts == 0 and pp.in_serve_index is None and pp.winner_role == "returner"
    assert pp.last_hitter_role == "server" and len(pp.fault_serves) == 2


def test_time_violation_serve():
    pp = roundtrip("V", "4f29b3d@")
    assert pp.serves[0].is_violation and not pp.serves[0].is_contact
    assert pp.in_serve_index == 2 and pp.n_serve_attempts == 2 and pp.fault_serves == []
    assert pp.n_contacts == 3 and pp.winner_role == "returner"


# ---------------------------------------------------------------------------
# Rally shots
# ---------------------------------------------------------------------------


def test_every_shot_letter_side_and_family():
    expected = {
        "f": ("F", "groundstroke"), "b": ("B", "groundstroke"), "r": ("F", "slice"), "s": ("B", "slice"),
        "v": ("F", "volley"), "z": ("B", "volley"), "o": ("F", "overhead"), "p": ("B", "overhead"),
        "u": ("F", "drop"), "y": ("B", "drop"), "l": ("F", "lob"), "m": ("B", "lob"),
        "h": ("F", "half_volley"), "i": ("B", "half_volley"), "j": ("F", "swinging_volley"),
        "k": ("B", "swinging_volley"), "t": ("U", "trick"), "q": ("U", "unknown"),
    }
    assert set(expected) == set(mcp.SHOT_LETTERS)
    for letter, (side, fam) in expected.items():
        assert (stroke_side(letter), stroke_family(letter)) == (side, fam)
        pp = roundtrip(f"4{letter}2*")
        assert pp.shots[0].letter == letter and pp.shots[0].side == side and pp.shots[0].family == fam
    assert stroke_side("?") == "U" and stroke_family("?") == "unknown"


def test_shot_fields():
    pp = roundtrip("6f+;28b0f-3d#")
    a, b, c = pp.shots
    assert (a.letter, a.modifiers, a.direction, a.depth) == ("f", "+;", "2", "8")
    assert a.approach and a.net_cord and a.direction_name == "middle" and not a.outcome
    assert b.direction == "0" and b.direction_name == "unknown"
    assert (c.modifiers, c.direction, c.error, c.outcome) == ("-", "3", "d", "#")
    assert c.at_net and c.error_type == "deep" and c.outcome_name == "forced_error"
    assert pp.outcome == "forced_error" and pp.n_contacts == 4 and pp.last_hitter_role == "returner"
    assert pp.winner_role == "server"
    assert pp.contact_roles == ["server", "returner", "server", "returner"]


def test_baseline_drop_and_shank_modifiers():
    pp = roundtrip("4f18o=1*")
    assert pp.shots[1].at_baseline
    pp = roundtrip("4f18v^1*")
    assert pp.shots[1].drop
    pp = roundtrip("6f!3#")  # shank written in the modifier slot
    assert pp.shots[0].modifiers == "!" and pp.shots[0].shank and pp.shots[0].error_type == "shank"
    pp = roundtrip("6f1w!#")  # shank as an extra error letter
    assert pp.shots[0].error == "w!" and pp.shots[0].error_type == "wide"


def test_missing_directions_and_outcome():
    pp = roundtrip("4fbf#")
    assert [s.direction for s in pp.shots] == [None, None, None] and pp.outcome == "forced_error"
    pp = roundtrip("4f8b1*")
    assert pp.shots[0].direction is None and pp.shots[0].depth == "8"
    pp = roundtrip("6f2f3w")  # final (server's) shot without outcome symbol: incomplete but parseable
    assert pp.outcome == "other" and pp.winner_role == "returner" and any("incomplete" in w for w in pp.warnings)


def test_return_error_and_rally_counts():
    pp = roundtrip("4b2n@")
    assert pp.n_contacts == 2 and pp.outcome == "unforced_error" and pp.winner_role == "server"
    pp = roundtrip("4f18f28s28f18f1d#")  # AO 2026 final, point 1: the server won
    assert pp.n_contacts == 6 and pp.last_hitter_role == "returner" and pp.winner_role == "server"


# ---------------------------------------------------------------------------
# Whole-point codes, challenges, variants and errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code,winner", [("S", "server"), ("R", "returner"), ("P", "returner"), ("Q", "server")])
def test_point_codes(code, winner):
    pp = roundtrip(code)
    assert pp.special == code and pp.winner_role == winner
    assert pp.n_contacts is None and pp.in_serve_index is None and pp.outcome == "other"


def test_challenge():
    pp = roundtrip("4b29f29C")
    assert pp.challenge and pp.n_contacts == 3 and pp.winner_role == "server" and pp.outcome == "other"
    pp = roundtrip("4x", "6C")  # returner challenged the second serve
    assert pp.challenge and pp.n_contacts == 1 and pp.winner_role == "server"


@pytest.mark.parametrize(
    "raw,canonical",
    [
        ("4b72f3*", "4b27f3*"),  # depth before direction
        ("4f1+v2*", "4f+1v2*"),  # modifier after direction
        ("5bbf@n", "5bbfn@"),  # outcome before error letter
        ("4;w", "4w;"),  # net cord before a serve fault letter
        ("n4", "4n"),  # fault letter before the serve digit
        ("4+^f1*", "4^+f1*"),  # serve modifiers in non-canonical order
    ],
)
def test_normalized_variants(raw, canonical):
    second = "5f1*" if raw in ("4;w", "n4") else ""
    pp = parse_point(raw, second)
    assert pp.ok and not pp.canonical and any(w.startswith("1st: normalized") for w in pp.warnings)
    assert serialize_point(pp)[0] == canonical
    again = parse_point(*serialize_point(pp))
    assert again.ok and again.canonical and again.serves == pp.serves and again.shots == pp.shots


@pytest.mark.parametrize(
    "first,second,fragment",
    [
        (")*", "", "expected a serve direction"),
        ("3f1w#", "", "expected a serve direction"),
        ("4f12f3*", "", "two direction digits"),
        ("4b289f1*", "", "two depth digits"),
        ("4f1*f3*", "", "outcome on non-final shot"),
        ("4f1nf3*", "", "error letter on non-final shot"),
        ("6n28f1*", "", "unexpected text after a fault"),
        ("4f2C3*", "", "'C' not at end"),
        ("4f", "6f1d@", "present but the first serve was in play"),
        ("4n", "", "2nd column is empty"),
        ("S", "4f1*", "after a whole-point code"),
        ("4s+f-3d@17D&", "", ""),
    ],
)
def test_errors(first, second, fragment):
    pp = parse_point(first, second)
    assert not pp.ok and pp.errors
    assert fragment in " ".join(pp.errors)
    assert pp.n_contacts is None and pp.outcome == "other"


def test_whitespace_is_stripped():
    pp = parse_point("4d ", " 6f2n@ ")
    assert pp.ok and serialize_point(pp) == ("4d", "6f2n@")
    assert parse_point("6f1*", " ").ok


def test_hitter_alternation():
    pp = parse_point("6x;", "5b28f2b3b2f1f1f+3*")
    assert pp.contact_roles == ["server", "returner"] * 4
    assert pp.last_hitter_role == "returner" and pp.winner_role == "returner"


# ---------------------------------------------------------------------------
# Whole-repository checks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def points():
    return mcp.load_points()


@pytest.fixture(scope="module")
def matches():
    return mcp.load_matches()


@pytest.fixture(scope="module")
def parsed(points):
    return mcp.parse_points(points)


@needs_data
def test_loaders(points, matches):
    assert set(matches["gender"]) == {"m", "w"}
    assert matches["best_of"].isin([3, 5]).all()
    assert (matches["Tournament"] == matches["Tournament"].str.strip()).all()
    assert set(matches["Tournament"]) == {"US Open", "Australian Open"}
    assert not points.duplicated(["match_id", "Pt"]).any()
    # within each match the points run 1..N after sorting and de-duplication
    g = points.groupby("match_id")["Pt"]
    assert ((g.max() - g.min() + 1) == g.size()).all() and (g.min() == 1).all()
    assert points["match_id"].isin(set(matches["match_id"])).all()
    assert "20010908-W-US_Open-F-Serena_Williams-Venus_Williams" in points.attrs["dropped_copies"]
    sub = mcp.load_points(match_ids=["20240908-M-US_Open-F-Taylor_Fritz-Jannik_Sinner"])
    assert sub["match_id"].nunique() == 1 and len(sub) == 175


@needs_data
def test_parse_rate_all_strings(points, parsed):
    n_strings = int((points["1st"] != "").sum() + (points["2nd"] != "").sum())
    failed = 0
    for pp, second in zip(parsed, points["2nd"]):
        failed += any(e.startswith("1st:") for e in pp.errors)
        failed += bool(second) and any(e.startswith("2nd:") for e in pp.errors)
    rate = 1 - failed / n_strings
    print(f"string parse rate {rate:.5f} ({failed} of {n_strings} failed)")
    assert rate >= 0.999
    point_rate = sum(pp.ok for pp in parsed) / len(parsed)
    print(f"point parse rate {point_rate:.5f}")
    assert point_rate >= 0.998


@needs_data
def test_roundtrip_all_strings(points, parsed):
    n_exact = n_norm = 0
    for pp, a, b in zip(parsed, points["1st"], points["2nd"]):
        if not pp.ok:
            continue
        out = serialize_point(pp)
        if pp.canonical:
            assert out == (a, b), (a, b, out)
            n_exact += 1
        else:
            again = parse_point(*out)
            assert again.ok and again.canonical, (a, b, out, again.errors)
            assert (again.serves, again.shots, again.special, again.challenge) == (
                pp.serves, pp.shots, pp.special, pp.challenge), (a, b, out)
            n_norm += 1
    print(f"exact round trips {n_exact}, normalized variants {n_norm}")
    assert n_norm / (n_exact + n_norm) < 0.01


@needs_data
def test_notation_agrees_with_point_winner(points, matches):
    pt = mcp.points_table(points, matches)
    known = pt["notation_winner_agrees"].dropna()
    assert len(known) > 0.99 * len(pt)
    assert known.mean() > 0.9999


@needs_data
def test_tables_on_a_known_match(matches):
    pts = mcp.load_points(match_ids=["20240908-M-US_Open-F-Taylor_Fritz-Jannik_Sinner"])
    st = mcp.shots_table(pts, matches)
    p1 = st[st["Pt"] == 1]  # 6x; | 5b28f2b3b2f1f1f+3*   (Fritz serving)
    assert list(p1["shot_no"]) == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    fault = p1.iloc[0]
    assert fault["is_fault_serve"] and fault["serve_attempt"] == 1 and fault["error"] == "x"
    serve = p1.iloc[1]
    assert serve["is_serve"] and not serve["is_fault_serve"] and serve["serve_attempt"] == 2
    assert serve["letter"] == "serve" and serve["family"] == "serve" and serve["direction"] == "5"
    assert serve["hitter"] == 1 and serve["hitter_name"] == "Taylor Fritz" and serve["hitter_role"] == "server"
    ret = p1.iloc[2]
    assert ret["hitter"] == 2 and ret["hitter_name"] == "Jannik Sinner" and ret["hitter_role"] == "returner"
    assert (ret["letter"], ret["side"], ret["family"], ret["direction"], ret["depth"]) == (
        "b", "B", "groundstroke", "2", "8")
    last = p1.iloc[-1]
    assert last["is_last_contact"] and last["outcome"] == "*" and last["modifiers"] == "+"
    assert (p1["n_contacts"] == 8).all()
    assert (st["hitter_hand"] == "R").all()
    pt = mcp.points_table(pts, matches)
    r = pt.iloc[0]
    assert (r["server"], r["server_name"], r["returner_name"]) == (1, "Taylor Fritz", "Jannik Sinner")
    assert (r["n_serve_attempts"], r["in_serve"], r["n_contacts"], r["outcome"]) == (2, 2, 8, "winner")
    assert r["PtWinner"] == 2 and r["notation_winner_agrees"]
    # every non-fault contact row count equals the points' n_contacts total
    contacts = st[~st["is_fault_serve"]].groupby("Pt").size()
    assert (contacts == pt.set_index("Pt")["n_contacts"].loc[contacts.index]).all()
