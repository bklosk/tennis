import pytest

from tennischart.mcp.notation import (
    build_attempt,
    letter_for,
    parse_attempt,
    parse_point,
    serialize_attempt,
    stroke_family,
    stroke_side,
)


def test_rally_point_structure():
    p = parse_point("4f18f28s28f18f1d#", "")
    assert p.ok and p.serve_number == 1
    shots = p.shots
    assert [s.stroke for s in shots] == ["serve", "f", "f", "s", "f", "f"]
    assert shots[0].direction == 4
    assert shots[1].direction == 1 and shots[1].depth == 8
    assert shots[-1].error == "d" and shots[-1].outcome == "#"
    assert [s.hitter for s in shots[:3]] == ["server", "returner", "server"]
    assert p.rally_length == 6
    # last shot (index 5, returner) was a forced error -> server wins
    assert p.winner() == "server"


def test_fault_then_second_serve():
    p = parse_point("4n", "6f2n@")
    assert p.ok and p.serve_number == 2
    assert p.attempts[0].is_fault and p.attempts[0].serve.error == "n"
    assert p.rally_length == 2
    assert p.winner() == "server"  # return (returner) unforced error


def test_double_fault_and_aces():
    assert parse_point("6d", "4n").winner() == "returner"
    assert parse_point("6d", "4n").double_fault
    assert parse_point("4*").winner() == "server"
    assert parse_point("5#").final.ended == "unreturnable"
    assert parse_point("c6*").final.lets == 1


def test_serve_and_volley_and_modifiers():
    a = parse_attempt("6+b3z-3*")
    assert a.ok
    assert a.serve.serve_and_volley
    assert a.shots[2].at_net and a.shots[2].stroke == "z"
    a = parse_attempt("4f27f+3f-2v1*")
    assert a.shots[2].approach and a.shots[3].at_net


@pytest.mark.parametrize("code,winner", [("S", "server"), ("R", "returner"), ("P", None), ("Q", None)])
def test_whole_point_codes(code, winner):
    p = parse_point(code)
    assert p.ok and p.final.code == code
    if winner:
        assert p.winner() == winner


def test_time_violation_is_a_fault():
    p = parse_point("V", "4f1*")
    assert p.serve_number == 2


def test_typos_are_tolerated_and_round_trip():
    for s in ["4b81f1f3b2j+3b1v1f1*", "4x;", "6d;", "4wd", "4f1f3@d", "n4", "4f28f1w!@"]:
        a = parse_attempt(s)
        assert a.ok, (s, a.problem)
        assert a.irregular
        assert serialize_attempt(a) == s
    # a bare unknown fault is valid notation, like a bare foot fault
    assert parse_attempt("e").ok and parse_attempt("e").is_fault


def test_embedded_second_serve():
    p = parse_point("6d5f28b2f3b3f;3s2m2d#", "")
    assert p.ok and p.serve_number == 2
    assert p.attempts[1].serve.direction == 5


def test_junk_is_rejected():
    for s in [")*", "3f28f3*", "o=38A&*", "5f37b3f1f2f1f2f2s2s*2m1*"]:
        assert not parse_attempt(s).ok


@pytest.mark.parametrize("s", ["4f18f28s28f18f1d#", "c4b28f+3s#", "6+b3z^17f-2v29t2v^3n@", "0q#",
                               "5b27f38b39C", "4n", "g", "6+f27i^27m-1d#"])
def test_round_trip(s):
    assert serialize_attempt(parse_attempt(s)) == s


def test_build_attempt():
    a = build_attempt([{"stroke": "serve", "direction": 6},
                       {"stroke": "b", "direction": 2, "depth": 8},
                       {"stroke": "f", "direction": 1, "outcome": "*"}])
    assert a.raw == "6b28f1*"
    assert parse_attempt(a.raw).winner() == "server"


def test_letter_mapping():
    assert stroke_side("r") == "forehand" and stroke_family("r") == "slice"
    assert stroke_side("i") == "backhand" and stroke_family("i") == "half_volley"
    assert letter_for("volley", "backhand") == "z"
    assert letter_for("groundstroke", None) == "q"
    for letter in "fbrsvzopuylmhijk":
        assert letter_for(stroke_family(letter), stroke_side(letter)) == letter
