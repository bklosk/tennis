from tennischart.scoring import MatchFormat, format_for, new_match, replay


def play(st, winners):
    for w in winners:
        st = st.point(w)
    return st


def test_game_and_deuce_strings():
    st = new_match(MatchFormat(best_of=3), first_server=2)
    assert st.score_string() == "0-0" and st.side == "deuce"
    st = st.point(1)  # receiver wins -> server-first "0-15"
    assert st.score_string() == "0-15" and st.side == "ad"
    st = play(st, [2, 2, 1, 2, 1])  # 3-3 -> deuce
    assert st.score_string() == "40-40" and st.side == "deuce"
    st = st.point(2)
    assert st.score_string() == "AD-40" and st.side == "ad"
    st = st.point(2)
    assert st.games == (0, 1) and st.server == 1


def test_ends_change_on_odd_games():
    st = new_match(MatchFormat(best_of=3), first_server=1, near=1)
    st = play(st, [1] * 4)          # 1-0
    assert st.near == 2
    st = play(st, [1] * 4)          # 2-0: even, no change
    assert st.near == 2
    st = play(st, [1] * 4)          # 3-0: odd
    assert st.near == 1


def test_tiebreak_rotation_and_end_changes():
    st = new_match(MatchFormat(best_of=3), first_server=1, near=1)
    # 6-6 with holds
    for _ in range(6):
        st = play(st, [st.server] * 4)
        st = play(st, [st.server] * 4)
    assert st.games == (6, 6) and st.tiebreak
    first = st.server
    near0 = st.near
    servers = []
    sides = []
    for k in range(6):
        servers.append(st.server)
        sides.append(st.side)
        st = st.point(1 if k % 2 == 0 else 2)  # 3-3 after six points
    assert servers == [first, 3 - first, 3 - first, first, first, 3 - first]
    assert sides == ["deuce", "ad", "deuce", "ad", "deuce", "ad"]
    assert st.near == 3 - near0          # change after six points
    st = play(st, [1, 1, 1, 1])          # 7-3
    assert st.sets == (1, 0) and not st.tiebreak
    assert st.server == 3 - first        # receiver of the first TB point serves next set
    assert st.near == near0              # 13 games: change at set end


def test_final_set_formats():
    assert format_for("US Open", 2021, "M").final_set == "tb7"
    assert format_for("US Open", 2022, "W").final_set == "tb10"
    assert format_for("Australian Open", 2018, "M").final_set == "adv"
    assert format_for("Australian Open", 2019, "M").final_set == "tb10"
    st = new_match(MatchFormat(best_of=3, final_set="tb10"), first_server=1)
    st = play(st, [1] * 24)              # 6-0
    st = play(st, [2] * 24)              # set all
    for _ in range(6):
        st = play(st, [st.server] * 4)
        st = play(st, [st.server] * 4)
    assert st.tiebreak and st.tiebreak_target == 10
    st = play(st, [1] * 7)
    assert st.winner is None
    st = play(st, [1] * 3)
    assert st.winner == 1


def test_advantage_final_set_has_no_tiebreak():
    st = new_match(MatchFormat(best_of=3, final_set="adv"), first_server=1)
    st = play(st, [1] * 24)
    st = play(st, [2] * 24)
    for _ in range(6):
        st = play(st, [st.server] * 4)
        st = play(st, [st.server] * 4)
    assert st.games == (6, 6) and not st.tiebreak


def test_replay_matches_mcp_sample():
    from tennischart.mcp.load import points_for

    g = points_for("20260201-M-Australian_Open-F-Novak_Djokovic-Carlos_Alcaraz")
    if g.empty:
        return
    states = replay(format_for("Australian Open", 2026, "M"), int(g["Svr"].iloc[0]),
                    [int(w) for w in g["PtWinner"]])
    for st, row in zip(states, g.itertuples(index=False)):
        assert st.score_string() == row.Pts
        assert st.server == int(row.Svr)
        assert st.games == (int(row.Gm1), int(row.Gm2))
