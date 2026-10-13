"""Tests for the analytics grids, the rally-time scorer, the incremental API ledger tally and the
batch-match MCP lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analytics as A  # noqa: E402

from uso import decisions, rallytime, truth  # noqa: E402
from uso.paths import BATCH_MATCHES  # noqa: E402


def test_grid_row0_is_the_net_and_x_runs_left_to_right():
    # one sample just behind the baseline, 2 m right of centre (hitter frame)
    h = A.grid([2.1], [-A.BASELINE - 0.1], 0.5)
    i, j = np.argwhere(h == 1)[0]
    assert h.shape == (40, 32)
    assert i == int((0 - (-A.BASELINE - 0.1)) // 0.5)  # rows count down from the net
    assert j == int((2.1 - A.HALF["x"][0]) // 0.5)


def test_heat_panels_share_one_scale():
    a = pd.DataFrame(dict(x=[0.0] * 10, y=[-12.0] * 10))
    b = pd.DataFrame(dict(x=[0.0, 3.0], y=[-12.0, -12.0]))
    p = A.heat_panels({"a": a, "b": b})
    assert p["a"]["dens"].max() == pytest.approx(1.0)  # the denser panel sets the peak
    assert p["b"]["dens"].max() < 1.0


def test_win_map_cells_start_at_the_baseline():
    r = pd.DataFrame(dict(hx=[0.1] * 50 + [3.0] * 50, depth=[0.2] * 50 + [-0.4] * 50,
                          won=[1.0] * 30 + [0.0] * 20 + [1.0] * 10 + [0.0] * 40))
    r["hy"] = -A.BASELINE - r.depth
    m = A.win_map(r, min_n=40)
    cells = {(c["x0"], c["d0"]): c for c in m["cells"]}
    assert cells[(-0.75, 0.0)]["win"] == 60.0  # centred on the centre line, starts at the baseline
    assert cells[(2.25, -1.5)]["win"] == 20.0
    assert m["bands"][0]["zone"] == "Inside the baseline" and m["bands"][0]["shots"] == 50


def test_rallytime_score():
    y = np.array([1, 1, 0, 0, 1], bool)
    p = np.array([0.9, 0.2, 0.7, 0.1, 0.8])
    sc = rallytime.score(y, p)
    assert sc["precision"] == 2 / 3 and sc["recall"] == 2 / 3 and sc["acc"] == 0.6


def test_ledger_tally_reads_only_new_lines(tmp_path, monkeypatch):
    led = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(decisions, "LEDGER", led)
    monkeypatch.setattr(decisions, "_TALLY", {"offset": 0, "usd": 0.0, "tokens": 0})
    assert decisions.DecisionClient.spent() == 0.0
    rows = [dict(t=0, key="a", input_tokens=700, usd=7e-5, label="x"), dict(t=1, key="b", input_tokens=300, usd=3e-5, label="x")]
    led.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert decisions.DecisionClient.spent() == pytest.approx(1e-4)
    with led.open("a") as f:  # another process appends; a half-written line is not counted yet
        f.write(json.dumps(dict(t=2, key="c", input_tokens=100, usd=1e-5, label="x")) + "\n" + '{"t": 3, "us')
    assert decisions.DecisionClient.spent() == pytest.approx(1.1e-4)
    assert decisions.DecisionClient.tokens_spent() == 1100


@pytest.mark.skipif(not BATCH_MATCHES.exists(), reason="no batch match table")
def test_match_row_falls_back_to_charted_batch_matches():
    bm = pd.read_csv(BATCH_MATCHES)
    charted = bm[bm.role == "charted"].iloc[0]
    assert truth.match_row(charted.video_id).mcp_match_id == charted.mcp_match_id
    assert truth.match_row("Q4UB3yY_C2k").role == "dev"  # evaluation table wins
    unch = bm[bm.role == "uncharted"]
    if len(unch):
        with pytest.raises(KeyError):
            truth.match_row(unch.iloc[0].video_id)
