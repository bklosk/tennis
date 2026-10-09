import numpy as np

from uso import rally


def test_decode_alternating_rally_skips_bounces():
    # serve at 0; true contacts at 0.9 (far), 2.0 (near), 3.1 (far); bounces in between
    t = np.array([0.5, 0.9, 1.5, 2.0, 2.6, 3.1, 7.0])
    near = np.array([-2, -3, -2, 3, -2, -3, -3.0])
    far = np.array([-2, 3, -2, -3, -2, 3, -3.0])
    idx = rally.decode_rally(t, near, far, t_serve=0.0, first="far")
    assert list(t[idx]) == [0.9, 2.0, 3.1]


def test_decode_ace_returns_nothing():
    t = np.array([0.6, 1.4])
    idx = rally.decode_rally(t, np.array([-3.0, -3.0]), np.array([-3.0, -3.0]), 0.0, "far")
    assert idx == []


def test_serve_side_geometry():
    assert rally.serve_side("near", 1.0) == "deuce"
    assert rally.serve_side("near", -1.0) == "ad"
    assert rally.serve_side("far", -1.0) == "deuce"
    assert rally.serve_side("far", 1.0) == "ad"


def test_phantom_bridges_one_missed_contact():
    # contacts: 1.0 far, (2.1 near: no onset), 3.2 far, 4.3 near
    t = np.array([1.0, 3.2, 4.3])
    near = np.array([-3.0, -3.0, 3.0])
    far = np.array([3.0, 3.0, -3.0])
    idx, ph = rally.decode_rally(t, near, far, 0.0, "far", phantom=-1.0, return_phantoms=True)
    assert list(t[idx]) == [1.0, 3.2, 4.3]
    assert len(ph) == 1 and abs(ph[0] - 2.1) < 1e-9
    # without phantoms the decoder stops after the first contact (wrong hitter parity for 3.2)
    assert len(rally.decode_rally(t, near, far, 0.0, "far")) < 3
