import numpy as np
import torch

from tennis_pipeline import serve, tracknet

from test_serve import CONTACT, make_tracks, onsets_at


class ChannelNet(torch.nn.Module):
    """Stand-in for TrackNet whose heatmap class is 255 x (input channel `k`)."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x):
        classes = torch.arange(256.0).view(1, 256, 1, 1)
        return -(classes - 255 * x[:, self.k:self.k + 1]) ** 2


def test_ball_mask_matches_argmax_threshold():
    """The Neural Engine head takes (current, previous, previous-but-one) HWC pixel frames and
    must reproduce argmax > 127 of the 9-channel model input."""
    torch.manual_seed(0)
    frames = [torch.randint(0, 256, (1, 24, 32, 3)).float() for _ in range(3)]
    x = torch.cat(frames, 3).permute(0, 3, 1, 2) / 255
    for k in (0, 4, 8):  # blue of the current frame, green of the previous, red of the oldest
        net = ChannelNet(k)
        ref = net(x).argmax(1) > 127
        got = tracknet.BallMask(net)(*frames) > 0
        assert ref.any() and (~ref).any()
        assert torch.equal(ref, got)


def test_serve_contact_follows_the_toss_peak():
    """The box is tallest at the toss; contact is the server's first sound after it."""
    tr, b = make_tracks()
    late = CONTACT + 20  # racket sound 0.67 s after the tallest box
    cands = serve.detect(tr, b, [], onsets_at(CONTACT - 40, late))
    acc = [c for c in cands if c["accepted"]]
    assert len(acc) == 1 and acc[0]["frame"] == late


def test_serve_contact_prefers_the_servers_own_hit():
    tr, b = make_tracks()
    hit = {"frame": CONTACT + 25, "side": "near", "dv": 30.0, "pos": np.array([600.0, 400.0]), "gap_frames": 2,
           "player_seen": True, "toss": False}
    cands = serve.detect(tr, b, [hit], None)
    acc = [c for c in cands if c["accepted"]]
    assert len(acc) == 1 and acc[0]["frame"] == CONTACT + 25 and acc[0]["hit_near"] == 1.0

