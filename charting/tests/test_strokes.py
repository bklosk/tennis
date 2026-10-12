import numpy as np

from uso import strokes


def _pose(rw, lw, facing_camera: bool):
    """17x3 COCO keypoints for an upright player; image y grows downward.
    Facing the camera, the player's right shoulder appears on the image left."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    sx = -20 if facing_camera else 20  # image x of the right shoulder relative to centre
    k[5] = (100 - sx, 100, 0.9)  # left shoulder
    k[6] = (100 + sx, 100, 0.9)  # right shoulder
    k[11] = (100 - sx / 2, 160, 0.9)  # left hip
    k[12] = (100 + sx / 2, 160, 0.9)  # right hip
    k[0] = (100, 80, 0.9)
    k[9] = (*lw, 0.9)
    k[10] = (*rw, 0.9)
    return k


def test_side_from_image_mapping():
    # right-hander: forehand is image-right at the near end, image-left at the far end
    assert strokes.side_from_image(0.1, 0.9, "near", "R") > 0.5
    assert strokes.side_from_image(0.1, 0.9, "far", "R") < 0.5
    # left-hander mirrors
    assert strokes.side_from_image(0.1, 0.9, "near", "L") < 0.5
    assert strokes.side_from_image(0.9, 0.1, "far", "L") < 0.5


def test_axis_feature_is_orientation_free():
    # right wrist out on the player's own right side: forehand side, whichever way they face
    back = _pose(rw=(150, 130), lw=(90, 140), facing_camera=False)
    front = _pose(rw=(50, 130), lw=(110, 140), facing_camera=True)
    for kp in (back, front):
        f = strokes.axis_features(kp, "R")
        assert f["rw_ax"] > 0.5
    # the same image for a left-hander: the racket (left) wrist is on the body's right of centre
    f = strokes.axis_features(back, "L")
    assert f["rw_ax"] < 0.5


def test_mirrored_body_frame_features():
    near = _pose(rw=(150, 130), lw=(90, 140), facing_camera=False)
    f = strokes.pose_features([near], "near", "R")
    assert f["rw_x_0"] > 0  # forehand side is +x after mirroring
    far = _pose(rw=(50, 130), lw=(110, 140), facing_camera=True)
    f = strokes.pose_features([far], "far", "R")
    assert f["rw_x_0"] > 0
