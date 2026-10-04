import numpy as np
from temporal_pose_smoothing import smooth_pose_tracks


def test_constant_track_unchanged():
    t = 30
    xy = np.zeros((t, 17, 2), np.float32)
    xy[:, 0] = [10.0, 20.0]
    conf = np.ones((t, 17), np.float32)
    sm, vis = smooth_pose_tracks(xy, conf, window=9)
    assert vis[:, 0].all()
    assert np.allclose(sm[:, 0, 0], 10.0, atol=0.05)


def test_spike_is_damped():
    t = 31
    xy = np.zeros((t, 17, 2), np.float32)
    xy[:, 5, 1] = 100.0
    xy[15, 5, 1] = 180.0
    conf = np.ones((t, 17), np.float32)
    sm, _ = smooth_pose_tracks(xy, conf, window=11)
    assert abs(sm[15, 5, 1] - 100.0) < abs(180.0 - 100.0) * 0.5
