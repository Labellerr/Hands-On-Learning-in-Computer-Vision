# Temporal pose smoothing (recorded video)

YOLO pose (and the DeadLift overlay notebook) draws **per-frame** keypoints.
That is the source of overlay jitter — not a bad weight, and not a live-stream
requirement.

For **offline** clips, run detection on every frame, interpolate short gaps,
then Savitzky–Golay each coordinate:

```python
from temporal_pose_smoothing import smooth_pose_tracks
sm, vis = smooth_pose_tracks(xy, conf)  # xy: (T, 17, 2)
```

Causal One-Euro is the realtime alternative; it still lags. Two-pass Savgol is
what you want for a 10s analysis render.

Requires: `numpy`, `scipy`.
