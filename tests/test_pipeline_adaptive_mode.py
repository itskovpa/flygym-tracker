"""The pipeline in adaptive rotation mode needs NO preset enter/exit thresholds and drops in the
AdaptiveRotationDetector. Guards the config wiring (rotation.detector: adaptive)."""
import numpy as np

from flygym_tracker.adaptive_rotation import AdaptiveRotationDetector
from flygym_tracker.calibration import build_calibration_from_boxes, load_calibration, save_calibration
from flygym_tracker.config import load_config
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.types import Frame

W, H = 100, 80


def _base():
    f = np.full((H, W), 30, np.uint8)
    f[10:70, 15:45] = 200
    f[10:70, 55:85] = 200
    return f


class _Seq(FrameSource):
    def __init__(self, frames): self._f = frames; self._i = 0
    def open(self): self._i = 0
    def read(self):
        if self._i >= len(self._f):
            return None
        fr = Frame(image=self._f[self._i], index=self._i, t_monotonic=self._i * 0.05,
                   t_wall_iso="2026-07-18T00:00:%02d" % (self._i % 60))
        self._i += 1
        return fr
    def close(self): pass
    @property
    def fps(self): return 20.0
    @property
    def frame_size(self): return (W, H)


def test_adaptive_mode_requires_no_thresholds_and_detects_rotation(tmp_path):
    calib, illum, ov = build_calibration_from_boxes(_base(), "A", [(15, 10, 30, 60), (55, 10, 30, 60)], [True, True])
    cdir = tmp_path / "calib"; cdir.mkdir()
    save_calibration(calib, illum, str(cdir), overlay=ov)
    calibration = load_calibration(str(cdir))

    frames = ([_base() for _ in range(35)]                               # quiet (seeds the floor)
              + [np.roll(_base(), (k + 1) * 6, axis=1) for k in range(10)]  # global shift = rotation
              + [_base() for _ in range(25)])                            # quiet again

    cfg = load_config(overrides={
        "rotation": {"detector": "adaptive", "debounce_frames": 3, "min_stationary_frames": 3},
        "activity": {"pixel_threshold": 30},
        "binning": {"bin_seconds": 1},
    })
    logger = ActivityLogger(str(tmp_path / "out"), run_id="ad", fmt="csv")
    # NOTE: no enter_threshold/exit_threshold passed or configured -> would raise in threshold mode.
    pipe = TrackerPipeline(cfg, calibration, _Seq(frames), logger, clock="index", pixel_threshold=30)

    assert pipe.detector_mode == "adaptive"
    assert isinstance(pipe.rotation, AdaptiveRotationDetector)
    summary = pipe.run()
    assert summary["frames_processed"] == len(frames)
    assert summary["n_rotations"] >= 1   # the injected global shift is caught with no preset threshold
