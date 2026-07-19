"""Regression: registration must not alias ROIs onto neighbouring vials on the periodic lattice.

The vial lattice is a row of near-identical vials, so phase correlation can lock onto a whole
vial-pitch offset with HIGH confidence (low residual). Without a shift-magnitude guard, that would
shift every ROI onto its neighbour after a rotation, misattributing all per-vial activity. The
pipeline's `max_shift` guard (default 0.4x the tightest vial pitch) must reject such a lock and keep
ROIs at their calibration anchors.

Found by the orchestrator's end-to-end validation, not the original unit tests (which used
distinguishable scenes that don't trigger the lattice ambiguity).
"""
import glob

import cv2
import numpy as np
import pandas as pd
import pytest

from flygym_tracker.calibration import (
    build_calibration_from_boxes,
    load_calibration,
    save_calibration,
)
from flygym_tracker.config import load_config
from flygym_tracker.frame_source import FrameSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.registration import estimate_shift
from flygym_tracker.types import Frame

W, H, BG, BAR, FLY = 120, 80, 30, 200, 40
BOX_A, BOX_B = (15, 10, 30, 60), (75, 10, 30, 60)  # identical bars, 60 px pitch


def _base():
    f = np.full((H, W), BG, np.uint8)
    for (x, y, w, h) in (BOX_A, BOX_B):
        f[y:y + h, x:x + w] = BAR
    return f


def _with_fly(box, t):
    f = _base()
    x, y, w, h = box
    oy = y + 5 + (t * 3) % (h - 15)
    f[oy:oy + 5, x + 12:x + 17] = FLY
    return f


class _SeqSource(FrameSource):
    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def open(self):
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return None
        im = self._frames[self._i]
        fr = Frame(image=im, index=self._i, t_monotonic=self._i * 0.1,
                   t_wall_iso="2026-07-18T00:%02d:%02d" % (self._i // 60, self._i % 60))
        self._i += 1
        return fr

    def close(self):
        pass

    @property
    def fps(self):
        return 10.0

    @property
    def frame_size(self):
        return (W, H)


def test_phase_correlation_locks_onto_the_lattice_pitch():
    """Documents the underlying hazard the guard exists for: on this periodic scene, estimate_shift
    reports ~a full vial pitch (60 px), NOT ~0, and with high confidence."""
    ref = _with_fly(BOX_A, 0)
    cur = _with_fly(BOX_B, 0)
    mask = _base() == BAR
    dx, dy, residual = estimate_shift(cur, ref, mask=mask)
    assert abs(abs(dx) - 60) < 3      # locked onto the 60 px bar pitch
    assert residual < 0.2             # ...with high confidence -> residual guard alone is blind


def test_registration_guard_keeps_per_vial_attribution_after_rotation(tmp_path):
    calib, illum, overlay = build_calibration_from_boxes(_base(), "A", [BOX_A, BOX_B], [True, True])
    cdir = tmp_path / "calib"
    cdir.mkdir()
    save_calibration(calib, illum, str(cdir), overlay=overlay)
    calibration = load_calibration(str(cdir))

    frames = [_with_fly(BOX_A, t) for t in range(30)]           # phase 1: fly in vial A
    frames += [np.roll(_base(), (t + 1) * 8, axis=1) for t in range(12)]  # phase 2: rotation
    frames += [_with_fly(BOX_B, t) for t in range(40)]          # phase 3: fly in vial B

    cfg = load_config(overrides={
        "binning": {"bin_seconds": 1},
        "rotation": {"debounce_frames": 3, "min_stationary_frames": 3},
    })
    logger = ActivityLogger(str(tmp_path / "out"), run_id="reg", fmt="csv")
    pipe = TrackerPipeline(
        cfg, calibration, _SeqSource(frames), logger,
        clock="index", pixel_threshold=30, enter_threshold=15, exit_threshold=8,
    )
    # sanity: guard is well below the 60 px pitch (0.4 * 60 = 24)
    assert pipe.max_shift < 60
    summary = pipe.run()
    assert summary["n_rotations"] == 1

    adf = pd.concat([pd.read_csv(c) for c in glob.glob(str(tmp_path / "out" / "activity_*.csv"))])
    edf = pd.read_csv(str(tmp_path / "out" / "events.csv"))

    late = adf[adf.elapsed_s >= 5]  # deep into phase 3
    b_late = late[late.vial_id == 2].motion_px_sum.sum()
    a_late = late[late.vial_id == 1].motion_px_sum.sum()
    assert b_late > 100, f"vial B should carry phase-3 motion, got {b_late}"
    assert a_late < 20, f"vial A must stay quiet in phase 3 (no ROI aliasing), got {a_late}"
    assert "mis_registration" in set(edf["event"]), "the lattice-pitch lock should be rejected"


def test_max_shift_is_measured_within_a_face_not_across_faces(tmp_path):
    """Regression: identical two-face coordinates collapsed `max_shift` to 0.0.

    The hand-drawing flow gives face B face A's polygons VERBATIM -- one drawing covers the whole
    drum -- so once the pitch was measured over both faces POOLED, every vial had a twin at
    distance exactly 0. On the real 32-vial bundle 16 of the 31 sorted centre gaps were 0.0, the
    tightest pitch became 0.0, `max_shift` went with it, and the guard then rejected EVERY shift
    including (0.0, 0.0): ROI drift correction silently stopped for the entire run, visible only
    as a stream of `mis_registration` events.
    """
    from flygym_tracker.calibration import build_two_face_calibration_from_polygons
    from flygym_tracker.pipeline import DEFAULT_MAX_SHIFT_FRAC

    # Two vials 60 px apart, drawn once and shared by both faces -- the real geometry.
    polys = [[[10, 20], [50, 20], [50, 220], [10, 220]],
             [[70, 20], [110, 20], [110, 220], [70, 220]]]
    calib, masks, overlays = build_two_face_calibration_from_polygons(
        polys, _base(), (_base().shape[1], _base().shape[0]), faces=("A", "B"))
    cdir = tmp_path / "calib"
    cdir.mkdir()
    save_calibration(calib, masks, str(cdir), overlay=overlays)
    calibration = load_calibration(str(cdir))

    centres = [(v.x + v.w / 2.0) for v in calibration.faces["A"].vials]
    assert len(calibration.faces) == 2
    assert [round(c) for c in centres] == [round(c) for c in
                                           [(v.x + v.w / 2.0) for v in calibration.faces["B"].vials]],         "the two faces must share coordinates -- that is what made this bug possible"

    pipe = TrackerPipeline(
        load_config(overrides={"binning": {"bin_seconds": 1}}), calibration,
        _SeqSource([_base()]), ActivityLogger(str(tmp_path / "out"), run_id="ms", fmt="csv"),
        clock="index", pixel_threshold=30, enter_threshold=15, exit_threshold=8,
    )

    assert pipe.max_shift > 0.0, "a zero max_shift rejects every shift, even (0.0, 0.0)"
    within_face_pitch = 60.0
    assert pipe.max_shift == pytest.approx(DEFAULT_MAX_SHIFT_FRAC * within_face_pitch, rel=0.05)
