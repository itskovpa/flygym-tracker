"""Fly tracking reaches the data: behaviour.csv, one row per vial per dwell.

THE UNIT IS A DWELL, NOT A BIN, and that is why `_harvest_dwell` exists separately. A bin is 10 s
and a dwell is ~2 s, so a bin spans several dwells and usually a face change; taking the summaries
when the BIN rolls would attribute every dwell in it to whichever face happened to be showing at
the end. One tracker per dwell is `fly_tracking`'s rule, so one row per dwell is the honest unit.
"""
from __future__ import annotations

import csv
import glob
import os

import numpy as np
import pytest

from flygym_tracker.calibration import (build_two_face_calibration_from_polygons,
                                        load_calibration, save_calibration)
from flygym_tracker.config import load_config
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline
from flygym_tracker.types import BEHAVIOUR_COLUMNS, Frame

H, W = 240, 400
FPS = 20.0


def _frame(i, rotating=False):
    """Vial 1: a fly climbing 3 px/frame. Vial 2: a fly sitting still. Vial 3: empty."""
    frame = np.full((H, W), 200, dtype=np.uint8)
    if rotating:
        frame = np.roll(frame, 17 * i, axis=1)
        frame[::3, :] = 60
        return frame
    y = 190 - 3 * (i % 40)
    frame[y:y + 8, 55:65] = 40
    frame[120:128, 175:185] = 40
    return frame


class _Source:
    fps = FPS

    def __init__(self, frames):
        self._frames = frames
        self.i = 0

    def open(self):
        pass

    def close(self):
        pass

    def read(self):
        if self.i >= len(self._frames):
            return None
        frame = Frame(image=self._frames[self.i], index=self.i, t_monotonic=self.i / FPS,
                      t_wall_iso="2026-01-01T00:00:%06.3f" % (self.i / FPS))
        self.i += 1
        return frame


def _run(tmp_path, frames, **kwargs):
    polygons = [[[20 + 120 * i, 40], [100 + 120 * i, 40],
                 [100 + 120 * i, 200], [20 + 120 * i, 200]] for i in range(3)]
    calib, masks, _ = build_two_face_calibration_from_polygons(
        polygons, frames[0], (W, H), faces=("A", "B"))
    bundle = str(tmp_path / "calib")
    save_calibration(calib, masks, bundle)
    out = str(tmp_path / "out")
    logger = ActivityLogger(output_dir=out, run_id="t", fmt="csv", rolling="daily", meta={})
    config = load_config(overrides={"binning": {"bin_seconds": 1.0},
                                    "activity": {"pixel_threshold": 1.0},
                                    "rotation": {"detector": "adaptive"}})
    pipeline = TrackerPipeline(config=config, calibration=load_calibration(bundle),
                               source=_Source(frames), logger=logger, marker_detector=None,
                               clock="index", **kwargs)
    return pipeline.run(), out


def _rows(out):
    path = os.path.join(out, "behaviour.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _two_dwells():
    return ([_frame(i) for i in range(50)] + [_frame(i, True) for i in range(6)]
            + [_frame(i) for i in range(50)])


# =============================================================================================
# It reaches the file, with the right numbers
# =============================================================================================
def test_a_run_writes_behaviour_rows(tmp_path):
    summary, out = _run(tmp_path, _two_dwells())
    rows = _rows(out)
    assert rows, "the run produced no behaviour.csv"
    assert summary["n_behaviour_records"] == len(rows)
    assert summary["behaviour_write_errors"] == 0
    assert set(rows[0]) == set(BEHAVIOUR_COLUMNS)


def test_a_fly_climbing_at_a_known_speed_reaches_the_file_at_that_speed(tmp_path):
    """3 px/frame at 20 fps is 60 px/s by construction. If the axis, the timestamps or the harvest
    were wrong, every column in this file would still look like a plausible number."""
    _summary, out = _run(tmp_path, _two_dwells())
    climbing = [r for r in _rows(out) if r["vial_id"] == "1"]
    assert climbing, "the climbing vial produced no row"
    for row in climbing:
        assert float(row["median_speed"]) == pytest.approx(60.0, rel=0.1)


def test_an_empty_vial_reports_nan_and_not_zero(tmp_path):
    """An empty vial has no height, which is NOT the same claim as "its flies are at the bottom".
    A zero here would be the file inventing a measurement of nothing."""
    _summary, out = _run(tmp_path, _two_dwells())
    empty = [r for r in _rows(out) if r["vial_id"] == "3"]
    assert empty, "the empty vial produced no row"
    for row in empty:
        assert row["mean_height"] in ("nan", ""), row["mean_height"]
        assert row["median_path_length"] in ("nan", ""), row["median_path_length"]


def test_a_still_fly_reports_no_movement(tmp_path):
    _summary, out = _run(tmp_path, _two_dwells())
    still = [r for r in _rows(out) if r["vial_id"] == "2"]
    assert still
    for row in still:
        assert float(row["median_speed"]) == pytest.approx(0.0, abs=1.0)


# =============================================================================================
# The unit is a dwell
# =============================================================================================
def test_each_dwell_produces_its_own_row_per_vial(tmp_path):
    """Two dwells separated by a rotation must give two rows per vial, not one averaged over both
    -- across a rotation the flies are shaken and every identity is lost."""
    summary, out = _run(tmp_path, _two_dwells())
    assert summary["n_rotations"] >= 1, "the synthetic rotation was not detected"
    per_vial = {}
    for row in _rows(out):
        per_vial.setdefault(row["vial_id"], []).append(row)
    assert per_vial, "no rows at all"
    for vial_id, rows in per_vial.items():
        assert len(rows) >= 2, ("vial %s produced %d row(s), expected one per dwell"
                                % (vial_id, len(rows)))


def test_the_last_dwell_is_not_thrown_away_at_the_end_of_a_run(tmp_path):
    """A run stopped mid-dwell holds frames that were measured -- the same reason the accumulator
    flushes its final, partial bin rather than dropping it."""
    summary, out = _run(tmp_path, [_frame(i) for i in range(40)])
    assert summary["n_rotations"] == 0, "this run was supposed to be a single dwell"
    assert _rows(out), "the only dwell of the run produced no rows"


# =============================================================================================
# It never costs the activity measurement
# =============================================================================================
def _activity(out):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "activity*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return [(r["vial_id"], r["elapsed_s"], r["motion_px_sum"]) for r in rows]


def test_activity_is_unchanged_by_tracking(tmp_path):
    """Tracking is an ADDITION. If turning it on moved a single activity number, the addition
    would have changed the primary result -- and three days of data would be incomparable with
    every run recorded before it."""
    frames = _two_dwells()
    with_tracking, out_a = _run(tmp_path / "a", frames, track_flies=True)
    without, out_b = _run(tmp_path / "b", frames, track_flies=False)

    assert with_tracking["n_activity_records"] == without["n_activity_records"]
    assert with_tracking["n_rotations"] == without["n_rotations"]
    assert _activity(out_a) == _activity(out_b), "tracking changed the activity measurement"


def test_tracking_off_writes_no_behaviour_file(tmp_path):
    """Rather than a file full of empty columns, which reads like a measurement that failed."""
    summary, out = _run(tmp_path, _two_dwells(), track_flies=False)
    assert summary["n_behaviour_records"] == 0
    assert not os.path.exists(os.path.join(out, "behaviour.csv"))


def test_the_run_summary_says_how_much_was_actually_tracked(tmp_path):
    """A behavioural figure computed from 60% of the frames is a real measurement of those frames
    -- as long as the number saying so is on the record."""
    summary, _out = _run(tmp_path, _two_dwells())
    stats = summary["tracking"]
    assert stats["frames_tracked"] > 0
    assert 0.0 <= stats["fraction_tracked"] <= 1.0
    assert stats["frames_tracked"] + stats["frames_dropped"] > 0
    assert stats["vial_failures"] == 0


def test_a_failed_behaviour_write_is_counted_and_never_silent(tmp_path):
    """MEASURED, on the first end-to-end run: `log_behaviour` raised NameError, a broad except
    swallowed it, and the run reported "0 behaviour rows" while the summaries had been produced
    for every vial twice over. Measured rows were thrown away and the summary said nothing had
    been measured -- the one outcome worse than crashing, because it looks like a rig with no
    flies in it."""
    frames = _two_dwells()
    polygons = [[[20 + 120 * i, 40], [100 + 120 * i, 40],
                 [100 + 120 * i, 200], [20 + 120 * i, 200]] for i in range(3)]
    calib, masks, _ = build_two_face_calibration_from_polygons(
        polygons, frames[0], (W, H), faces=("A", "B"))
    bundle = str(tmp_path / "calib")
    save_calibration(calib, masks, bundle)
    out = str(tmp_path / "out")
    logger = ActivityLogger(output_dir=out, run_id="t", fmt="csv", rolling="daily", meta={})

    def explode(rows):
        raise RuntimeError("disk full")

    logger.log_behaviour = explode
    config = load_config(overrides={"binning": {"bin_seconds": 1.0},
                                    "activity": {"pixel_threshold": 1.0},
                                    "rotation": {"detector": "adaptive"}})
    pipeline = TrackerPipeline(config=config, calibration=load_calibration(bundle),
                               source=_Source(frames), logger=logger, marker_detector=None,
                               clock="index")
    summary = pipeline.run()

    assert summary["behaviour_write_errors"] > 0, "a failed write was not counted"
    assert summary["n_activity_records"] > 0, "a behaviour failure took the activity run down"

    with open(os.path.join(out, "events.csv"), newline="", encoding="utf-8") as f:
        events = [row["event"] for row in csv.DictReader(f)]
    assert "behaviour_write_failed" in events, "the failure never reached events.csv"
