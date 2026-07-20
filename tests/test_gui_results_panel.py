"""The measurement is on screen while it is being made -- and it matches the file.

WHAT WAS MISSING. The window showed frames, elapsed time, fps, a rotation count and a strip of
cells whose brightness tracked activity. Every one of those says THE MACHINE IS RUNNING. None of
them is a measurement. An operator could watch a three-day experiment start to finish without once
seeing a number that would reach the results -- so a threshold set too high, a vial that never
reports, or a face never identified all look exactly like a healthy run until the CSV is opened
afterwards, by which time the flies are gone.

THE LOAD-BEARING TEST IS THE LAST ONE: it drives a REAL `TrackerPipeline` over synthetic frames and
asserts that the rows the panel displays are the rows that landed in activity.csv, field for field.
A results view that showed plausible-but-different numbers would be worse than showing none.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from flygym_tracker.gui.results_panel import COLUMNS, MAX_ROWS, ResultsPanel   # noqa: E402


def _row(vial_id=1, face="A", motion=12, elapsed=10.0, present=True):
    return {"run_id": "r", "bin_start_iso": "", "bin_end_iso": "", "elapsed_s": elapsed,
            "face": face, "vial_id": vial_id, "row": 0, "col": 0, "present": present,
            "n_stationary_frames": 30, "n_rotating_frames": 0, "motion_px_sum": motion,
            "active_fraction_mean": 0.25, "lit_area_px": 900}


# =============================================================================================
# The live grid
# =============================================================================================
def test_a_vial_the_pipeline_did_not_report_stays_blank_rather_than_reading_zero(qapp):
    """"No reading" and "a reading of zero" are different facts. On this rig the distinction is
    load-bearing: only one drum face is visible at a time, so the other sixteen SHOULD be blank,
    and drawing them as 0 would be the display inventing a measurement."""
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {0: (250, 900, 0.3), 1: (0, 900, 0.0)}})
    assert panel.grid.cells[0].text() == "250"
    assert panel.grid.cells[1].text() == "0", "a real zero must still be shown"
    assert panel.grid.cells[5].text() == "", "an unreported vial was drawn as a number"


def test_the_live_readout_is_labelled_as_not_being_in_the_file(qapp):
    """It changes faster than it can be read and it is not binned. Showing it beside the recorded
    rows without saying which is which invites reading the fast one as the measurement -- and the
    fast one is the only one that is NOT in the output."""
    panel = ResultsPanel()
    panel.set_progress({"vial_results": {0: (5, 900, 0.1)}, "face": "B", "pixel_threshold": 12.0})
    note = panel.live_note.text()
    assert "not in the file" in note
    assert "pixel threshold 12.0" in note, \
        "the threshold that decides what counts as motion is not shown beside the numbers"


# =============================================================================================
# The recorded rows
# =============================================================================================
def test_completed_bins_appear_as_rows(qapp):
    panel = ResultsPanel()
    panel.add_bin({"records": [_row(vial_id=1), _row(vial_id=2)]})
    assert panel.table.rowCount() == 2
    assert "2 row(s) written" in panel.recorded_note.text()


def test_the_columns_are_the_csv_field_names(qapp):
    """What is on screen and what is in the file cannot drift apart if they are keyed by the same
    field names."""
    from flygym_tracker.types import ACTIVITY_COLUMNS

    for _label, field in COLUMNS:
        assert field in ACTIVITY_COLUMNS, "%r is not a column of activity.csv" % field


def test_the_table_is_bounded_so_a_three_day_run_cannot_grow_it_forever(qapp):
    """~26000 rows per vial at 10 s bins. The file has them all; this pane is for watching."""
    panel = ResultsPanel()
    for _ in range(MAX_ROWS + 60):
        panel.add_bin({"records": [_row()]})
    assert panel.table.rowCount() == MAX_ROWS
    assert "%d row(s) written" % (MAX_ROWS + 60) in panel.recorded_note.text(), \
        "the count must still report everything written, not just what is on screen"


def test_an_empty_bin_does_not_count_as_one(qapp):
    panel = ResultsPanel()
    panel.add_bin({"records": []})
    assert panel.table.rowCount() == 0
    assert "no bins finished yet" in panel.recorded_note.text()


# =============================================================================================
# The claim that matters: what is shown IS what was written
# =============================================================================================
def test_the_rows_on_screen_are_the_rows_in_activity_csv(qapp, tmp_path):
    """END TO END through a REAL `TrackerPipeline`: run it over synthetic frames with the panel
    wired to the same bin observer the window uses, then read activity.csv off disk and compare.

    A results view that showed plausible-but-different numbers would be worse than showing none --
    it would be a second, unowned implementation of the measurement, and the operator would have no
    way to know which one to believe.
    """
    import csv

    import numpy as np

    from flygym_tracker.calibration import (build_two_face_calibration_from_polygons,
                                            load_calibration, save_calibration)
    from flygym_tracker.logger import ActivityLogger
    from flygym_tracker.pipeline import TrackerPipeline
    from flygym_tracker.config import load_config
    from flygym_tracker.types import Frame

    rng = np.random.default_rng(4)
    frames = [rng.integers(40, 60, size=(120, 240), dtype=np.uint8) for _ in range(60)]

    class Source:
        fps = 30.0

        def __init__(self):
            self.i = 0

        def open(self):
            pass

        def close(self):
            pass

        def read(self):
            if self.i >= len(frames):
                return None
            # The REAL `types.Frame`, not a stand-in: the pipeline reads `t_wall_iso` too, and a
            # duck-typed frame that happens to satisfy today's field accesses is a test that
            # passes for a reason the shipped code does not share.
            frame = Frame(image=frames[self.i], index=self.i, t_monotonic=self.i / 30.0,
                          t_wall_iso="2026-01-01T00:00:%06.3f" % (self.i / 30.0))
            self.i += 1
            return frame

    polygons = [[[10 + 40 * i, 20], [40 + 40 * i, 20], [40 + 40 * i, 90], [10 + 40 * i, 90]]
                for i in range(3)]
    calib, masks, _ = build_two_face_calibration_from_polygons(
        polygons, frames[0], (240, 120), faces=("A", "B"))
    # SAVED AND RELOADED so the illumination-mask paths are resolved for this machine -- the same
    # round trip the run path makes, rather than a hand-built object the pipeline cannot read.
    bundle = str(tmp_path / "calib")
    save_calibration(calib, masks, bundle)
    calib = load_calibration(bundle)
    out = str(tmp_path / "out")
    logger = ActivityLogger(output_dir=out, run_id="test", fmt="csv", rolling="daily", meta={})
    config = load_config(overrides={"binning": {"bin_seconds": 0.5},
                                    "activity": {"pixel_threshold": 1.0},
                                    "rotation": {"detector": "adaptive"}})
    pipeline = TrackerPipeline(config=config, calibration=calib, source=Source(),
                               logger=logger, marker_detector=None, clock="index")

    panel = ResultsPanel()
    # EXACTLY WHAT `RunWorker._on_bin` DOES, so this tests the shipped shaping of the payload.
    pipeline.add_bin_observer(lambda p: panel.add_bin(
        {"records": [r.as_row() for r in (p.get("records") or [])]}))
    pipeline.run()

    # `rolling="daily"` names the file activity_YYYYMMDD.csv, and a run can straddle midnight, so
    # every day file is read in name order rather than one being guessed at.
    import glob

    paths = sorted(glob.glob(str(tmp_path / "out" / "activity*.csv")))
    assert paths, "the run wrote no activity csv to compare against"
    written = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            written.extend(csv.DictReader(f))
    assert written, "activity.csv is empty"
    assert panel.table.rowCount() == len(written), \
        "the panel shows %d rows, the file has %d" % (panel.table.rowCount(), len(written))

    labels = [label for label, _ in COLUMNS]
    fields = [field for _, field in COLUMNS]
    for index, record in enumerate(written):
        for column, field in enumerate(fields):
            shown = panel.table.item(index, column).text()
            expected = record[field]
            # The panel formats floats for reading; compare numerically where the CSV is numeric.
            try:
                assert float(shown) == pytest.approx(float(expected), abs=5e-4), (
                    "row %d %s: screen %r vs file %r" % (index, labels[column], shown, expected))
            except ValueError:
                assert shown == expected, (
                    "row %d %s: screen %r vs file %r" % (index, labels[column], shown, expected))
