"""The two ways a three-day unattended run used to end badly, and the guards that stop them.

Both were found by auditing for failures that only appear after HOURS, which is exactly the class of
bug this rig cannot afford: nobody is watching, and the flies are only good once.

  1. THE CAMERA DIES AND THE RUN DOES NOT NOTICE. `_read_frame` retries a hiccup, which is right;
     the read loop then treated a PERMANENT failure as another hiccup and skipped it forever. An
     unplugged camera at hour 20 became a hot loop that processed no frames, wrote no rows and never
     returned -- while the window still said the run was in progress.

  2. THE SPREADSHEET EXPORT TAKES THE CAMERA WITH IT. `logger.close()` regenerates the .xlsx copies
     and can fail on its own terms (a multi-day behaviour.csv exceeds the format's 1,048,576-row
     ceiling). It ran one line before `source.close()` in the same teardown, unguarded -- so the
     failure stranded the exclusive USB3 handle and the NEXT session could not open the camera.

These tests are deliberately about CONTROL FLOW, not about pixels: what the run does when the frames
stop, and what teardown still guarantees when part of it throws.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from flygym_tracker.logger import ActivityLogger
from flygym_tracker.types import Frame

from test_pipeline import P, FPS, FakeSource, _calibration, _config, _one
from flygym_tracker.pipeline import TrackerPipeline


class DyingSource(FakeSource):
    """Serves `n_good` frames, then raises on every read from then on -- a camera that is GONE.

    Distinct from `FakeSource(raise_on_calls=...)`, which models isolated hiccups: this never
    recovers, which is the case the read loop had no answer for.
    """

    def __init__(self, frames: List[np.ndarray], n_good: int, fps: float = FPS):
        super().__init__(frames, fps=fps)
        self._n_good = int(n_good)

    def read(self) -> Optional[Frame]:
        if self._read_calls >= self._n_good:
            self._read_calls += 1
            raise IOError("camera is gone (simulated permanent failure)")
        return super().read()


def _quiet(n: int) -> List[np.ndarray]:
    return [P.copy() for _ in range(n)]


def _pipe(tmp_path, source, *, fmt="csv", max_consecutive=5, logger=None):
    logger = logger or ActivityLogger(output_dir=tmp_path, run_id="test_run", fmt=fmt)
    return TrackerPipeline(
        _config(), _calibration(tmp_path), source, logger,
        reference_frames={"A": P.copy()}, clock="index", read_retry_sleep=0.0,
        max_consecutive_read_errors=max_consecutive,
    )


# =============================================================================================
# 1. A camera that stops delivering ends the run
# =============================================================================================
def test_a_dead_camera_ends_the_run_instead_of_retrying_forever(tmp_path):
    """THE REGRESSION: this test hanging IS the bug. Before the cap, the read loop skipped a failed
    read and looped again with no limit, so a permanently dead camera spun here until killed."""
    source = DyingSource(_quiet(6), n_good=6)
    pipe = _pipe(tmp_path, source, max_consecutive=5)

    result = pipe.run()

    assert result["stopped_reason"] == "camera_lost", \
        "a permanently dead camera did not stop the run"
    assert result["frames_processed"] == 6, "the frames read before the failure were lost"


def test_the_dead_camera_run_still_flushes_its_data_and_releases_the_camera(tmp_path):
    """Stopping is only half the requirement. The run must end through the NORMAL teardown, so the
    measured bins reach the disk and the exclusive USB3 handle is handed back."""
    source = DyingSource(_quiet(6), n_good=6)
    pipe = _pipe(tmp_path, source, max_consecutive=5)

    pipe.run()

    assert source.closed, "the camera was not released when the run gave up on it"
    events = pd.read_csv(_one(tmp_path, "events_*.csv"), keep_default_na=False)
    assert "camera_lost" in set(events["event"]), \
        "nothing in events.csv says why the run stopped"


def test_an_isolated_hiccup_still_does_not_end_the_run(tmp_path):
    """THE OTHER HALF OF THE CONTRACT, and the reason the counter is CONSECUTIVE. A rig that gave up
    on the first dropped frame would be far worse than one that retried forever."""
    source = FakeSource(_quiet(12), fps=FPS, raise_on_calls={2, 5, 9})
    pipe = _pipe(tmp_path, source, max_consecutive=5)

    result = pipe.run()

    assert result["stopped_reason"] == "eof", "scattered hiccups ended the run"
    assert result["frames_processed"] == 12, "a hiccup cost a frame it should have retried"


def test_scattered_hiccups_never_accumulate_into_a_false_camera_loss(tmp_path):
    """The counter measures UNBROKEN silence: 6 isolated failures with a cap of 3 must not trip it,
    because a delivered frame in between proves the camera is alive."""
    source = FakeSource(_quiet(14), fps=FPS, raise_on_calls={2, 4, 6, 8, 10, 12})
    pipe = _pipe(tmp_path, source, max_consecutive=3)

    result = pipe.run()

    assert result["stopped_reason"] == "eof"
    assert result["frames_processed"] == 14


# =============================================================================================
# 2. The results export never keeps the camera
# =============================================================================================
class ExplodingLogger(ActivityLogger):
    """A logger whose close() fails the way a too-large workbook rebuild does."""

    def close(self) -> None:
        raise ValueError("This sheet is too large! Your sheet size is: 2000000, 16384")


def test_the_camera_is_released_even_when_closing_the_results_fails(tmp_path):
    """THE STRANDED-HANDLE REGRESSION. `logger.close()` sat one unguarded line above
    `source.close()`, so a workbook that could not be written cost the next session its camera."""
    source = FakeSource(_quiet(6), fps=FPS)
    pipe = _pipe(tmp_path, source,
                 logger=ExplodingLogger(output_dir=tmp_path, run_id="test_run", fmt="csv"))

    pipe.run()      # must not raise: the export failing is not the run failing

    assert source.closed, "a failed results export stranded the exclusive camera handle"


def test_a_workbook_that_cannot_be_written_still_lets_the_run_be_stamped_finished(tmp_path):
    """`close()` stamps `stop_iso` AFTER flushing the workbooks. When the flush raised, the run's
    own metadata was left saying it had never stopped -- so a completed run looked like a crashed
    one. The CSV is the result; the workbook is a convenience, and it does not get to decide."""
    logger = ActivityLogger(output_dir=tmp_path, run_id="test_run", fmt="both")

    def boom(_csv_path):
        raise ValueError("This sheet is too large! Your sheet size is: 2000000, 16384")

    logger._rewrite_xlsx = boom
    logger.log_event(None)                       # touch the logger so there is something to flush
    logger._dirty_csvs.add(tmp_path / "activity_test_run_20260718.csv")

    logger.close()                               # must not raise

    import json

    meta = json.loads(_one(tmp_path, "run_meta_*.json").read_text(encoding="utf-8"))
    assert meta.get("stop_iso"), "a failed workbook rebuild left the run unstamped"


def test_the_csv_is_complete_even_when_the_workbook_step_fails(tmp_path):
    """What survives matters more than what does not: every row must already be durable in the CSV
    before the workbook is attempted, so losing the workbook loses nothing."""
    source = FakeSource(_quiet(30), fps=FPS)
    logger = ActivityLogger(output_dir=tmp_path, run_id="test_run", fmt="both")
    logger._rewrite_xlsx = lambda _p: (_ for _ in ()).throw(ValueError("too large"))
    pipe = _pipe(tmp_path, source, logger=logger)

    pipe.run()

    activity = pd.read_csv(_one(tmp_path, "activity_*.csv"))
    assert not activity.empty, "the measurement was lost with the workbook"
