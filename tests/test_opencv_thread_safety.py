"""No two threads may be inside OpenCV at once, and the activity path may not be inside it at all.

THE CRASH THIS PREVENTS. `cv_setup.CV_LOCK` exists because concurrent OpenCV calls corrupt the
allocator and take the process down with an access violation. The rotation detector, the tracking
workers and the video recorder all took it. `activity.per_frame_activity` did not -- and it is the
most frequent OpenCV call in the program, once per vial per frame on the pipeline thread. So every
run with fly tracking enabled had 32 unsynchronised chances per frame to be inside OpenCV while a
tracking worker was, and the app died with 0xc0000005 inside python314.dll. Activity-only runs never
crashed, because then nothing else was in OpenCV -- which is exactly the pattern seen on the rig.

Captured live under faulthandler: the pipeline thread in `activity.py per_frame_activity` while
`flygym-track-1` was in `fly_tracking.detect_flies`.

The fix is not to lock the activity path but to take it OUT of OpenCV, because `fly_runner`'s own
rule is that tracking must never jeopardise the activity measurement -- and a shared lock would make
the primary measurement wait on the optional one.
"""
from __future__ import annotations

import ast
import pathlib

import cv2
import numpy as np
import pytest

from flygym_tracker.activity import per_frame_activity

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "flygym_tracker"


def _imports(module_path: pathlib.Path):
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_activity_path_does_not_import_opencv():
    """THE INVARIANT, checked statically so it cannot regress by accident. If a future change wants
    OpenCV here it must also answer for the lock -- and this test is where that argument happens."""
    assert "cv2" not in _imports(SRC / "activity.py"), (
        "activity.py imports cv2 again: the per-vial hot path is back inside OpenCV, "
        "unsynchronised against the tracking workers")


@pytest.mark.parametrize("shape", [(4, 4), (37, 61), (400, 160)])
def test_the_replacement_is_bit_identical_to_cv2_absdiff(shape):
    """Numerically this must be a no-op. It is a concurrency fix, not a measurement change."""
    rng = np.random.default_rng(0)
    cur = rng.integers(0, 256, shape, dtype=np.uint8)
    prev = rng.integers(0, 256, shape, dtype=np.uint8)
    mask = np.ones(shape, dtype=bool)

    expected_motion = int(np.count_nonzero(cv2.absdiff(cur, prev) > 15.0))
    motion_px, lit, frac = per_frame_activity(cur, prev, mask, 15.0)

    assert motion_px == expected_motion
    assert lit == cur.size
    assert frac == pytest.approx(expected_motion / cur.size)


@pytest.mark.parametrize("a,b", [(0, 255), (255, 0), (0, 0), (255, 255), (7, 7), (1, 254)])
def test_no_unsigned_wrap_at_the_extremes(a, b):
    """`maximum - minimum` cannot underflow; plain subtraction would wrap and report huge motion."""
    cur = np.full((8, 8), a, np.uint8)
    prev = np.full((8, 8), b, np.uint8)
    mask = np.ones((8, 8), dtype=bool)
    motion_px, _lit, _frac = per_frame_activity(cur, prev, mask, 15.0)
    expected = int(np.count_nonzero(cv2.absdiff(cur, prev) > 15.0))
    assert motion_px == expected, "wrapped on unsigned subtraction at %d vs %d" % (a, b)


def test_every_other_opencv_caller_still_takes_the_lock():
    """The three modules that DO belong inside OpenCV must keep holding CV_LOCK when they are."""
    for name in ("fly_runner.py", "adaptive_rotation.py", "video_recorder.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "with CV_LOCK:" in text, "%s stopped serialising its OpenCV work" % name


def test_the_pipeline_locks_its_per_dwell_opencv_work():
    """`find_strips` and `identify_face` are OpenCV on the pipeline thread, once per dwell."""
    text = (SRC / "pipeline.py").read_text(encoding="utf-8")
    assert text.count("with CV_LOCK:") >= 2, \
        "the per-dwell marker/band OpenCV calls are no longer serialised"
