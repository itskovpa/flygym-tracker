"""Fly tracking beside the measurement loop: correct numbers, and never in its way.

The claims that matter, in order:

  * the numbers are RIGHT -- a fly moved at a known speed through a known height must come back at
    that speed and that height, or every behavioural parameter downstream is decoration;
  * tracking NEVER blocks the pipeline, and what it dropped is counted rather than swallowed;
  * a dwell boundary throws identities away, because across a rotation the flies are shaken and
    linking through it would be fiction.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from flygym_tracker.fly_runner import FlyTrackingPool, vial_axis

H, W = 200, 300
FPS = 20.0


def _mask(x0, x1):
    mask = np.zeros((H, W), dtype=bool)
    mask[30:170, x0:x1] = True
    return mask


def _vials():
    return {
        1: (_mask(20, 90), vial_axis([[20, 30], [90, 30], [90, 170], [20, 170]])),
        2: (_mask(150, 220), vial_axis([[150, 30], [220, 30], [220, 170], [150, 170]])),
    }


def _frame(y1=None, y2=100):
    """A lit frame with a dark fly in vial 1 at `y1` and one in vial 2 at `y2`."""
    frame = np.full((H, W), 200, dtype=np.uint8)
    if y1 is not None:
        frame[y1:y1 + 8, 45:55] = 40
    if y2 is not None:
        frame[y2:y2 + 8, 175:185] = 40
    return frame


def _drain(pool, timeout=3.0):
    """Wait for the workers to catch up. They are threads, so a result is not ready on return."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pool.take_summaries():
            return True
        time.sleep(0.02)
    return False


# =============================================================================================
# The numbers
# =============================================================================================
def test_a_fly_climbing_at_a_known_speed_is_measured_at_that_speed():
    """THE LOAD-BEARING TEST. 3 px per frame at 20 fps is 60 px/s by construction; if the tracker
    or the axis or the timestamps were wrong this is where it shows, and every parameter built on
    them -- climbing score, path length, speed -- would be wrong in a way that still looks like a
    plausible number in a CSV."""
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=2)
    pool.start()
    try:
        for i in range(40):
            pool.submit(_frame(y1=160 - 3 * i), i / FPS)
            time.sleep(0.005)
        time.sleep(1.0)
        summaries = pool.take_summaries()
        assert 1 in summaries, "the climbing vial produced no summary"
        assert summaries[1]["median_speed"] == pytest.approx(60.0, rel=0.1)
    finally:
        pool.close()


def test_a_still_fly_reports_no_speed_and_no_climbing():
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=1)
    pool.start()
    try:
        for i in range(30):
            pool.submit(_frame(y1=None, y2=100), i / FPS)
            time.sleep(0.005)
        time.sleep(1.0)
        summaries = pool.take_summaries()
        assert 2 in summaries
        assert summaries[2]["median_speed"] == pytest.approx(0.0, abs=1.0)
        assert summaries[2]["frac_above_mid"] == pytest.approx(0.0, abs=1e-6)
    finally:
        pool.close()


def test_the_axis_puts_the_food_end_first_for_both_faces():
    """Flies climb against gravity; gravity is down; down is +y in an image. The drum turning 180
    degrees swaps which face is seen but does not move gravity, so "up the tube" is toward smaller
    y whichever face is showing -- no per-face flip, and a flip would invert every climbing score.
    """
    (x0, y0), (x1, y1) = vial_axis([[10, 30], [50, 30], [50, 170], [10, 170]])
    assert y0 > y1, "the axis does not start at the bottom of the image"
    assert x0 == x1 == 30.0, "the axis is not up the middle of the tube"


# =============================================================================================
# It stays out of the pipeline's way
# =============================================================================================
def test_submitting_never_blocks_and_drops_are_counted():
    """Activity is the primary result and it is what a three-day experiment is for. A tracking
    figure computed from 60% of the frames is still a real measurement of those frames -- as long
    as the number saying so is on the record rather than swallowed."""
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=1, queue_depth=1)
    # NOT started: nothing consumes the queue, so it fills at once and every later submit must be
    # refused instantly rather than waiting for a worker that is never coming.
    started = time.monotonic()
    accepted = sum(1 for i in range(50) if pool.submit(_frame(y1=100), i / FPS))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, "submit blocked for %.2f s -- the pipeline would have stalled" % elapsed
    assert accepted < 50, "nothing was dropped, so this proves nothing about backpressure"
    stats = pool.stats()
    assert stats["frames_dropped"] == 50 - accepted
    assert stats["frames_tracked"] == accepted
    assert 0.0 <= stats["fraction_tracked"] <= 1.0
    pool.close()


def test_a_frame_reaches_every_vial_or_none_of_them():
    """A frame that reached half the vials would make vials silently incomparable inside the same
    bin, and "vial 3 saw 40 frames, vial 4 saw 39" is a difference nobody would think to look for
    when the numbers disagree later."""
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=2)
    pool.start()
    try:
        for i in range(30):
            pool.submit(_frame(y1=120), i / FPS)
            time.sleep(0.01)
        time.sleep(1.0)
        summaries = pool.take_summaries()
        counts = {gvid: row["n_frames"] for gvid, row in summaries.items()}
        assert len(set(counts.values())) == 1, "vials saw different numbers of frames: %r" % counts
    finally:
        pool.close()


# =============================================================================================
# Dwell boundaries
# =============================================================================================
def test_a_rotation_throws_every_identity_away():
    """`fly_tracking`'s own rule: across a rotation the flies are shaken, the pose changes and
    every identity is lost, so linking across that boundary would be fiction."""
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=1)
    pool.start()
    try:
        for i in range(20):
            pool.submit(_frame(y1=150 - 2 * i), i / FPS)
            time.sleep(0.005)
        time.sleep(0.8)
        assert pool.take_summaries(), "nothing was tracked before the rotation"

        pool.reset_dwell()
        time.sleep(0.5)
        assert pool.tracks() == {} or all(not paths for paths in pool.tracks().values()), \
            "tracks survived a rotation"
    finally:
        pool.close()


def test_tracks_come_back_as_polylines_for_the_overlay():
    """A track is a PATH. Drawing loose points would show where flies have been but not which
    movements were one fly, which is the thing a trajectory is for."""
    pool = FlyTrackingPool(_vials(), fps=FPS, n_workers=1)
    pool.start()
    try:
        for i in range(25):
            pool.submit(_frame(y1=150 - 3 * i), i / FPS)
            time.sleep(0.005)
        time.sleep(1.0)
        tracks = pool.tracks()
        assert 1 in tracks and tracks[1], "the climbing vial produced no track"
        path = tracks[1][0]
        assert len(path) >= 2
        assert all(len(point) == 2 for point in path)
        # It climbed, so the path must run upward in image coordinates.
        assert path[-1][1] < path[0][1], "the track does not follow the fly up the tube"
    finally:
        pool.close()


# =============================================================================================
# Failures are contained
# =============================================================================================
def test_one_vials_failure_does_not_stop_the_others():
    """A raise on the worker thread would stop tracking every vial that worker owns, silently."""
    vials = _vials()
    vials[3] = (np.zeros((10, 10), dtype=bool), ((0.0, 0.0), (0.0, 1.0)))   # wrong-shaped mask
    pool = FlyTrackingPool(vials, fps=FPS, n_workers=1)
    pool.start()
    try:
        for i in range(20):
            pool.submit(_frame(y1=140 - 2 * i), i / FPS)
            time.sleep(0.005)
        time.sleep(1.0)
        summaries = pool.take_summaries()
        assert 1 in summaries, "a broken vial took the healthy ones down with it"
        assert pool.stats()["vial_failures"] > 0, "the failure was not counted"
    finally:
        pool.close()


def test_a_pool_with_no_vials_is_harmless():
    pool = FlyTrackingPool({}, fps=FPS)
    pool.start()
    assert pool.submit(_frame(), 0.0) is False
    assert pool.take_summaries() == {}
    assert pool.tracks() == {}
    pool.close()
