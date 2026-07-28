"""A dwell that never ends must stop costing memory, and the cap must be switchable off.

WHAT THIS PROTECTS. A dwell is normally ~2 s: the drum flips, `reset_dwell` throws the trackers
away, the next dwell starts. Nothing bounded a dwell that never ENDED -- a jammed motor, a missed
flip -- so the trackers accumulated one `FrameStats` per vial per frame for the rest of the run.
Measured on real footage with a parked drum: ~0.6 objects/frame/vial and about 4 GB over three
days, with throughput decaying as it grew until the process died. An unattended experiment lost to
a stuck motor rather than to anything about flies.

The cap pauses TRACKING only. Activity is the primary result and is measured on the pipeline
thread, nowhere near this.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from flygym_tracker.fly_runner import FlyTrackingPool, vial_axis


def _vials(n: int = 2):
    """`n` vials with small, disjoint, non-empty masks on one 120x160 frame."""
    out = {}
    for i in range(n):
        mask = np.zeros((120, 160), dtype=bool)
        mask[20:100, 10 + i * 60: 60 + i * 60] = True
        shape = [[10 + i * 60, 20], [60 + i * 60, 20], [60 + i * 60, 100], [10 + i * 60, 100]]
        out[i + 1] = (mask, vial_axis(shape))
    return out


def _frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def _drain(pool, frames, timeout=10.0):
    """Submit `frames` frames, retrying when the queue is full, then let the workers finish."""
    sent = 0
    deadline = time.monotonic() + timeout
    while sent < frames and time.monotonic() < deadline:
        if pool.submit(_frame(), sent / 20.0):
            sent += 1
        else:
            time.sleep(0.002)
    time.sleep(0.5)          # let the workers drain what they accepted
    return sent


def test_a_dwell_that_never_ends_stops_accumulating(qapp_not_needed=None):
    """THE CRASH THIS PREVENTS. With no rotation there is no reset, so without a cap the trackers
    grow for as long as the run lasts."""
    pool = FlyTrackingPool(_vials(), fps=20.0, n_workers=1, max_dwell_frames=25)
    pool.start()
    try:
        _drain(pool, 200)                      # far past the cap, and never a reset_dwell
        stats = pool.stats()
        assert stats["capped_dwells"] >= 1, "the endless dwell was never capped"
        for worker in pool._workers:
            for tracker in worker._trackers.values():
                assert len(tracker.frames) <= 25, \
                    "a tracker kept accumulating past the cap (%d frames)" % len(tracker.frames)
    finally:
        pool.close()


def test_zero_restores_the_old_unbounded_behaviour(qapp_not_needed=None):
    """THE ROLLBACK, and it is a real one rather than a smaller number: 0 must mean the cap code
    never runs, so a rig that preferred the old behaviour gets exactly the old behaviour."""
    pool = FlyTrackingPool(_vials(), fps=20.0, n_workers=1, max_dwell_frames=0)
    pool.start()
    try:
        sent = _drain(pool, 60)
        stats = pool.stats()
        assert stats["capped_dwells"] == 0, "the cap engaged when it was switched off"
        seen = max((len(t.frames) for w in pool._workers for t in w._trackers.values()),
                   default=0)
        assert seen > 25, "with the cap off the trackers should have kept going (saw %d of %d)" % (
            seen, sent)
    finally:
        pool.close()


def test_a_rotation_resets_the_dwell_budget(qapp_not_needed=None):
    """The cap counts frames WITHIN a dwell, so a normally-flipping rig never approaches it."""
    pool = FlyTrackingPool(_vials(), fps=20.0, n_workers=1, max_dwell_frames=25)
    pool.start()
    try:
        for _ in range(4):
            _drain(pool, 15)                   # a short dwell, well under the cap
            pool.reset_dwell()
            time.sleep(0.05)
        assert pool.stats()["capped_dwells"] == 0, \
            "a normally flipping drum tripped the stalled-dwell cap"
    finally:
        pool.close()


def test_the_cap_is_reported_rather_than_silent(qapp_not_needed=None):
    """A tracking gap with a mechanical cause must be distinguishable from a biological one."""
    pool = FlyTrackingPool(_vials(), fps=20.0, n_workers=1, max_dwell_frames=10)
    pool.start()
    try:
        _drain(pool, 80)
        assert "capped_dwells" in pool.stats(), "nothing in the run summary says the cap engaged"
        assert pool.stats()["capped_dwells"] >= 1
    finally:
        pool.close()
