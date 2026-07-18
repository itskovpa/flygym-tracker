"""Tests for flygym_tracker.activity (DESIGN.md §5.3)."""
import numpy as np
import pytest

from flygym_tracker.activity import ActivityAccumulator, per_frame_activity
from flygym_tracker.types import TrackState


# ---- per_frame_activity (pure function) -------------------------------------


def test_per_frame_activity_exact_count_and_masking():
    h, w = 10, 10
    vial_mask = np.zeros((h, w), dtype=bool)
    vial_mask[2:5, 2:5] = True  # 3x3 = 9 px vial area
    lit_area = int(vial_mask.sum())
    assert lit_area == 9

    prev = np.full((h, w), 50, dtype=np.uint8)
    cur = prev.copy()
    pixel_threshold = 10

    # Exactly N=4 pixels INSIDE the mask change by > threshold (delta=20).
    inside_over = [(2, 2), (2, 3), (3, 2), (4, 4)]
    for (r, c) in inside_over:
        cur[r, c] = prev[r, c] + 20
    N = len(inside_over)

    # One pixel inside the mask changes but NOT above threshold -> must not count.
    cur[3, 3] = prev[3, 3] + 5  # delta=5 < 10

    # Several pixels OUTSIDE the mask change by a lot -> must be ignored entirely.
    outside_over = [(0, 0), (1, 1), (7, 7), (8, 8), (9, 9)]
    for (r, c) in outside_over:
        cur[r, c] = prev[r, c] + 100

    motion_px, lit_area_px, active_fraction = per_frame_activity(cur, prev, vial_mask, pixel_threshold)

    assert motion_px == N
    assert lit_area_px == lit_area
    assert active_fraction == pytest.approx(N / lit_area)


def test_per_frame_activity_threshold_is_strict_greater_than():
    h, w = 4, 4
    vial_mask = np.ones((h, w), dtype=bool)
    prev = np.full((h, w), 100, dtype=np.uint8)
    cur = prev.copy()
    cur[0, 0] = 110  # delta exactly == threshold -> must NOT count ('>' not '>=')
    cur[1, 1] = 111  # delta > threshold -> must count

    motion_px, lit_area_px, active_fraction = per_frame_activity(cur, prev, vial_mask, pixel_threshold=10)
    assert motion_px == 1
    assert lit_area_px == 16
    assert active_fraction == pytest.approx(1 / 16)


def test_per_frame_activity_zero_lit_area_guards_division():
    h, w = 5, 5
    vial_mask = np.zeros((h, w), dtype=bool)  # nothing lit / vial absent
    prev = np.zeros((h, w), dtype=np.uint8)
    cur = np.full((h, w), 255, dtype=np.uint8)

    motion_px, lit_area_px, active_fraction = per_frame_activity(cur, prev, vial_mask, pixel_threshold=10)
    assert lit_area_px == 0
    assert motion_px == 0
    assert active_fraction == 0.0


def test_per_frame_activity_no_uint8_wraparound():
    # prev has high values, cur has low values -> naive uint8 subtraction would wrap around
    # instead of producing a large difference.
    h, w = 3, 3
    vial_mask = np.ones((h, w), dtype=bool)
    prev = np.full((h, w), 250, dtype=np.uint8)
    cur = np.full((h, w), 5, dtype=np.uint8)  # true abs diff = 245

    motion_px, lit_area_px, active_fraction = per_frame_activity(cur, prev, vial_mask, pixel_threshold=200)
    assert motion_px == 9  # all 9 pixels exceed threshold once wraparound is avoided
    assert active_fraction == pytest.approx(1.0)


# ---- ActivityAccumulator -----------------------------------------------------


def test_activity_accumulator_bin_rollover_and_aggregation():
    bin_seconds = 10.0
    acc = ActivityAccumulator(bin_seconds=bin_seconds)

    vial_id = 1
    lit_area = 100

    # Bin 0 ([0, 10)s): 5 STATIONARY frames with known motion_px values.
    motion_values = [5, 10, 15, 20, 25]
    elapsed_times = [0.0, 2.0, 4.0, 6.0, 8.0]

    for t, mv in zip(elapsed_times, motion_values):
        af = mv / lit_area
        result = acc.add(t, TrackState.STATIONARY, {vial_id: (mv, lit_area, af)})
        assert result is None  # still inside bin 0, no rollover yet

    # 2 ROTATING frames, still inside bin 0. Their motion/active_fraction values are bogus on
    # purpose (999) to prove the accumulator ignores them for STATIONARY-only fields.
    assert acc.add(9.0, TrackState.ROTATING, {vial_id: (999, lit_area, 0.9)}) is None
    assert acc.add(9.5, TrackState.ROTATING, {vial_id: (999, lit_area, 0.9)}) is None

    # A SETTLING frame inside bin 0 -- must not affect any count.
    assert acc.add(9.8, TrackState.SETTLING, {vial_id: (999, lit_area, 0.9)}) is None

    # Cross into bin 1 ([10, 20)s) -> triggers rollover, returns bin 0's completed data.
    rollover = acc.add(10.5, TrackState.STATIONARY, {vial_id: (1, lit_area, 1 / lit_area)})
    assert rollover is not None
    assert rollover.bin_start_s == 0.0
    assert rollover.bin_end_s == 10.0

    v = rollover.vials[vial_id]
    assert v["motion_px_sum"] == sum(motion_values)  # 75, rotating/settling excluded
    expected_mean_af = sum(mv / lit_area for mv in motion_values) / len(motion_values)
    assert v["active_fraction_mean"] == pytest.approx(expected_mean_af)
    assert v["n_stationary_frames"] == len(motion_values)  # 5
    assert v["n_rotating_frames"] == 2  # settling frame not counted as rotating either
    assert v["lit_area_px"] == lit_area

    # Force-flush the now-partial bin 1 (just the one STATIONARY sample added at t=10.5).
    final = acc.flush()
    assert final is not None
    assert final.bin_start_s == 10.0
    assert final.bin_end_s == 20.0
    v2 = final.vials[vial_id]
    assert v2["motion_px_sum"] == 1
    assert v2["n_stationary_frames"] == 1
    assert v2["n_rotating_frames"] == 0
    assert v2["active_fraction_mean"] == pytest.approx(1 / lit_area)

    # Flushing again with nothing pending returns None.
    assert acc.flush() is None


def test_activity_accumulator_multi_vial_independent_aggregation():
    acc = ActivityAccumulator(bin_seconds=5.0)
    v1, v2 = 1, 2

    acc.add(0.0, TrackState.STATIONARY, {v1: (10, 50, 0.2), v2: (30, 60, 0.5)})
    acc.add(1.0, TrackState.STATIONARY, {v1: (20, 50, 0.4), v2: (0, 60, 0.0)})
    # vial 2 momentarily absent this frame (e.g. slot fell out of the lit region) -> only v1 updates
    acc.add(2.0, TrackState.STATIONARY, {v1: (0, 50, 0.0)})

    bin0 = acc.flush()
    assert bin0.vials[v1]["motion_px_sum"] == 30
    assert bin0.vials[v1]["n_stationary_frames"] == 3
    assert bin0.vials[v1]["active_fraction_mean"] == pytest.approx((0.2 + 0.4 + 0.0) / 3)

    assert bin0.vials[v2]["motion_px_sum"] == 30
    assert bin0.vials[v2]["n_stationary_frames"] == 2  # only present in 2 of the 3 frames
    assert bin0.vials[v2]["active_fraction_mean"] == pytest.approx((0.5 + 0.0) / 2)


def test_activity_accumulator_settling_and_unknown_are_excluded():
    acc = ActivityAccumulator(bin_seconds=5.0)
    vial_id = 1

    acc.add(0.0, TrackState.UNKNOWN, {vial_id: (999, 50, 0.9)})
    acc.add(0.5, TrackState.SETTLING, {vial_id: (999, 50, 0.9)})
    result = acc.flush()

    # Nothing STATIONARY/ROTATING was ever added, so the vial never appears in the accumulator.
    assert result.vials == {}


def test_activity_accumulator_rejects_nonpositive_bin_seconds():
    with pytest.raises(ValueError):
        ActivityAccumulator(bin_seconds=0)
    with pytest.raises(ValueError):
        ActivityAccumulator(bin_seconds=-5)
