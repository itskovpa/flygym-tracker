"""Re-binning is not re-measuring, and a gap in a line must mean a gap in the data.

The arithmetic that decides what a graph CLAIMS lives here rather than in `paintEvent`, so it can
be checked without looking at a screen.
"""
from __future__ import annotations

import pytest

from flygym_tracker.gui.behaviour_series import (BIN_CHOICES, PLOTTABLE, BehaviourSeries)


def _row(elapsed, face, vial_id, value, field="median_path_length"):
    return {"run_id": "r", "iso_time": "", "elapsed_s": elapsed, "face": face,
            "vial_id": vial_id, field: value}


# =============================================================================================
# Binning
# =============================================================================================
def test_the_default_parameter_is_median_track_length():
    """The rig owner's choice, and the reason is in the data: with up to ~20 flies per vial a
    merged blob contributes one long fragment and crowding shreds the rest into short ones, so
    the mean sits wherever the merges put it."""
    assert PLOTTABLE[0][0] == "median_path_length"


def test_rows_in_one_window_become_one_point():
    series = BehaviourSeries()
    series.add([_row(t, "A", 1, 10.0) for t in (0.0, 2.0, 4.0, 6.0, 8.0)])
    points = series.series("median_path_length", "A", 0, bin_seconds=10.0)
    assert len(points) == 1
    assert points[0][1] == pytest.approx(10.0)


def test_the_point_is_the_median_not_the_mean():
    """One merged blob in a window must not drag the whole point with it."""
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, v) for v in (1.0, 2.0, 3.0, 4.0, 500.0)])
    points = series.series("median_path_length", "A", 0, bin_seconds=10.0)
    assert points[0][1] == pytest.approx(3.0), "an outlier moved the point"


def test_changing_the_bin_width_does_not_change_the_underlying_rows():
    """RE-BINNING IS NOT RE-MEASURING: behaviour.csv keeps one row per vial per dwell whatever
    the display says, so a bin width chosen for readability can never alter the data."""
    series = BehaviourSeries()
    rows = [_row(float(t), "A", 1, float(t)) for t in range(60)]
    series.add(rows)
    coarse = series.series("median_path_length", "A", 0, bin_seconds=60.0)
    fine = series.series("median_path_length", "A", 0, bin_seconds=1.0)
    assert len(coarse) == 1 and len(fine) == 60
    assert len(series) == 60, "re-binning consumed rows"
    # And asking again gives the same answer -- the store is not mutated by reading it.
    assert series.series("median_path_length", "A", 0, bin_seconds=60.0) == coarse


@pytest.mark.parametrize("seconds", BIN_CHOICES)
def test_every_offered_bin_width_works(seconds):
    series = BehaviourSeries()
    series.add([_row(float(t), "A", 1, 5.0) for t in range(0, 120, 2)])
    assert series.series("median_path_length", "A", 0, bin_seconds=seconds)


# =============================================================================================
# nan is not zero
# =============================================================================================
def test_a_nan_row_is_dropped_rather_than_counted_as_zero():
    """An empty vial has no track length, which is not the same claim as "its flies did not move".
    Counting nan as 0 would draw a confident line along the floor."""
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, float("nan")), _row(1.0, "A", 1, 8.0)])
    points = series.series("median_path_length", "A", 0, bin_seconds=10.0)
    assert len(points) == 1
    assert points[0][1] == pytest.approx(8.0)


def test_a_window_with_nothing_measurable_produces_no_point():
    """So a GAP in the line reads as a gap in the data, rather than a dip toward zero."""
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, 5.0), _row(100.0, "A", 1, 5.0)])
    points = series.series("median_path_length", "A", 0, bin_seconds=10.0)
    assert len(points) == 2, "the empty windows between were filled in"


# =============================================================================================
# Cumulative
# =============================================================================================
def test_cumulative_is_a_running_sum_of_the_binned_values():
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, 1.0), _row(10.0, "A", 1, 2.0), _row(20.0, "A", 1, 3.0)])
    plain = series.series("median_path_length", "A", 0, bin_seconds=10.0)
    stacked = series.series("median_path_length", "A", 0, bin_seconds=10.0, cumulative=True)
    assert [v for _t, v in plain] == pytest.approx([1.0, 2.0, 3.0])
    assert [v for _t, v in stacked] == pytest.approx([1.0, 3.0, 6.0])


def test_cumulative_never_decreases_for_a_non_negative_parameter():
    series = BehaviourSeries()
    series.add([_row(float(10 * i), "A", 1, float(i % 4)) for i in range(20)])
    values = [v for _t, v in series.series("median_path_length", "A", 0, bin_seconds=10.0,
                                           cumulative=True)]
    assert all(b >= a for a, b in zip(values, values[1:]))


# =============================================================================================
# Vials and faces
# =============================================================================================
def test_a_vial_only_shows_its_own_rows():
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, 10.0), _row(0.0, "A", 2, 99.0)])
    assert series.series("median_path_length", "A", 0, bin_seconds=10.0)[0][1] == 10.0
    assert series.series("median_path_length", "A", 1, bin_seconds=10.0)[0][1] == 99.0


def test_face_b_global_ids_map_onto_its_own_sixteen_cells():
    """Global ids are face_index*16 + local, so face B's vial 1 is global 17. Getting this wrong
    would draw face B's data in face A's grid, or in no grid at all."""
    series = BehaviourSeries()
    series.add([_row(0.0, "B", 17, 42.0), _row(0.0, "B", 32, 7.0)])
    assert series.series("median_path_length", "B", 0, bin_seconds=10.0)[0][1] == 42.0
    assert series.series("median_path_length", "B", 15, bin_seconds=10.0)[0][1] == 7.0


def test_a_face_series_covers_all_sixteen_cells():
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 3, 5.0)])
    face = series.face_series("median_path_length", "A", bin_seconds=10.0)
    assert len(face) == 16
    assert face[2] and not face[0], "the wrong cell was populated"


# =============================================================================================
# One shared y range
# =============================================================================================
def test_the_value_range_spans_both_faces():
    """ONE RANGE FOR ALL 32 CELLS: per-cell autoscaling would draw an empty vial's noise at the
    same amplitude as a busy vial's real signal, and the grid exists so vials can be compared."""
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, 5.0), _row(0.0, "B", 17, 50.0)])
    low, high = series.value_range("median_path_length", bin_seconds=10.0)
    assert low == pytest.approx(5.0)
    assert high == pytest.approx(50.0)


def test_a_flat_series_still_gets_a_drawable_band():
    """Otherwise every point lands exactly on one edge of its cell."""
    series = BehaviourSeries()
    series.add([_row(float(10 * i), "A", 1, 7.0) for i in range(5)])
    low, high = series.value_range("median_path_length", bin_seconds=10.0)
    assert high > low


def test_no_data_means_no_range_rather_than_a_made_up_one():
    assert BehaviourSeries().value_range("median_path_length") is None
    assert BehaviourSeries().time_range() is None


# =============================================================================================
# Housekeeping
# =============================================================================================
def test_a_long_run_drops_the_oldest_rows_and_says_how_many():
    """A 3-day run at ~2 s dwells over 32 vials is millions of rows. The FILE keeps them all;
    this store is for watching."""
    series = BehaviourSeries(max_rows=100)
    series.add([_row(float(i), "A", 1, 1.0) for i in range(250)])
    assert len(series) == 100
    assert series.dropped_rows == 150


def test_malformed_rows_are_skipped_rather_than_raising():
    """A plot must never be able to take down a running experiment."""
    series = BehaviourSeries()
    added = series.add([{"nonsense": 1}, {"elapsed_s": "x", "face": "A", "vial_id": 1},
                        _row(0.0, "A", 1, 3.0)])
    assert added == 1
    assert len(series) == 1


def test_clearing_starts_a_new_run_from_nothing():
    series = BehaviourSeries()
    series.add([_row(0.0, "A", 1, 5.0)])
    series.clear()
    assert len(series) == 0
    assert series.series("median_path_length", "A", 0) == []


# =============================================================================================
# The y axis: labelled, and shared by default
# =============================================================================================
def test_the_axis_tick_stays_short_enough_to_fit_a_cell():
    """A 30 px gutter cannot hold "1234567.891". An axis number nobody can read is the same as
    no axis number, which is what the operator asked to stop having."""
    from flygym_tracker.gui.plot_dock import _tick

    for value, expected in ((0.5, "0.50"), (7.25, "7.2"), (412.0, "412"),
                            (5400.0, "5k"), (2_500_000.0, "2.5M")):
        assert _tick(value) == expected, "%r formatted as %r" % (value, _tick(value))
    assert all(len(_tick(v)) <= 6 for v in (0.001, 1.0, 99.9, 1e4, 1e7))


def test_per_cell_scaling_uses_only_that_cells_points():
    from flygym_tracker.gui.plot_dock import _range_of_points

    assert _range_of_points([(0.0, 3.0), (1.0, 9.0)]) == (3.0, 9.0)
    assert _range_of_points([]) is None
    low, high = _range_of_points([(0.0, 4.0), (1.0, 4.0)])
    assert high > low, "a flat cell got no drawable band"
