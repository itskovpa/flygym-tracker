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


def test_an_unobserved_activity_bin_gaps_but_an_observed_zero_stays():
    """THE FLIP-CYCLE DILUTION FIX. `pipeline._bin_to_records` now emits active_fraction_mean=None
    for a bin whose vial had NO stationary frames -- the OTHER drum face was up, so it was not
    observed. That must be a GAP, or each face's curve dips to the floor every time the other face
    is showing (the periodic oscillation the operator reported). But a bin that WAS observed and
    whose flies simply did not move is a real 0 and must stay, or a genuinely quiet vial vanishes.
    So None gaps; 0.0 is a point on the floor."""
    series = BehaviourSeries()
    series.add([
        _row(0.0, "A", 1, None, field="active_fraction_mean"),     # other face up -> unobserved
        _row(10.0, "A", 1, 0.0, field="active_fraction_mean"),     # observed, flies still -> real 0
        _row(20.0, "A", 1, 0.05, field="active_fraction_mean"),
    ])
    points = series.series("active_fraction_mean", "A", 0, bin_seconds=10.0)
    assert len(points) == 2, "the unobserved (None) bin was not gapped, or the observed 0 was dropped"
    assert points[0][1] == pytest.approx(0.0), "an observed zero must stay a point on the floor"
    assert points[1][1] == pytest.approx(0.05)


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


# =============================================================================================
# A gap in TIME is a gap in the LINE (the drum-face artifact)
# ---------------------------------------------------------------------------------------------
# The store already drops empty bins (above), so a stretch with no measurement produces no point.
# But the painter joined the surviving points with ONE polyline, bridging that stretch with a
# straight ramp -- activity the flies never produced, and the misleading graph reported from the
# rig. `_contiguous_segments` splits the run across those gaps so each filmed stretch is its own
# line and the unfilmed stretch stays blank.
# =============================================================================================
def _seg_lengths(segments):
    return [len(s) for s in segments]


def test_a_run_with_no_gap_stays_one_line():
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    # Bin 0.5 s, three adjacent bin centres (0.5 s apart): nothing to break.
    pts = [(0.25, 1.0), (0.75, 1.0), (1.25, 1.0)]
    assert _seg_lengths(_contiguous_segments(pts, 0.5)) == [3]


def test_an_empty_bin_between_two_points_breaks_the_line():
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    # Bin 0.5 s: a filmed stretch (4 centres 0.5 s apart), then the drum shows the other face for
    # several bins (no points), then filmed again. The jump across the unfilmed bins must split.
    filmed_a = [(0.25, 0.02), (0.75, 0.03), (1.25, 0.04), (1.75, 0.03)]
    filmed_b = [(6.25, 0.05), (6.75, 0.04), (7.25, 0.03)]
    segments = _contiguous_segments(filmed_a + filmed_b, 0.5)
    assert _seg_lengths(segments) == [4, 3], "the unfilmed stretch was bridged"
    assert segments[0][-1][0] == 1.75 and segments[1][0][0] == 6.25


def test_the_alternating_drum_cadence_becomes_many_short_lines():
    """The real rig scenario: this face is filmed, then not, then filmed, repeatedly. Each filmed
    stretch is its own segment; none is joined across the stretch when the other face was up."""
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    pts = []
    for cycle in range(4):
        base = cycle * 4.0                       # 2 s filmed, then ~2 s on the other face
        pts += [(base + 0.25, 0.03), (base + 0.75, 0.03), (base + 1.25, 0.03)]
    segments = _contiguous_segments(pts, 0.5)
    assert _seg_lengths(segments) == [3, 3, 3, 3], "filmed stretches were joined across the gaps"


def test_a_coarse_bin_that_samples_every_window_stays_continuous():
    """REGRESSION: at 10 s every bin still contains some frames of this face (the drum flips every
    ~2 s), so consecutive points are one bin apart and the line must NOT be chopped up. Breaking
    here would turn the production view into confetti."""
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    pts = [(5.0, 0.02), (15.0, 0.03), (25.0, 0.02), (35.0, 0.04)]
    assert _seg_lengths(_contiguous_segments(pts, 10.0)) == [4]


def test_a_lone_point_between_two_gaps_is_its_own_segment():
    """Drawn as a dot, not silently joined to a neighbour across a gap."""
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    pts = [(0.25, 0.02), (5.25, 0.03), (10.25, 0.02)]      # bin 0.5 s: 5 s gaps between each
    assert _seg_lengths(_contiguous_segments(pts, 0.5)) == [1, 1, 1]


def test_raw_view_estimates_the_cadence_and_breaks_the_big_gaps():
    """The raw view (bin 0) has no fixed bin, so the normal spacing is read from the data. A filmed
    burst at the recording cadence, an unfilmed gap, then another burst -> two segments."""
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    burst_a = [(0.0, 0.02), (0.2, 0.03), (0.4, 0.02), (0.6, 0.03)]
    burst_b = [(3.0, 0.04), (3.2, 0.03), (3.4, 0.02)]
    assert _seg_lengths(_contiguous_segments(burst_a + burst_b, 0.0)) == [4, 3]


def test_segmenting_is_safe_on_zero_or_one_point():
    from flygym_tracker.gui.plot_dock import _contiguous_segments
    assert _contiguous_segments([], 0.5) == []
    assert _contiguous_segments(None, 0.5) == []
    assert _contiguous_segments([(0.0, 1.0)], 0.5) == [[(0.0, 1.0)]]


# =============================================================================================
# Per-ROI-area normalization
# ---------------------------------------------------------------------------------------------
# ROIs are hand-drawn and vary in size. A metric that scales with area (a pixel sum, a fly count)
# reads bigger for a bigger box at the same real activity. `normalize_area=True` rescales each vial
# by median_ROI/its_own_ROI so vials compare -- but ONLY for the extensive fields, and never for a
# speed, height or per-fly figure, which do not depend on how big the box was drawn.
# =============================================================================================
def _area_row(elapsed, face, vial_id, area, **fields):
    row = {"run_id": "r", "iso_time": "", "elapsed_s": elapsed, "face": face, "vial_id": vial_id}
    if area is not None:
        row["lit_area_px"] = area
    row.update(fields)
    return row


def test_only_the_extensive_fields_are_area_normalized():
    """active_fraction is ALREADY motion/area, so it must not be divided again; speeds, heights and
    per-fly path length are area-independent and must be left alone."""
    from flygym_tracker.gui.behaviour_series import AREA_NORMALIZED_FIELDS

    assert "motion_px_sum" in AREA_NORMALIZED_FIELDS
    assert "active_fraction_mean" not in AREA_NORMALIZED_FIELDS, "would double-divide by area"
    for intensive in ("median_speed", "mean_speed", "p90_speed", "mean_height", "median_height",
                      "frac_above_mid", "median_path_length", "mean_fragment_frames"):
        assert intensive not in AREA_NORMALIZED_FIELDS, "%s is not extensive" % intensive


def test_area_normalization_scales_an_extensive_metric_to_the_median_roi():
    series = BehaviourSeries()
    # vial 1 lit area 1000, vial 2 lit area 2000 -> median 1500; factors 1.5 and 0.75.
    series.add([_area_row(0.0, "A", 1, 1000, motion_px_sum=100),
                _area_row(0.0, "A", 2, 2000, motion_px_sum=100)])
    v1 = series.series("motion_px_sum", "A", 0, bin_seconds=10.0, normalize_area=True)
    v2 = series.series("motion_px_sum", "A", 1, bin_seconds=10.0, normalize_area=True)
    assert v1[0][1] == pytest.approx(150.0), "100 * 1500/1000"
    assert v2[0][1] == pytest.approx(75.0), "100 * 1500/2000"
    # OFF unless asked: the recorded value comes back untouched.
    assert series.series("motion_px_sum", "A", 0, bin_seconds=10.0)[0][1] == pytest.approx(100.0)


def test_area_normalization_never_touches_an_intensive_metric():
    """A bigger ROI does not make a fly faster, so median_speed must be identical with the flag on
    or off even though this vial's area is known."""
    series = BehaviourSeries()
    series.add([_area_row(0.0, "A", 1, 1000, median_speed=4.0),
                _area_row(0.0, "A", 2, 5000, median_speed=4.0)])
    on = series.series("median_speed", "A", 0, bin_seconds=10.0, normalize_area=True)
    off = series.series("median_speed", "A", 0, bin_seconds=10.0)
    assert on == off, "an intensive metric was rescaled by area"
    assert on[0][1] == pytest.approx(4.0)


def test_area_for_a_tracking_metric_comes_from_the_vials_activity_row():
    """behaviour.csv rows carry no lit_area of their own; the constant per-vial area is harvested
    from the activity rows and applied to the SAME vial's tracking metric."""
    series = BehaviourSeries()
    series.add([_area_row(0.0, "A", 1, 1000, motion_px_sum=1),      # activity rows set the areas
                _area_row(0.0, "A", 2, 3000, motion_px_sum=1)])
    series.add([_area_row(1.0, "A", 1, None, total_path_length=200.0),   # tracking rows: no area
                _area_row(1.0, "A", 2, None, total_path_length=200.0)])
    # median area = 2000; vial 1 factor 2.0, vial 2 factor 2000/3000.
    v1 = series.series("total_path_length", "A", 0, bin_seconds=10.0, normalize_area=True)
    v2 = series.series("total_path_length", "A", 1, bin_seconds=10.0, normalize_area=True)
    assert v1[0][1] == pytest.approx(200.0 * 2000 / 1000)
    assert v2[0][1] == pytest.approx(200.0 * 2000 / 3000)


def test_area_normalization_is_a_no_op_when_the_area_is_unknown():
    """No area harvested yet -> factor 1, value returned exactly as recorded rather than dropped."""
    series = BehaviourSeries()
    series.add([_area_row(0.0, "A", 1, None, motion_px_sum=100)])
    assert series.area_reference() is None
    v1 = series.series("motion_px_sum", "A", 0, bin_seconds=10.0, normalize_area=True)
    assert v1[0][1] == pytest.approx(100.0)


def test_the_area_reference_is_the_median_not_the_mean():
    """One enormous hand-drawn ROI must not drag the reference the other vials are scaled to."""
    series = BehaviourSeries()
    for vid, area in ((1, 100), (2, 100), (3, 100), (4, 10_000)):
        series.add([_area_row(0.0, "A", vid, area, motion_px_sum=1)])
    assert series.area_reference() == pytest.approx(100.0)


def test_area_normalization_applies_before_the_cumulative_sum():
    series = BehaviourSeries()
    # vial 1 area 1000, vial 2 area 2000 -> median 1500 -> vial 2 factor 0.75.
    series.add([_area_row(0.0, "A", 1, 1000, motion_px_sum=0),
                _area_row(0.0, "A", 2, 2000, motion_px_sum=40),
                _area_row(10.0, "A", 2, 2000, motion_px_sum=40)])
    pts = series.series("motion_px_sum", "A", 1, bin_seconds=10.0, cumulative=True,
                        normalize_area=True)
    assert [round(v, 6) for _t, v in pts] == [30.0, 60.0], "each 40*0.75=30, cumulative 30 then 60"


def test_clearing_forgets_the_roi_areas():
    series = BehaviourSeries()
    series.add([_area_row(0.0, "A", 1, 500, motion_px_sum=1)])
    series.clear()
    assert series.area_reference() is None
