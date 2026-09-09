"""Scientific mapping and GUI integration for the activity heatmap."""
import time

import pytest
from flygym_tracker.gui.activity_heatmap import (ActivityHeatmapDock, ActivityHeatmapPanel,
                                                 _heat_colour)
from flygym_tracker.gui.behaviour_series import (HEATMAP_MEASURED, HEATMAP_MISSING,
                                                 HEATMAP_OTHER_FACE, BehaviourSeries)


def _row(t, face, vial, value, field="active_fraction_mean", width=10.0):
    row = {"elapsed_s": t, "face": face, "vial_id": vial, field: value}
    if width is not None:
        row.update(bin_start_iso="2026-01-01T00:00:%06.3f" % t,
                   bin_end_iso="2026-01-01T00:00:%06.3f" % (t + width))
    return row


def test_global_vial_ids_map_to_local_heatmap_rows_on_both_faces():
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, .1), _row(0, "A", 16, .2),
               _row(0, "B", 17, .3), _row(0, "B", 32, .4)])
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=10)
    assert snap.value("A", 0, 0) == pytest.approx(.1)
    assert snap.value("A", 15, 0) == pytest.approx(.2)
    assert snap.value("B", 0, 0) == pytest.approx(.3)
    assert snap.value("B", 15, 0) == pytest.approx(.4)


def test_missing_other_face_and_measured_zero_are_three_distinct_states():
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, 0.0),            # measured zero on face A
               _row(10, "A", 1, None),          # A absent while B is measured
               _row(10, "B", 17, .2),
               _row(20, "A", 1, None)])         # neither face measured
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=10)
    assert snap.state("A", 0, 0) == HEATMAP_MEASURED
    assert snap.value("A", 0, 0) == 0.0
    assert snap.state("A", 0, 1) == HEATMAP_OTHER_FACE
    assert snap.state("A", 0, 2) == HEATMAP_MISSING
    assert ("A", 0, 2) not in snap.values


def test_decimal_boundaries_make_separate_heatmap_columns():
    store = BehaviourSeries()
    store.add([_row(i / 5, "A", 1, i, width=.2) for i in range(10)])
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=.2)
    assert snap.buckets == tuple(range(10))
    assert [snap.value("A", 0, i) for i in range(10)] == list(range(10))
    assert snap.time_range == pytest.approx((0.0, 2.0))


def test_shared_colour_range_spans_faces_and_recent_cap_bounds_time():
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, 0), _row(10, "B", 17, 2),
               _row(20, "A", 1, 4), _row(30, "B", 17, 8)])
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=10, max_columns=2)
    assert snap.buckets == (2, 3)
    assert snap.time_range == pytest.approx((20, 40))
    assert snap.value_range == pytest.approx((0, 8))
    assert _heat_colour(0, snap.value_range) != _heat_colour(8, snap.value_range)


def test_heatmap_medians_match_existing_display_binning():
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, 1), _row(1, "A", 1, 100), _row(2, "A", 1, 3)])
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=10)
    assert snap.value("A", 0, 0) == 3
    assert store.series("active_fraction_mean", "A", 0, bin_seconds=10)[0][1] == 3


def test_panel_defaults_to_active_fraction_and_never_splits_recorded_bins(qapp):
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, .1, width=10)])
    panel = ActivityHeatmapPanel(store)
    assert panel.field() == "active_fraction_mean"
    panel.bin_box.setCurrentIndex(panel.bin_box.findData(.2))
    assert panel.effective_bin_seconds() == 10
    assert panel.heatmap.snapshot.bin_seconds == 10
    assert "cannot recover finer measurements" in panel.resolution_label.text()


def test_switching_metrics_updates_data_and_area_control(qapp):
    store = BehaviourSeries()
    row = _row(0, "A", 1, .25)
    row.update(motion_px_sum=125, lit_area_px=1000)
    store.add([row])
    panel = ActivityHeatmapPanel(store)
    assert panel.heatmap.snapshot.value("A", 0, 0) == pytest.approx(.25)
    panel.metric_box.setCurrentIndex(panel.metric_box.findData("motion_px_sum"))
    assert panel.area_box.isEnabled() and panel.area_box.isChecked()
    assert panel.heatmap.snapshot.field == "motion_px_sum"
    assert panel.heatmap.snapshot.value("A", 0, 0) == pytest.approx(125)


def test_heatmap_dock_scrolls_and_renders_at_small_and_normal_sizes(qapp):
    store = BehaviourSeries()
    store.add([_row(0, "A", 1, 0), _row(10, "B", 17, .4)])
    dock = ActivityHeatmapDock(store)
    dock.resize(320, 260)
    dock.show()
    qapp.processEvents()
    scroll = dock.widget()
    assert scroll.verticalScrollBar().maximum() > 0
    image = dock.grab().toImage()
    assert image.width() == 320 and image.height() == 260
    assert image.pixelColor(10, 10).isValid()
    dock.resize(620, 620)
    qapp.processEvents()
    assert dock.panel.heatmap.width() >= 360


def test_snapshot_refresh_cost_is_measured_on_bounded_synthetic_history():
    """A generous guard plus a measurement: 500 columns x 32 rows is the intended live window."""
    store = BehaviourSeries(max_rows=20_000)
    rows = []
    for bucket in range(500):
        face, offset = (("A", 0) if bucket % 2 == 0 else ("B", 16))
        rows.extend(_row(bucket * 10, face, offset + vial + 1, (bucket + vial) % 20 / 20,
                         width=None)
                    for vial in range(16))
    store.add(rows)
    started = time.perf_counter()
    snap = store.activity_heatmap("active_fraction_mean", bin_seconds=10, max_columns=500)
    elapsed = time.perf_counter() - started
    assert len(snap.values) == 8_000
    assert elapsed < 1.0, "bounded synthetic snapshot took %.3f s" % elapsed
