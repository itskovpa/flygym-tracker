import pytest

from flygym_tracker.gui.behaviour_series import BehaviourSeries
from flygym_tracker.gui.plot_dock import BehaviourPlotPanel


@pytest.mark.parametrize("times,width", [([10], 10), ([9, 20], 10), ([1, 12], 300)])
def test_all_samples_axis_contains_every_bin_center(qapp, times, width):
    store = BehaviourSeries()
    store.add([dict(elapsed_s=t, face="A", vial_id=1, motion_px_sum=20 + t)
               for t in times])
    panel = BehaviourPlotPanel(store, "motion_px_sum")
    panel.bin_box.setCurrentIndex(panel.bin_box.findData(width))
    panel.refresh()
    grid = panel.grids["A"]
    low, high = grid._time
    assert high > low
    assert all(low <= t <= high for t, _ in grid._points[0])
    assert low <= min(times) and max(times) <= high
