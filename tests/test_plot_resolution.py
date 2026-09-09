from flygym_tracker.gui.behaviour_series import BehaviourSeries
from flygym_tracker.gui.plot_dock import BehaviourPlotPanel


def test_display_explains_recorded_resolution_and_clears_between_runs(qapp):
    store = BehaviourSeries()
    store.add([dict(elapsed_s=0, face='A', vial_id=1, motion_px_sum=100,
                    bin_start_iso='2026-07-22T16:00:00', bin_end_iso='2026-07-22T16:00:10')])
    panel = BehaviourPlotPanel(store, 'motion_px_sum')
    panel.bin_box.setCurrentIndex(panel.bin_box.findData(.2))
    assert 'Recorded activity bins: 10 s' in panel.resolution_label.text()
    assert 'cannot recover detail' in panel.resolution_label.text()
    panel.bin_box.setCurrentIndex(panel.bin_box.findData(10))
    assert 'cannot recover detail' not in panel.resolution_label.text()
    store.clear()
    panel.refresh()
    assert panel.resolution_label.text() == ''


def test_subsecond_recording_supports_matching_display_bins(qapp):
    store = BehaviourSeries()
    store.add([dict(elapsed_s=0, face='A', vial_id=1, active_fraction_mean=.1,
                    bin_start_iso='2026-07-22T16:00:00', bin_end_iso='2026-07-22T16:00:00.200')])
    panel = BehaviourPlotPanel(store, 'active_fraction_mean')
    panel.bin_box.setCurrentIndex(panel.bin_box.findData(.2))
    assert '200 ms' in panel.resolution_label.text()
    assert 'cannot recover detail' not in panel.resolution_label.text()
