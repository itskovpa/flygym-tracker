import numpy as np
from flygym_tracker.activity import per_frame_activity
from flygym_tracker.spatial_activity import SpatialActivity
from flygym_tracker.gui.activity_heatmap import ActivityHeatmapPanel, overlay_image


def test_pixel_output_preserves_positions_threshold_and_mask_and_unions_overlaps():
    previous = np.zeros((4, 5), np.uint8)
    current = previous.copy()
    current[1, 2] = 20
    current[3, 4] = 21
    current[0, 0] = 255
    mask = np.ones(previous.shape, bool)
    mask[0, 0] = False
    motion = np.zeros_like(mask)
    result = per_frame_activity(current, previous, mask, 20, motion_out=motion)
    assert result[0] == 1
    assert np.argwhere(motion).tolist() == [[3, 4]]
    per_frame_activity(current, previous, mask, 20, motion_out=motion)
    assert motion.sum() == 1


def test_counts_accumulate_by_face_background_stays_static_and_snapshots_are_owned():
    store = SpatialActivity()
    gray = np.full((4, 5), 80, np.uint8)
    motion = np.zeros(gray.shape, bool)
    motion[1, 2] = True
    store.add('A', gray, motion)
    gray[:] = 100
    store.add('A', gray, motion)
    store.add('B', gray, np.zeros_like(motion))
    assert store.snapshot(9.99) is None
    first = store.snapshot(10)
    assert first['faces']['A']['counts'][1, 2] == 2
    assert first['faces']['B']['counts'].sum() == 0
    assert np.all(first['faces']['A']['background'] == 80)
    store.add('A', gray, motion)
    assert first['faces']['A']['counts'][1, 2] == 2
    assert store.snapshot(19.99) is None
    assert store.snapshot(20)['maximum'] == 3
    assert store.snapshot(21, force=True)['maximum'] == 3


def test_overlay_leaves_inactive_pixels_exactly_at_background():
    gray = np.full((4, 5), 80, np.uint8)
    counts = np.zeros(gray.shape, np.uint64)
    counts[1, 2] = 10
    rgb = overlay_image(gray, counts, 10)
    assert (rgb[0, 0] == [80, 80, 80]).all()
    assert not (rgb[1, 2] == [80, 80, 80]).all()
    assert np.count_nonzero(np.any(rgb != 80, axis=2)) == 1


def test_face_selection_refresh_and_new_run_clear(qapp):
    store = SpatialActivity()
    gray = np.full((40, 50), 80, np.uint8)
    store.add('A', gray, np.ones_like(gray, bool))
    payload = store.snapshot(10)
    panel = ActivityHeatmapPanel(payload)
    panel.resize(500, 500)
    panel.show()
    qapp.processEvents()
    assert panel.heatmap.image.width() == 50
    assert not panel.grab().isNull()
    panel.face_box.setCurrentIndex(1)
    assert panel.heatmap.image is None
    panel.face_box.setCurrentIndex(0)
    assert panel.heatmap.image is not None
    payload.clear()
    panel.refresh()
    assert panel.heatmap.image is None


def test_pipeline_spatial_counts_match_activity_and_exclude_rotation(tmp_path):
    from test_pipeline import _config, _calibration, _full_scene, FakeSource
    from flygym_tracker.pipeline import TrackerPipeline
    from flygym_tracker.logger import ActivityLogger
    pipe = TrackerPipeline(_config(), _calibration(tmp_path), FakeSource(_full_scene()),
                           ActivityLogger(tmp_path/'out', 'spatial', fmt='csv'), clock='index')
    pipe.spatial_activity = SpatialActivity()
    totals = []
    pipe.add_observer(lambda p: totals.append(sum(v[0] for v in (p['vial_results'] or {}).values())))
    summary = pipe.run()
    assert summary['n_rotations'] > 0
    assert summary['observer_failures'] == 0
    counts = sum(int(d['counts'].sum()) for d in pipe.spatial_activity.faces.values())
    assert counts > 0
    assert counts == sum(totals)

def test_live_mode_renders_exact_frame_mask_and_keeps_cumulative_snapshot(qapp):
    gray = np.full((4, 5), 80, np.uint8)
    motion = np.zeros(gray.shape, bool)
    motion[1, 2] = True
    store = SpatialActivity()
    store.add('A', gray, motion)
    payload = store.snapshot(10)
    payload['live'] = dict(frame=gray, motion=motion, threshold=15, face='A', elapsed_s=11)
    panel = ActivityHeatmapPanel(payload)
    changes = []
    panel.threshold_requested.connect(changes.append)
    panel.mode_box.setCurrentIndex(1)
    assert '1 pixels detected' in panel.range_label.text()
    assert panel.heatmap.image.pixelColor(0, 0).red() == 80
    assert panel.heatmap.image.pixelColor(2, 1).red() > 80
    panel.threshold_box.setValue(25)
    panel.apply_button.click()
    assert changes == [25]
    # Editing the detector does not relabel the threshold used for this older frame.
    assert 'actual threshold 15' in panel.range_label.text()
    payload['live'] = dict(frame=gray, motion=None, threshold=25, face='B', elapsed_s=12)
    panel.refresh()
    assert 'not measured' in panel.range_label.text()
    assert panel.heatmap.image.pixelColor(2, 1).red() == 80
    panel.mode_box.setCurrentIndex(0)
    assert payload['faces']['A']['counts'][1, 2] == 1
    assert 'updated at 10.0' in panel.range_label.text()
