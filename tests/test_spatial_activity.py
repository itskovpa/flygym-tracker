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

def test_inspector_step_buttons_respond_to_mouse_clicks(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractSpinBox
    panel = ActivityHeatmapPanel({})
    panel.mode_box.setCurrentIndex(1)
    panel.resize(700, 600)
    panel.show()
    panel.threshold_box.setValue(4)
    qapp.processEvents()
    stepper = panel.threshold_stepper
    assert panel.threshold_box.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
    assert stepper.up.isVisible() and stepper.up.width() > 0
    QTest.mouseClick(stepper.up, Qt.MouseButton.LeftButton, pos=stepper.up.rect().center())
    assert panel.threshold_box.value() == 5
    QTest.mouseClick(stepper.down, Qt.MouseButton.LeftButton, pos=stepper.down.rect().center())
    assert panel.threshold_box.value() == 4
    panel.threshold_box.setValue(255)
    assert not stepper.up.isEnabled()
    panel.threshold_box.setValue(0)
    assert not stepper.down.isEnabled()


def test_registered_counts_do_not_smear_and_do_not_wrap_at_edges():
    store = SpatialActivity()
    gray = np.full((5, 8), 100, np.uint8)
    motion = np.zeros_like(gray, bool); motion[2, 2] = True
    store.add('A', gray, motion, offset=(0, 0))
    moved = np.zeros_like(motion); moved[2, 4] = True
    moved[2, 0] = True  # This camera pixel lies outside the aligned reference image.
    store.add('A', gray, moved, offset=(2, 0))
    result = store.snapshot(10)['faces']['A']
    assert np.argwhere(result['counts']).tolist() == [[2, 2]]
    assert result['counts'][2, 2] == 2
    assert result['rate'][2, 2] == 1
    assert result['mean'][2, 7] == 100  # No zero padding enters the average.


def test_frame_sum_exceeds_16_bits_and_includes_reference_without_motion():
    store = SpatialActivity()
    gray = np.full((2, 3), 255, np.uint8)
    for _ in range(300):
        store.add('A', gray, None)
    data = store.faces['A']
    assert data['intensity_sum'].dtype == np.uint64
    assert data['intensity_sum'][0, 0] == 76500
    published = store.snapshot(10)['faces']['A']
    assert published['image_frames'] == 300 and published['frames'] == 0
    assert np.all(published['mean'] == 255)
    assert published['background_samples'] <= 32
    assert not data['rolling'].samples[0][1].flags.writeable
    store.close()


def test_lighting_corrected_background_recovers_known_relative_darkness():
    from concurrent.futures import Future
    from flygym_tracker.rolling_background import RollingBackground
    class ImmediateExecutor:
        def submit(self, fn, values):
            future = Future();future.set_result(fn(values));return future
    rolling = RollingBackground(50, window_s=30, executor=ImmediateExecutor())
    for i in range(64):
        light = 200 if i % 2 else 100
        gray = np.full((5, 10), light, np.uint8)
        gray[0, 0] = light//2
        if 32 <= i < 40:
            gray[2, 3] = light//2
        normal = gray.astype(np.float32) / np.percentile(gray, 90)
        if i == 32:
            rolling.total[:] = 0;rolling.count[:] = 0
        rolling.add(normal.ravel(), i)
    result = rolling.mean().reshape(5, 10)
    assert result[0, 0] == 0
    assert result[1, 1] == 0
    assert result[2, 3] == .125  # 25% occupation x 50% contrast, not occupancy time.


def test_moving_window_expires_old_background_and_keeps_corrected_history():
    from concurrent.futures import Future
    from flygym_tracker.rolling_background import RollingBackground
    class ImmediateExecutor:
        def submit(self, fn, values):
            future = Future();future.set_result(fn(values));return future
    rolling = RollingBackground(1, window_s=10, executor=ImmediateExecutor())
    for i in range(8):
        rolling.add(np.array([1.0]), i)
    rolling.add(np.array([.5]), 8)
    history = rolling.total.copy()
    assert history[0] == .5
    rolling.add(np.array([.6]), 30)
    assert len(rolling.samples) == 1
    assert rolling.background is None
    np.testing.assert_array_equal(history, rolling.total)
    for i in range(31, 45):
        rolling.add(np.array([.6]), i)
    assert np.allclose(rolling.background, .6)
    np.testing.assert_array_equal(history, rolling.total)
    rolling.set_window(20)
    assert rolling.background is None and not rolling.samples
    np.testing.assert_array_equal(history, rolling.total)


def test_new_map_modes_switch_without_stale_render_and_keep_faces_separate(qapp):
    store = SpatialActivity()
    gray = np.full((4, 5), 100, np.uint8)
    store.add('A', gray, None)
    darker = gray.copy(); darker[1, 2] = 50
    store.add('A', darker, np.zeros_like(gray, bool))
    payload = store.snapshot(10)
    panel = ActivityHeatmapPanel(payload)
    panel.mode_box.setCurrentIndex(2)
    assert '2 stationary frames' in panel.range_label.text()
    panel.mode_box.setCurrentIndex(3)
    assert 'background samples' in panel.range_label.text()
    assert 'Learning background' in panel.note.text()
    requested = []
    panel.background_window_requested.connect(requested.append)
    panel.background_window.setValue(60)
    assert requested == [60]
    panel.face_box.setCurrentIndex(1)
    assert panel.heatmap.image is None
    panel.close()


def test_fast_background_percentile_matches_nanpercentile():
    import warnings
    from flygym_tracker.rolling_background import bright_percentile
    values = np.random.default_rng(4).random((32, 100))
    values[::3, ::7] = np.nan
    values[:, 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = np.nanpercentile(values, 90, axis=0)
    np.testing.assert_allclose(bright_percentile(values), expected, atol=1e-14)


def test_cached_mask_updates_equal_general_mask_updates():
    rng = np.random.default_rng(8)
    mask = rng.random((20, 30)) > .4
    mask.setflags(write=False)
    fast, general = SpatialActivity(), SpatialActivity()
    for i in range(8):
        gray = rng.integers(0, 256, mask.shape, dtype=np.uint8)
        motion = rng.random(mask.shape) > .8
        fast.add('A', gray, motion, mask, elapsed_s=i/20)
        general.add('A', gray, motion, mask.copy(), elapsed_s=i/20)
    for key in ('intensity_sum', 'samples', 'counts', 'normal_sum', 'pair_samples'):
        np.testing.assert_array_equal(fast.faces['A'][key], general.faces['A'][key])
    fast.close();general.close()


def test_live_background_window_setting_routes_through_pipeline(tmp_path):
    from test_pipeline import _config, _calibration, _full_scene, FakeSource
    from flygym_tracker.pipeline import TrackerPipeline
    from flygym_tracker.logger import ActivityLogger
    logger = ActivityLogger(tmp_path/'out', 'background-window', fmt='csv')
    pipe = TrackerPipeline(_config(), _calibration(tmp_path), FakeSource(_full_scene()), logger, clock='index')
    pipe.spatial_activity = SpatialActivity()
    assert pipe.apply_setting('spatial.background_window_s', 60)
    assert pipe.spatial_activity.background_window_s == 60
    assert not pipe.apply_setting('spatial.background_window_s', 0)
    assert pipe.spatial_activity.background_window_s == 60
    pipe.spatial_activity.close();logger.close()
