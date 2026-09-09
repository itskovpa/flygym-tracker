import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from flygym_tracker.gui.tracking_setup import TrackingSetup, detection_preview
from flygym_tracker.gui import gui_state
from test_gui_main_window import window, rig_config


def snapshot():
    gray = np.full((30, 40), 150, np.uint8)
    dark = np.zeros(gray.shape, np.float32)
    dark[3:5, 3:5] = .2
    dark[10:14, 10:14] = .3
    dark[20:25, 20:25] = .4
    return dict(frame=gray, normalized=gray/200, background=np.ones(gray.shape),
                contrast=dark, geometry={1: ((0, 0, 40, 30), np.ones(gray.shape, bool))},
                elapsed_s=4, frame_index=80, face='B', vials={},
                stats=dict(processing_mean_ms=12.3, frames_completed=20, frames_submitted=22,
                           pending_frames=2, frames_dropped=3))


def test_classification_retains_large_regions_and_respects_mask():
    data = snapshot()
    binary, rgb, counts = detection_preview(data, .15, 8, 20)
    assert counts == [1, 1, 1]
    assert np.count_nonzero(binary) == 45
    assert tuple(rgb[21, 21]) == (255, 160, 35)
    data['geometry'][1][1][20:25, 20:25] = False
    assert detection_preview(data, .15, 8, 20)[2] == [1, 1, 0]


def test_preview_controls_do_not_save_until_apply(qapp):
    data = snapshot()
    state = gui_state.default_state()
    saved = []
    dialog = TrackingSetup(state, lambda: {'fast_tracking': data})
    dialog.applied.connect(saved.append)
    dialog.show()
    dialog.stage.setCurrentIndex(4)
    qapp.processEvents()
    assert dialog.image.image.pixelColor(11, 11).red() == 255
    spin = dialog.controls['fast_tracking_threshold']
    spin.setValue(35)
    assert dialog.image.image.pixelColor(11, 11).red() == 0
    assert saved == [] and state['fast_tracking_threshold'] == 15
    QTest.mouseClick(dialog.steppers['fast_tracking_threshold'].up, Qt.MouseButton.LeftButton)
    assert spin.value() == 36
    dialog.controls['fast_tracking_min_area'].setValue(301)
    dialog._apply()
    assert saved == [] and 'Nothing saved' in dialog.message.text()
    dialog.controls['fast_tracking_min_area'].setValue(8)
    QTest.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
    assert saved[0]['fast_tracking_threshold'] == 36
    for stage in range(7):
        dialog.stage.setCurrentIndex(stage)
        assert dialog.image.image is not None
    assert '12.3 ms' in dialog.performance.text()
    dialog.close()


def test_setup_values_reach_next_run_and_persist(window):
    window.show_tracking_setup()
    dialog = window._tracking_setup
    dialog.controls['fast_tracking_max_speed'].setValue(220)
    dialog.controls['fast_tracking_max_gap_s'].setValue(.45)
    dialog.controls['fast_tracking_max_group_s'].setValue(2)
    dialog.controls['spatial_background_window_s'].setValue(60)
    dialog.enabled.setChecked(False)
    dialog._apply()
    config = window._config_for_run()
    assert config.tracking.fast.max_speed == 220
    assert config.tracking.fast.max_gap_s == .45
    assert config.tracking.fast.max_group_s == 2
    assert config.tracking.background_window_s == 60
    assert not window.session_bar.track_flies()
    restored = gui_state.load_state(window.root)
    assert restored['fast_tracking_max_gap_s'] == .45
    assert isinstance(restored['fast_tracking_max_gap_s'], float)


def test_apply_does_not_change_running_background(window):
    from flygym_tracker.gui.run_controller import RUNNING
    window.show_tracking_setup()
    window.run._state = RUNNING
    window._tracking_setup.controls['spatial_background_window_s'].setValue(60)
    window._tracking_setup._apply()
    assert window.run.background_window_s == 120
    assert window.state['spatial_background_window_s'] == 60
    window.run._state = 'idle'


def test_warmup_clears_old_corrected_image(qapp):
    data = snapshot()
    holder = {'fast_tracking': data}
    dialog = TrackingSetup(gui_state.default_state(), lambda: holder)
    dialog.show()
    dialog.stage.setCurrentIndex(3)
    assert dialog.image.image is not None
    holder['fast_tracking'] = dict(data, frame=data['frame'].copy(), contrast=None, background=None)
    dialog.refresh()
    assert dialog.image.image is None
    assert 'warming up' in dialog.caption.text()


def test_freeze_keeps_matching_stages_while_run_advances(qapp):
    data = snapshot()
    holder = {'fast_tracking': data}
    dialog = TrackingSetup(gui_state.default_state(), lambda: holder)
    dialog.show()
    dialog.refresh()
    dialog.freeze.setChecked(True)
    holder['fast_tracking'] = dict(data, frame=np.full((30, 40), 200, np.uint8), elapsed_s=5)
    dialog.stage.setCurrentIndex(1)
    dialog.stage.setCurrentIndex(0)
    assert dialog.image.image.pixelColor(10, 10).red() == 150
    assert 'Frozen snapshot' in dialog.caption.text()
    dialog.freeze.setChecked(False)
    assert dialog.image.image.pixelColor(10, 10).red() == 200
