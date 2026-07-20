"""The window while a run is going: the knobs stay live, the geometry visibly does not.

This is the user's request stated as assertions. "Camera settings and algorithm controls need to be
available when the experiment is running so I could adjust the setting live." The settings pane is
never disabled, never replaced and never covered while a run is in progress -- only the two
start-only rows block, with their reason on the row, which is machinery that already existed.

INVARIANT 3 HAS TWO HALVES AND BOTH ARE TESTED. The pipeline REFUSES a geometry change
(`tests/test_gui_run_controller.py`), and the row LOOKS DEAD so it is not pressed in the first
place (here). A control that only refuses when pressed is a control the operator presses.
"""
from __future__ import annotations

import pytest

from flygym_tracker.config import load_config
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.main_window import MainWindow
from flygym_tracker.gui.run_controller import RUNNING

from test_gui_camera_session import FakeSource


@pytest.fixture
def window(qapp, tmp_path):
    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    w = MainWindow(config=load_config(path="config/flygym_rig.yaml"),
                   config_path="config/flygym_rig.yaml", state=state, root=str(tmp_path),
                   camera_factory=lambda: FakeSource(), confirm=lambda text: True)
    w.show()
    yield w
    w.run.shutdown()
    w.session.shutdown()


def pretend_running(window):
    """Put the window in the run state WITHOUT starting a real pipeline.

    There is no camera, no calibration bundle and no cv2 on the test path, so a real
    `RunController.start` cannot get past `_build`. What is under test here is the WINDOW's
    response to the run state, so the state is set directly and the window is told, exactly as the
    controller's own signal would.
    """
    window.run._state = RUNNING
    window._on_run_state(RUNNING, "run test")
    return window


# =============================================================================================
# The knobs stay live
# =============================================================================================
def test_the_settings_pane_is_never_disabled_while_a_run_is_going(qapp, window):
    """THE HEADLINE. A settings pane that greys out during a run is the thing the user asked us to
    stop doing -- the whole point is adjusting while watching the effect."""
    pretend_running(window)
    qapp.processEvents()
    assert window.settings_view.isEnabled() is True
    assert window.settings_view.isVisible() is True


def test_a_live_camera_row_is_still_editable_during_a_run(qapp, window):
    pretend_running(window)
    row = window.settings_view.rows["source.camera.exposure_us"]
    assert row.arm_button.isEnabled() is True, "exposure cannot be armed during a run"


def test_an_algorithm_row_is_still_editable_during_a_run(qapp, window):
    """`pixel_threshold` is re-read per frame by `_compute_vial_results`, so it needs nothing
    restarted and there is no reason to lock it."""
    pretend_running(window)
    row = window.settings_view.rows["activity.pixel_threshold"]
    assert row.value_widget.isEnabled() is True


def test_a_live_edit_during_a_run_is_routed_to_the_pipeline_not_the_camera(qapp, window):
    """It must go through `TrackerPipeline.apply_setting`, which applies AND logs it. The preview
    camera is closed during a run, so a change routed there would go nowhere at all."""
    routed = []
    window.run.apply_setting = lambda key, value: (routed.append((key, value)), True)[1]
    pretend_running(window)
    assert window._on_setting_change("activity.pixel_threshold", 18.0) is True
    assert routed == [("activity.pixel_threshold", 18.0)]


def test_with_no_run_and_no_camera_a_change_is_simply_stored(qapp, window):
    """"Takes effect at next start" is TRUE in that state, and it is what the cv2 panel says."""
    assert window.run.is_running is False
    assert window._on_setting_change("activity.pixel_threshold", 18.0) is True


# =============================================================================================
# INVARIANT 3 -- width and height LOOK dead, not merely refuse
# =============================================================================================
@pytest.mark.parametrize("key", ["source.camera.width", "source.camera.height"])
def test_geometry_rows_are_visibly_blocked_during_a_run(qapp, window, key):
    pretend_running(window)
    row = window.settings_view.rows[key]
    assert row.arm_button.isEnabled() is False, "%s can still be armed mid-run" % key


@pytest.mark.parametrize("key", ["source.camera.width", "source.camera.height"])
def test_a_blocked_geometry_row_says_why_in_place_of_its_help_line(qapp, window, key):
    """A greyed control with no reason is a support call, and this one has a real reason: changing
    it would restart acquisition under an experiment that may have been recording for days."""
    pretend_running(window)
    row = window.settings_view.rows[key]
    assert "stop the run" in row.help.text().lower()
    assert row.help.isVisible() is True, "a block reason must never be hover-only"


@pytest.mark.parametrize("key", ["source.camera.frame_rate", "source.camera.exposure_us",
                                 "source.camera.gain_db", "activity.pixel_threshold"])
def test_only_the_geometry_rows_block_during_a_run(qapp, window, key):
    """The blocking must be narrow. Locking the whole pane "to be safe" would defeat the request."""
    pretend_running(window)
    assert window.controller.block_reason(key) is None, "%s should stay live" % key


def test_the_geometry_block_lifts_when_the_run_ends(qapp, window):
    """A row that stayed dead after the run would look like a broken control at the exact moment
    the operator is setting up the next experiment."""
    pretend_running(window)
    assert window.controller.block_reason("source.camera.width") is not None
    window.run._state = "idle"
    window._on_run_state("idle", "")
    qapp.processEvents()
    assert window.controller.block_reason("source.camera.width") is None


# =============================================================================================
# The run band's own state
# =============================================================================================
def test_starting_is_refused_while_the_preview_camera_holds_the_camera(qapp, window):
    """The window closes the preview first (confirmed), so this is the backstop for a caller that
    did not. Two holders of one exclusive USB3 handle is invariant 5."""
    window._confirm = lambda text: False       # decline the "close the preview?" question
    window.session._state = "streaming"
    window.start_run()
    assert window.run.is_running is False


def test_the_run_band_disables_the_cv2_tools_while_a_run_is_going(qapp, window):
    """Those tools want the camera and the run has it. Offering the button anyway is offering a job
    that can only fail with the SDK's culprit-free error."""
    pretend_running(window)
    assert window.run_panel.tool_draw_vials_button.isEnabled() is False
    assert window.run_panel.start_button.isEnabled() is False
    assert window.run_panel.stop_button.isEnabled() is True


def test_the_run_band_says_the_settings_are_live(qapp, window):
    """In words, on screen. The operator should not have to discover it by trying."""
    pretend_running(window)
    assert "live" in window.run_panel.state_label.text().lower()


def test_the_progress_readout_shows_only_figures_the_pipeline_counted(qapp, window):
    """INVARIANT 6. Nothing here is sampled or estimated by the GUI."""
    window.run_panel.set_progress({"frames": 1234, "elapsed_s": 3661.0, "fps_est": 88.5,
                                   "n_rotations": 7, "face": "B", "vial_results": {}})
    text = window.run_panel.readout.text()
    assert "1234 frames" in text
    assert "1:01:01" in text
    assert "88.5 fps" in text
    assert "face B" in text
