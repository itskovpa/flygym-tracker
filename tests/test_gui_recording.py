"""The recording tick box, and the measurement moving under the picture.

Both are the rig owner's calls:

  * "put this panel underneath the main image with the video and tracks the same width as the
    video";
  * "I also need an option to record video ... By default video recording is off, activated by a
    tick box. Video recording settings need to be in the same collapsable menu (called Experiment)".
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def window(qapp, tmp_path):
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda text: True)
    win.show()
    qapp.processEvents()
    yield win
    win.run.shutdown()
    win.session.shutdown()


# =============================================================================================
# Under the picture, at the picture's width
# =============================================================================================
def test_the_measurement_sits_under_the_picture(qapp, window):
    """WHY IT IS THE RIGHT SHAPE, not just the asked-for one: the measurement is 32 cells in two
    rows of sixteen -- wide and short. Beside the picture it had to be squeezed into a column
    narrow enough to leave the drum visible, so the cells scrolled sideways and half of face B was
    off the edge."""
    from PySide6.QtCore import Qt

    assert window._splitter.orientation() == Qt.Orientation.Vertical
    assert window._splitter.indexOf(window.stage) == 0
    assert window._splitter.indexOf(window.results) == 1


def test_the_measurement_is_as_wide_as_the_picture(qapp, window):
    """The whole point of moving it. In a vertical splitter both panes are the splitter's width,
    so this is checking that the results really are IN that splitter rather than beside it."""
    window.results.setVisible(True)
    qapp.processEvents()
    assert window.results.width() == window.stage.width()


def test_the_picture_keeps_the_height_when_the_measurement_appears(qapp, window):
    """The measurement has a natural height -- two rows of cells and two lines of text -- and gains
    nothing from more. The picture is the pane that cannot be read by scrolling."""
    window.resize(1180, 820)               # a rig screen, not whatever the test platform defaults to
    window.results.setVisible(True)
    qapp.processEvents()
    window._share_width_with_results()
    qapp.processEvents()
    picture, results = window._splitter.sizes()
    assert results <= window.RESULTS_HEIGHT + 1, "the measurement took %d px of height" % results
    assert picture >= window.MIN_PICTURE_HEIGHT, "the picture was squeezed to %d px" % picture


def test_a_running_experiment_does_not_force_the_window_wider_than_the_rig_screen(qapp, window):
    """MEASURED REGRESSION, and the sixth of its kind on this project. `live_note` was a plain
    QLabel, so its minimum width was its whole sentence -- and its sentence grows once a run starts
    posting to it. The window's minimum went to 1740 px on a 1440 px desktop THE MOMENT A RUN
    BEGAN, which the existing layout tests could not catch: they build the window but never push a
    progress payload through it."""
    window.results.setVisible(True)
    window.results.set_progress({
        "vial_results": {i: (120, 900, 0.3) for i in range(1, 33)},
        "face": "A", "pixel_threshold": 15.0,
        "video": {"frames_written": 4820, "frames_dropped": 0, "bytes": 337 << 20, "fps": 10.0,
                  "error": None}})
    window.results.add_bin({"records": [{}] * 32})
    qapp.processEvents()
    assert window.minimumSizeHint().width() <= 1400, window.minimumSizeHint().width()


def test_the_experiment_paths_fold_away_when_the_run_starts(qapp, window):
    """A HEIGHT DECISION. The measurement now sits UNDER the picture, so it takes its ~95 px from
    the picture rather than from the width beside it -- and on the rig laptop's 900 px desktop the
    picture had none to spare. Those rows are paths that cannot be changed mid-run anyway, and the
    one-line summary stays on screen either way."""
    from flygym_tracker.gui.run_controller import RUNNING

    assert window.session_bar.is_expanded()
    window._on_run_state(RUNNING, "run 1")
    qapp.processEvents()
    assert not window.session_bar.is_expanded()
    assert window.session_bar.summary.text().strip(), "collapsing hid what experiment is running"


def test_a_run_never_re_opens_a_section_the_operator_folded(qapp, window):
    """Collapse only, never expand: a run must not fight the operator about their own layout."""
    from flygym_tracker.gui.run_controller import DONE, RUNNING

    window.session_bar.set_expanded(False)
    window._on_run_state(RUNNING, "run 1")
    window._on_run_state(DONE, "finished")
    qapp.processEvents()
    assert not window.session_bar.is_expanded()


# =============================================================================================
# The tick box
# =============================================================================================
def test_recording_is_off_by_default(qapp, window):
    """IT HAS TO BE OFF, not merely conventionally so: a full-rate recording of a three-day run is
    hundreds of gigabytes, and an operator who never asked for video must not find out by finding
    the disk full at hour 50 -- which takes the experiment with it."""
    assert not window.session_bar.record_box.isChecked()
    assert window.session_bar.recording_settings()["enabled"] is False


def test_the_tick_box_lives_in_the_collapsible_experiment_section(qapp, window):
    """As asked. It is also where it belongs: recording is a property of the experiment, chosen
    once with the paths, not of the camera or the algorithm."""
    bar = window.session_bar
    assert bar.record_box.isVisible()
    bar.set_expanded(False)
    qapp.processEvents()
    assert not bar.record_box.isVisible(), "the recording controls did not collapse with the rest"
    bar.set_expanded(True)
    qapp.processEvents()
    assert bar.record_box.isVisible()


def test_the_cost_knobs_are_dead_until_recording_is_on(qapp, window):
    bar = window.session_bar
    assert not bar.record_every.isEnabled()
    bar.record_box.setChecked(True)
    assert bar.record_every.isEnabled() and bar.record_scale.isEnabled()


def test_the_choice_is_carried_into_the_run_plan(qapp, window, monkeypatch):
    """The plan is read ONCE, when Start is pressed. A run cannot pick up a mid-run change to this,
    which is deliberate -- see `_on_recording_changed`."""
    bar = window.session_bar
    bar.record_box.setChecked(True)
    bar.record_every.setValue(3)
    bar.record_scale.setValue(0.5)

    seen = {}
    monkeypatch.setattr(window.run, "start", lambda plan: seen.update(plan) or False)
    window._start_run_now()
    assert seen["recording"] == {"enabled": True, "every_nth": 3, "scale": 0.5}


def test_the_choice_survives_a_restart(qapp, window, tmp_path):
    from flygym_tracker.gui import gui_state

    window.session_bar.record_box.setChecked(True)
    window.session_bar.record_every.setValue(5)
    qapp.processEvents()
    reloaded = gui_state.load_state(str(tmp_path))
    assert reloaded["recording"]["enabled"] is True
    assert reloaded["recording"]["every_nth"] == 5


def test_a_broken_saved_value_costs_that_value_and_not_the_rest(qapp, tmp_path):
    """A hand-edited state file that broke `scale` must not also lose the operator's `enabled`."""
    import json

    from flygym_tracker.gui import gui_state

    (tmp_path / "gui_state.json").write_text(
        json.dumps({"recording": {"enabled": True, "scale": "banana"}}), encoding="utf-8")
    state = gui_state.load_state(str(tmp_path))
    assert state["recording"]["enabled"] is True
    assert state["recording"]["scale"] == gui_state.DEFAULTS["recording"]["scale"]


# =============================================================================================
# What the run says about the video
# =============================================================================================
def test_dropped_frames_are_named_on_screen_while_there_is_time_to_act(qapp, window):
    """A recorder quietly dropping frames to a slow or filling disk looks exactly like one keeping
    up. The answer is only worth anything while the run can still be told to record less."""
    window.results.setVisible(True)        # the band appears with the run; a hidden child is never visible
    qapp.processEvents()
    window.results.set_progress({"vial_results": {}, "face": "A",
                                 "video": {"frames_written": 100, "frames_dropped": 12,
                                           "bytes": 5 << 20, "error": None}})
    text = window.results.recording_note.text()
    assert window.results.recording_note.isVisible()
    assert "12" in text and "DROPPED" in text


def test_the_disk_cost_is_projected_from_what_was_actually_written(qapp, window):
    """MEASURED, NOT ESTIMATED FROM THE SETTINGS. Bytes per frame depend entirely on the picture --
    a still drum compresses to nearly nothing, thirty-two vials of moving flies do not -- so the
    only honest projection divides what this run has already written. 100 frames of 1 MB each at
    5 fps is 5 MB/s, which is about 412 GB in a day."""
    window.results.setVisible(True)
    qapp.processEvents()
    window.results.set_progress({"vial_results": {}, "face": "A",
                                 "video": {"frames_written": 100, "frames_dropped": 0,
                                           "bytes": 100 << 20, "fps": 5.0, "error": None}})
    assert "GB/day" in window.results.recording_note.text()


def test_no_projection_before_there_is_enough_to_project_from(qapp, window):
    """Three frames in is not a rate. Extrapolating from it would put a number on screen that moves
    by a factor of ten in the first minute."""
    window.results.setVisible(True)
    qapp.processEvents()
    window.results.set_progress({"vial_results": {}, "face": "A",
                                 "video": {"frames_written": 3, "frames_dropped": 0,
                                           "bytes": 1 << 20, "fps": 5.0, "error": None}})
    assert "GB/day" not in window.results.recording_note.text()


def test_no_video_means_no_line_about_video(qapp, window):
    window.results.setVisible(True)        # the band appears with the run; a hidden child is never visible
    qapp.processEvents()
    window.results.set_progress({"vial_results": {}, "face": "A", "video": None})
    assert not window.results.recording_note.isVisible()


def test_a_failed_recording_says_the_measurement_is_unaffected(qapp, window):
    """The one thing an operator needs to know on seeing this, and the one thing a bare error
    message would leave them guessing about at hour 40 of an experiment."""
    window.results.setVisible(True)        # the band appears with the run; a hidden child is never visible
    qapp.processEvents()
    window.results.set_progress({"vial_results": {}, "face": "A",
                                 "video": {"error": "disk full", "frames_written": 3}})
    assert "unaffected" in window.results.recording_note.text()


def test_the_run_summary_names_the_drops(qapp):
    from flygym_tracker.gui.run_controller import _video_summary

    assert _video_summary(None) == ""
    assert "DROPPED" in _video_summary({"frames_written": 10, "frames_dropped": 4, "bytes": 1024})
    assert "DROPPED" not in _video_summary({"frames_written": 10, "frames_dropped": 0,
                                            "bytes": 1024})
    assert "VIDEO FAILED" in _video_summary({"error": "no codec"})
