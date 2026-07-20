"""The whole window: the bands, the initial focus, the two things that confirm, and the shutdown.

The confirmations are all driven through an INJECTED callable rather than a real `QMessageBox`, the
same shape `camera_lock.release_camera(holders, confirm=...)` already established. Nothing in this
file can block, and nothing can reach a real modal -- which is what lets a rig that runs unattended
experiments also run its own test suite.
"""
from __future__ import annotations

import os

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QAbstractSpinBox

from flygym_tracker.config import load_config
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.camera_session import CLOSED, STREAMING
from flygym_tracker.gui.main_window import MainWindow
from flygym_tracker.readiness import BAD

from test_gui_camera_session import FakeSource


@pytest.fixture
def rig_config(tmp_path):
    path = tmp_path / "rig.yaml"
    path.write_text(
        "binning:\n  bin_seconds: 60      # one row per minute\n"
        "activity:\n  pixel_threshold: 12.0  # above the sensor noise floor\n"
        "source:\n  camera:\n    frame_rate: null   # fps\n    exposure_us: null\n",
        encoding="utf-8")
    return str(path)


@pytest.fixture
def window(qapp, tmp_path, rig_config):
    answers = []
    state = gui_state.default_state()
    state["config_path"] = rig_config
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    sources = []

    def factory():
        source = FakeSource()
        sources.append(source)
        return source

    w = MainWindow(config=load_config(path=rig_config), config_path=rig_config, state=state,
                   root=str(tmp_path), camera_factory=factory,
                   confirm=lambda text: answers.append(text) or True)
    w.answers = answers
    w.sources = sources
    w.show()
    w.take_initial_focus()
    yield w
    w.session.shutdown()


# =============================================================================================
# The bands
# =============================================================================================
def test_the_window_has_all_five_bands(qapp, window):
    """One window, no navigation: a nav rail for two things is ceremony."""
    assert window.status_bar is not None
    assert window.session_bar is not None
    assert window.settings_view is not None and window.preview is not None
    assert window.readiness_strip is not None
    assert window.settings_view.save_button is not None


def test_the_session_bar_shows_the_three_paths_run_bat_kept_in_a_batch_file(qapp, window,
                                                                           rig_config, tmp_path):
    assert window.session_bar.config_path() == rig_config
    assert window.session_bar.calib_field.value() == str(tmp_path / "calib")
    assert window.session_bar.output_field.value() == str(tmp_path / "out")


def test_bin_seconds_is_a_settings_row_not_an_app_path(qapp, window):
    """It decides what a row of the results MEANS, so it belongs in the experiment's YAML -- and
    therefore in run_meta.json -- rather than in the app's own state file."""
    assert "binning.bin_seconds" in window.settings_view.rows
    assert "bin_seconds" not in window.state


def test_the_camera_identity_names_whichever_selector_is_load_bearing(qapp, window):
    """Serial wins in `_find_device`, and index is only consulted when no serial is pinned, so the
    line names the one that decides -- rather than printing an undocumented `index` key the app
    would otherwise inherit in silence."""
    window.session_bar.set_camera_identity("DA4282883", 0)
    assert "DA4282883" in window.session_bar.camera_label.text()
    assert "index is ignored" in window.session_bar.camera_label.text()

    window.session_bar.set_camera_identity(None, 2)
    assert "index 2" in window.session_bar.camera_label.text()


def test_every_job_run_bat_used_to_offer_is_a_live_button_in_this_window(qapp, window):
    """THIS TEST WAS INVERTED, AND ON PURPose.

    It used to assert the opposite -- that starting a run, drawing vial positions, replaying a
    recording and measuring the noise floor were NOT here, and that one line of text said they
    were still in `run.bat`. That was right while they genuinely lived elsewhere: four disabled
    buttons are four things to try before reading the small print.

    They live here now. `run.bat` is a one-line launcher, so a line of text pointing at its menu
    would point at a menu that no longer exists, and the assertion that guarded against dead
    buttons has to become the assertion that they are alive instead.
    """
    from PySide6.QtWidgets import QPushButton

    labels = [b.text() for b in window.findChildren(QPushButton)]
    for expected in ("Start experiment", "Draw vial positions", "Replay a recording",
                     "Measure noise floor"):
        assert any(expected in text for text in labels), "%r is not in this window" % expected

    # ALIVE, not merely present. A dead control is the thing the old test was written against.
    assert window.run_panel.start_button.isEnabled() is True
    assert window.run_panel.tool_draw_vials_button.isEnabled() is True
    assert not hasattr(window, "elsewhere"), \
        "the 'still in run.bat' line outlived the features moving in"


def test_nothing_in_this_window_sends_the_operator_to_a_terminal(qapp, window):
    """"No terminal prompt anywhere in the app." A button whose help says to go and type something
    is the numbered menu again, wearing a different hat."""
    from PySide6.QtWidgets import QLabel, QPushButton

    texts = [w.text() for w in window.findChildren(QLabel)]
    texts += [w.text() for w in window.findChildren(QPushButton)]
    texts += [w.toolTip() for w in window.findChildren(QPushButton)]
    for text in texts:
        assert "run.bat" not in (text or ""), text
        assert "python -m flygym_tracker" not in (text or ""), text


# =============================================================================================
# Initial focus
# =============================================================================================
def test_no_spinbox_holds_focus_when_the_window_opens(qapp, window):
    """MEASURED: Qt auto-focuses the first focusable widget of a shown window, and here that would
    be a settings spinbox -- so a stray keypress at 2 am would edit a camera setting before the
    operator had looked at anything."""
    qapp.processEvents()
    focused = qapp.focusWidget()
    assert not isinstance(focused, QAbstractSpinBox), type(focused).__name__
    assert focused is window.filter_box


def test_the_widget_that_takes_the_focus_edits_nothing(qapp, window):
    """The worst a stray keystroke can do is hide some rows."""
    before = {k: window.controller.model.value(k) for k in window.controller.model.keys()}
    window.filter_box.setText("expo")
    qapp.processEvents()
    assert {k: window.controller.model.value(k) for k in window.controller.model.keys()} == before
    assert window.settings_view.rows["source.camera.exposure_us"].isVisible()
    assert not window.settings_view.rows["rotation.sensitivity"].isVisible()


def test_filtering_hides_rows_without_excluding_them_from_the_file(qapp, window):
    window.controller.commit("rotation.sensitivity", 2.0)
    window.filter_box.setText("exposure")
    qapp.processEvents()
    assert any(s.key == "rotation.sensitivity" for s in window.controller.changed())


# =============================================================================================
# The camera, end to end through the window
# =============================================================================================
def test_the_window_does_not_touch_the_camera_until_asked(qapp, window):
    assert window.session.state == CLOSED
    assert window.sources == []


def test_opening_the_camera_turns_the_status_bar_green_and_names_the_serial(qapp, window, pump):
    window.open_camera()
    assert pump(lambda: window.session.state == STREAMING, timeout=5.0)
    qapp.processEvents()
    assert "DA4282883" in window.status_bar.sentence.text()
    assert window.status_bar.close_button.isEnabled() is True
    assert window.status_bar.open_button.isEnabled() is False


def test_free_the_camera_is_never_offered_while_this_app_is_the_holder(qapp, window, pump):
    """Otherwise the app would be offering to kill itself, and the operator would have no way to
    know that is what the button meant."""
    assert window.status_bar.free_button.isEnabled() is True
    window.open_camera()
    assert pump(lambda: window.session.state == STREAMING, timeout=5.0)
    qapp.processEvents()
    assert window.status_bar.free_button.isEnabled() is False


def test_a_live_setting_change_reaches_the_camera_once_it_is_open(qapp, window, pump):
    window.open_camera()
    assert pump(lambda: window.session.state == STREAMING, timeout=5.0)
    window.settings_view.rows["source.camera.exposure_us"].arm_button.click()
    qapp.processEvents()
    assert pump(lambda: window.sources[0].sent != [], timeout=5.0)
    assert window.sources[0].sent[0][0] == "exposure_us"


def test_with_no_camera_open_a_camera_edit_is_stored_but_sent_nowhere(qapp, window):
    """It is a config edit at that point, and it belongs in the file for the next run."""
    window.settings_view.rows["source.camera.gain_db"].arm_button.click()
    qapp.processEvents()
    assert window.controller.model.value("source.camera.gain_db") is not None
    assert window.sources == []


def test_the_status_bar_never_says_measured_before_a_frame_has_been_timed(qapp, window):
    """INVARIANT 6: a rate of zero next to the word "measured" is a claim nobody made."""
    window.status_bar.set_state(STREAMING, "DA4282883 is yours", measured_fps=0.0)
    assert "measured" not in window.status_bar.sentence.text()
    window.status_bar.set_state(STREAMING, "DA4282883 is yours", measured_fps=88.5)
    assert "88.5 fps delivered (measured)" in window.status_bar.sentence.text()


# =============================================================================================
# The readiness strip
# =============================================================================================
def test_a_missing_calibration_bundle_is_a_cross_on_the_strip(qapp, window):
    window.refresh_readiness()
    from flygym_tracker import readiness

    result = readiness.evaluate(config_path=window.controller.config_path,
                                calib_dir=window.state["calib_dir"],
                                output_dir=window.state["output_dir"],
                                camera_state=window.session.state)
    assert any(c.key == "calibration" and c.state == BAD for c in result.checks)


def test_a_row_armed_with_no_camera_shows_up_on_the_strip(qapp, window):
    """The one experiment-ruining hazard in this surface, surfaced where it will be met before
    saving rather than after."""
    window.settings_view.rows["source.camera.frame_rate"].arm_button.click()
    qapp.processEvents()
    assert window.controller.never_checked() == ["source.camera.frame_rate"]
    from flygym_tracker import readiness

    check = readiness.check_unverified(window.controller.never_checked(),
                                       {"source.camera.frame_rate": "frame rate"})
    assert check.state == BAD


# =============================================================================================
# Saving and closing -- the only two things that confirm
# =============================================================================================
def test_saving_writes_the_yaml_and_keeps_its_comments(qapp, window, rig_config):
    window.settings_view.rows["activity.pixel_threshold"].value_widget.setValue(20.0)
    qapp.processEvents()
    window.save_settings()
    text = open(rig_config, encoding="utf-8").read()
    assert "pixel_threshold: 20.0" in text
    assert "# above the sensor noise floor" in text
    assert "# one row per minute" in text


def test_an_ordinary_save_asks_nothing(qapp, window):
    window.settings_view.rows["activity.pixel_threshold"].value_widget.setValue(20.0)
    qapp.processEvents()
    window.save_settings()
    assert window.answers == []


def test_saving_a_value_no_camera_confirmed_asks_first(qapp, window):
    window.settings_view.rows["source.camera.frame_rate"].arm_button.click()
    qapp.processEvents()
    window.save_settings()
    assert len(window.answers) == 1
    assert "0.1 fps" in window.answers[0]


def test_closing_with_unsaved_changes_asks_and_can_be_cancelled(qapp, tmp_path, rig_config):
    state = gui_state.default_state()
    w = MainWindow(config=load_config(path=rig_config), config_path=rig_config, state=state,
                   root=str(tmp_path), camera_factory=lambda: FakeSource(),
                   confirm=lambda text: False)          # "cancel"
    try:
        w.show()
        w.controller.commit("activity.pixel_threshold", 30.0)
        event = QCloseEvent()
        w.closeEvent(event)
        assert event.isAccepted() is False, "the window closed over unsaved changes"
    finally:
        w.session.shutdown()


def test_closing_releases_the_camera_before_the_window_goes_away(qapp, window, pump):
    """LEAKING AN EXCLUSIVE USB3 HANDLE IS WHAT CREATES THE NEXT SESSION'S "camera is busy", with
    nothing on screen to explain it."""
    window.open_camera()
    assert pump(lambda: window.session.state == STREAMING, timeout=5.0)
    source = window.sources[0]
    window.closeEvent(QCloseEvent())
    assert source.closed >= 1
    assert source.is_acquiring is False


def test_closing_remembers_the_session_paths(qapp, window, tmp_path):
    window.closeEvent(QCloseEvent())
    assert os.path.isfile(tmp_path / "gui_state.json")
    reloaded = gui_state.load_state(str(tmp_path))
    assert reloaded["output_dir"] == window.state["output_dir"]


# =========================================================================================
# A run measures with what is ON SCREEN, not what was on disk at launch
# =========================================================================================
def test_a_run_uses_the_settings_the_operator_can_see(qapp, tmp_path):
    """Regression, and the worst kind this project can produce: days of quietly wrong data.

    `build_settings` seeds each Setting by COPYING out of the config, `SettingsModel.set` assigns
    only to the Setting, and saving rewrites the FILE. Nothing wrote back into the live config
    object, so a run built from it used whatever was on disk when the app launched. Measured
    before the fix: set pixel threshold 12 -> 25, save (banner reads "no changes", file holds 25),
    start the run -- and the pipeline measured at 12.0 for the whole run while the row showed 25.
    run_meta.json recorded 12.0 as well, so nothing downstream could have caught it.
    """
    import shutil
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow
    from test_gui_camera_session import FakeSource

    cfgp = tmp_path / "rig.yaml"
    shutil.copy("config/flygym_rig.yaml", cfgp)
    win = MainWindow(config=load_config(path=str(cfgp)), config_path=str(cfgp),
                     state=gui_state.default_state(), root=str(tmp_path),
                     camera_factory=lambda: FakeSource(), confirm=lambda t: True)
    try:
        assert win.config.activity.pixel_threshold == 12.0

        win.controller.commit("activity.pixel_threshold", 25.0)
        # UNSAVED edits count too: what is on screen is what gets measured, which is the only
        # rule that cannot surprise the operator.
        assert win._config_for_run().activity.pixel_threshold == 25.0

        win.save_settings()
        assert win._config_for_run().activity.pixel_threshold == 25.0
    finally:
        win.session.shutdown()


def test_the_run_config_still_imposes_nothing_on_an_untouched_camera_row(qapp, tmp_path):
    """Invariant 1 must survive the fix: overlaying the model must not materialise camera keys."""
    import shutil
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow
    from test_gui_camera_session import FakeSource

    cfgp = tmp_path / "rig.yaml"
    shutil.copy("config/flygym_rig.yaml", cfgp)
    win = MainWindow(config=load_config(path=str(cfgp)), config_path=str(cfgp),
                     state=gui_state.default_state(), root=str(tmp_path),
                     camera_factory=lambda: FakeSource(), confirm=lambda t: True)
    try:
        win.controller.commit("activity.pixel_threshold", 25.0)
        cam = win._config_for_run().source.camera
        for key in ("width", "height", "frame_rate", "exposure_us", "gain_db"):
            assert cam.get(key) is None, "%s must stay unset -- the sensor keeps MVS's value" % key
    finally:
        win.session.shutdown()


# =============================================================================================
# Start experiment: one click, and the camera handover is sequenced rather than raced
# =============================================================================================
def test_starting_a_run_asks_nothing(qapp, window, pump):
    """The operator has just pressed "Start experiment". Handing the camera over IS what they
    asked for, and a dialog whose only sensible answer is Yes trains people to click past the
    dialogs that matter."""
    window.open_camera()
    pump(lambda: window.session.is_open)
    before = len(window.answers)
    window.run.start = lambda plan: True
    window.start_run()
    qapp.processEvents()
    assert window.answers[before:] == [],         "starting the run asked a question: %r" % (window.answers[before:],)
    window.run.shutdown()


def test_the_run_waits_for_the_camera_to_actually_be_released(qapp, window, pump):
    """THE RACE THE OLD PROMPT WAS HIDING. `session.close()` only POSTS the close to the camera
    thread: `is_open` goes false at once, but the SDK handle is released later, on that thread.
    Closing and starting in the same breath was correct only because a human took a second or two
    to read the dialog. Without it, the run would try to open a camera the preview still holds --
    and the SDK's answer to that is a culprit-free 0x80000203.
    """
    started = []
    window.run.start = lambda plan: started.append(plan) or True
    window.open_camera()
    pump(lambda: window.session.is_open)

    window.start_run()
    assert started == [], "the run was started before the camera reported CLOSED"
    assert window._pending_start, "nothing remembered that a run was wanted"

    pump(lambda: bool(started), timeout=5.0)
    assert started, "the run never started after the camera closed"
    assert not window._pending_start
    window.run.shutdown()


def test_the_picture_switches_to_the_run_immediately_but_does_not_claim_to_be_it(
        qapp, window, pump):
    """"Seamless" must not become "dishonest": for the second or two of the handover what is on
    screen is the preview's LAST frame, and captioning that as the experiment is the same class of
    claim as calling a recording live."""
    from flygym_tracker.gui.video_stage import RUN

    window.run.start = lambda plan: True
    window.open_camera()
    pump(lambda: window.session.is_open)
    window.start_run()
    qapp.processEvents()

    assert window.stage.mode == RUN, "the picture did not switch to the run"
    window.stage._update_caption()
    assert "not the run" in window.stage.caption.text(), window.stage.caption.text()
    window.run.shutdown()


def test_a_camera_that_never_releases_does_not_silently_start_a_run(qapp, window):
    """Starting anyway would meet the SDK's culprit-free error from inside a worker thread -- the
    exact failure `camera_lock` exists to diagnose, arriving with no diagnosis."""
    from flygym_tracker.gui.video_stage import CAMERA

    started = []
    window.run.start = lambda plan: started.append(plan) or True
    window._pending_start = True
    window._handover_timed_out()
    assert started == []
    assert not window._pending_start
    assert window.stage.mode == CAMERA
    assert "did not release" in window.run_panel.state_label.text()
