"""A video button does the job, including opening the camera. It does not describe a precondition.

THE REPORT, verbatim: "when I stop the experiment and click draw vial positions the software does
not react until I click open camera. Make it automatic, as soon as I click draw vial positions or
learn drum faces it needs to do the necessary steps for the operation, always prioritize live
camera for all measurements."

Before this, pressing Draw vial positions with the camera closed did NOTHING VISIBLE: the job
refused, wrote a sentence into the caption under the picture, and the operator was left to work out
that a different button in a different band had to be pressed first.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import numpy as np                                                      # noqa: E402


class FakeSource:
    """A camera that opens successfully and delivers frames."""

    serial = "FAKE"
    frame_size = (64, 48)

    def open(self):
        pass

    def close(self):
        pass

    def current_values(self):
        return {}

    def ranges(self):
        return {}

    def read(self):
        return type("F", (), {"image": np.full((48, 64), 90, dtype=np.uint8)})()


class RefusingSource(FakeSource):
    def open(self):
        raise RuntimeError("0x80000203 the device is already in use")


@pytest.fixture
def window(qapp, tmp_path, request):
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    factory = getattr(request, "param", FakeSource)
    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=factory, confirm=lambda text: True)
    win.show()
    qapp.processEvents()
    yield win
    win.run.shutdown()
    win.session.shutdown()


# =============================================================================================
# The camera is opened for the job
# =============================================================================================
def test_a_job_opens_the_camera_itself(qapp, window, pump):
    assert not window.session.is_open
    ran = []
    window.with_camera(lambda: ran.append(True), why="do the thing")
    assert ran == [], "the job ran before the camera was streaming"

    pump(lambda: bool(ran), timeout=5.0)
    assert ran == [True], "the job never ran after the camera opened"
    assert window.session.is_open


def test_an_already_open_camera_runs_the_job_at_once(qapp, window, pump):
    """No round trip when there is nothing to wait for -- an operator pressing the button twice
    should not see a delay the second time."""
    window.open_camera()
    pump(lambda: window.session.is_open)
    ran = []
    window.with_camera(lambda: ran.append(True), why="do the thing")
    assert ran == [True], "an open camera still made the job wait"


def test_drawing_vial_positions_no_longer_needs_open_camera_pressed_first(qapp, window, pump):
    """THE REPORTED BUG. The button did not do its job; it described a precondition."""
    from flygym_tracker.gui.video_stage import DRAW

    window._on_tool("draw_vials")
    pump(lambda: window.stage.mode == DRAW, timeout=5.0)
    assert window.stage.mode == DRAW, "Draw vial positions still does nothing on a closed camera"
    assert window.session.is_open


def test_learning_the_faces_prefers_the_live_camera_over_asking_for_a_recording(qapp, window,
                                                                                pump):
    """The live camera IS the point of this step -- it watches the drum turn NOW. Asking which
    recording to use before even trying the camera had it backwards, and a file dialog appearing
    on a rig with a working camera is a question with an obviously wrong default."""
    from flygym_tracker.gui.video_stage import JOB

    asked = []
    window._pick_video = lambda title: asked.append(title)
    window._on_tool("learn_faces")
    pump(lambda: window.stage.mode == JOB, timeout=5.0)
    assert asked == [], "it asked for a recording instead of using the camera"
    assert window.stage.mode == JOB
    assert window.session.is_open


# =============================================================================================
# When the camera cannot be had
# =============================================================================================
@pytest.mark.parametrize("window", [RefusingSource], indirect=True)
def test_a_camera_that_refuses_says_so_rather_than_leaving_the_job_pending(qapp, window, pump):
    """A job left waiting forever on a camera that will never arrive is the same silence the
    button had before, one step later."""
    ran = []
    window.with_camera(lambda: ran.append(True), why="draw the vial positions")
    pump(lambda: window._camera_then is None, timeout=5.0)
    assert ran == [], "the job ran on a camera that never opened"
    text = window.stage.caption.text()
    assert "could not open the camera" in text
    assert "draw the vial positions" in text, "the message does not name the job that wanted it"


def test_a_running_experiment_keeps_the_camera_and_the_job_says_why(qapp, window):
    """Stopping the run to draw vials would end an experiment in order to change its calibration,
    and opening a second handle would fail with the SDK's culprit-free error."""
    ran = []
    window.run._state = "running"
    window.with_camera(lambda: ran.append(True), why="draw the vial positions")
    assert ran == []
    assert "stop the run first" in window.stage.caption.text()


def test_the_camera_is_still_never_taken_without_a_click(qapp, window):
    """The rule this does NOT break. "The app never takes the camera by itself" is about LAUNCH:
    an app that grabs an exclusive device when it opens is an app that blocks the rig. Opening it
    because a button whose meaning is "do this to the rig now" was pressed is the opposite."""
    assert not window.session.is_open, "the window took the camera without being asked"
