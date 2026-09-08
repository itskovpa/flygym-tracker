"""Vial positions can be drawn on the recording a replay is playing.

THE BUG THIS FIXES, reported from the rig: "I cannot draw vials if I am replaying the video."

Two separate causes, and the second one lied. `VideoStage.begin_draw` has always taken a `video=`
argument and known how to draw on a recording -- but `MainWindow._begin_draw` never passed it, so
that path was unreachable dead code and every draw demanded the live camera. And because the request
went through `with_camera`, a running replay answered "the experiment has the camera - stop the run
first": true of a live run, false of a replay, which holds no camera at all and reads a file.

The running replay keeps the calibration it started with -- its masks were built at start -- so
anything drawn now applies to the NEXT replay. The operator is told that rather than left to infer
it, which is the difference between a useful tool and a misleading one.
"""
from __future__ import annotations

import pytest

from flygym_tracker.config import load_config
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.main_window import MainWindow


class _Run:
    """Stand-in for RunController: `_begin_draw`/`with_camera` only ask whether a run is going."""

    def __init__(self, running: bool):
        self.is_running = running
        self.state = "running" if running else "idle"
        self.detail = ""


@pytest.fixture
def window(qapp, tmp_path):
    state = gui_state.default_state()
    state["calib_dir"] = str(tmp_path / "calib")
    state["output_dir"] = str(tmp_path / "out")
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"), state=state,
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda text: True)
    win.show()
    yield win
    win.session.shutdown()


@pytest.fixture
def drawn(window, monkeypatch):
    """Capture what `stage.begin_draw` was asked for, without starting a real drawing session."""
    calls = []

    def fake_begin_draw(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(window.stage, "begin_draw", fake_begin_draw)
    return calls


def test_drawing_during_a_replay_draws_on_that_recording(window, drawn):
    """THE REPORTED BUG. The recording being replayed is exactly what should be drawn on."""
    window._replay_video = r"C:\clips\rig.avi"
    window.run = _Run(running=True)

    window._begin_draw(str(window.state["calib_dir"]))

    assert drawn, "pressing Draw vial positions during a replay did nothing"
    assert drawn[0].get("video") == r"C:\clips\rig.avi", \
        "the draw did not target the recording being replayed (%r)" % drawn[0].get("video")


def test_the_operator_is_told_the_running_replay_keeps_its_own_vials(window, drawn):
    """Its masks were built at start, so what is drawn now changes the NEXT replay. Silence here
    would let an operator believe they had corrected the run they are watching."""
    window._replay_video = r"C:\clips\rig.avi"
    window.run = _Run(running=True)

    window._begin_draw(str(window.state["calib_dir"]))

    notice = window.stage._notice or ""
    assert "NEXT replay" in notice, "nothing said the running replay keeps its old vial positions"


def test_a_replay_is_not_described_as_holding_the_camera(window):
    """It reads a file. Telling the operator to free a camera sent them after a phantom problem."""
    window._replay_video = r"C:\clips\rig.avi"
    window.run = _Run(running=True)

    window.with_camera(lambda: None, why="draw the vial positions")

    text = window.stage.caption.text()
    assert "replay" in text.lower(), "a running replay still claims to hold the camera: %r" % text
    assert "has the camera" not in text


def test_a_live_run_still_does_hold_the_camera(window):
    """THE REGRESSION GUARD. A live run genuinely owns the exclusive USB3 handle, and the honest
    answer there is unchanged -- the fix must not hand out the camera mid-experiment."""
    window._replay_video = None
    window.run = _Run(running=True)

    window.with_camera(lambda: None, why="draw the vial positions")

    assert "has the camera" in window.stage.caption.text()


def test_with_no_run_the_camera_path_is_unchanged(window, drawn, monkeypatch):
    """The rig owner's rule is that live camera is prioritised for measurements. Drawing with
    nothing playing must still go through `with_camera`, not quietly fall back to a file."""
    asked = []
    monkeypatch.setattr(window, "with_camera",
                        lambda then, why: asked.append(why))
    window._replay_video = None
    window.run = _Run(running=False)

    window._begin_draw(str(window.state["calib_dir"]))

    assert asked, "drawing with no run bypassed the live-camera path"
    assert not drawn, "it drew on a file instead of asking for the camera"


# =============================================================================================
# Through the BUTTON, not the method
# ---------------------------------------------------------------------------------------------
# The first version of these tests called `_begin_draw` directly and passed while the button that
# reaches it was greyed out -- so they proved the downstream logic and missed the actual bug. A
# test for "I cannot press this" has to ask whether it can be pressed.
# =============================================================================================
from flygym_tracker.gui.run_controller import RUNNING, DONE, IDLE          # noqa: E402


def test_the_draw_button_is_clickable_while_a_replay_plays(window):
    """THE REPORTED BUG, at the level the operator meets it: the button was grey."""
    window._replay_video = r"C:\clips\rig.avi"
    window._on_run_state(RUNNING, "replaying")

    assert window.run_panel.tool_draw_vials_button.isEnabled(), \
        "Draw vial positions is still greyed out during a replay"


def test_a_live_run_still_greys_the_video_tools(window):
    """THE REGRESSION GUARD. A live run really does hold the exclusive camera, and offering a job
    that could only fail with the SDK's culprit-free error is what the blocking is for."""
    window._replay_video = None
    window._on_run_state(RUNNING, "running")

    assert not window.run_panel.tool_draw_vials_button.isEnabled()


def test_only_the_wired_job_is_offered_during_a_replay(window):
    """`_begin_draw` is the one job that hands the recording through. Enabling the others would be
    the same broken promise moved somewhere new: a button whose job then refuses."""
    window._replay_video = r"C:\clips\rig.avi"
    window._on_run_state(RUNNING, "replaying")

    for action in ("mark_band", "noise", "learn_faces", "replay"):
        assert not getattr(window.run_panel, "tool_%s_button" % action).isEnabled(), \
            "%s was offered during a replay but is not wired to work on the recording" % action


def test_the_tools_come_back_when_the_replay_ends(window):
    """And the replay flag must not stick: a finished replay is not a replay."""
    window._replay_video = r"C:\clips\rig.avi"
    window._on_run_state(RUNNING, "replaying")
    window._on_run_state(DONE, "finished")

    assert window.run_panel.tool_draw_vials_button.isEnabled()
    assert window.run_panel.tool_replay_button.isEnabled(), "replay stayed blocked after the run"
    assert window._replay_video is None


# =============================================================================================
# Through the PICTURE as well as the button
# ---------------------------------------------------------------------------------------------
# The button tests above passed while the real window still greyed the button out. `_begin_replay`
# calls `stage.show_run()` after the run starts, the stage announces RUN mode, and the window
# counted that as "the stage is busy" -- which vetoed the replay allowance the tests had checked.
# Reported from the rig: "Draw vial positions is not active while I replay a recording." A test
# for a run has to take the picture the way a run does.
# =============================================================================================
def test_the_draw_button_survives_the_replay_taking_the_picture(window):
    """THE SECOND REPORTED BUG. The picture switching to the run must not grey the button."""
    window._replay_video = r"C:\clips\rig.avi"
    window._on_run_state(RUNNING, "replaying")
    window.stage.show_run()

    assert window.run_panel.tool_draw_vials_button.isEnabled(), \
        "Draw vial positions went grey the moment the replay's picture appeared"


def test_a_live_run_taking_the_picture_still_greys_the_tools(window):
    """RUN mode not counting as busy must not unblock anything for a live run."""
    window._replay_video = None
    window._on_run_state(RUNNING, "running")
    window.stage.show_run()

    assert not window.run_panel.tool_draw_vials_button.isEnabled()


def test_another_job_holding_the_picture_still_blocks_the_tools_during_a_replay(window):
    """Busy still means busy: a drawing in progress must not offer a second drawing."""
    from flygym_tracker.gui.video_stage import DRAW

    window._replay_video = r"C:\clips\rig.avi"
    window._on_run_state(RUNNING, "replaying")
    window._on_stage_mode(DRAW)

    assert not window.run_panel.tool_draw_vials_button.isEnabled()


# =============================================================================================
# The picture goes back to the replay when the drawing ends
# ---------------------------------------------------------------------------------------------
# A drawing used to end on the camera placeholder, "No picture - the camera is not open", while
# the replay it was drawn over was still running underneath. Twice over: `_on_draw_finished` went
# to the camera, and the file reader that had been feeding the drawing reported its own end a beat
# later and went to the camera again.
# =============================================================================================
from flygym_tracker.gui.video_stage import CAMERA, RUN, VideoStage        # noqa: E402


class _Box:
    def put(self, image):
        self._frame = image

    def take(self):
        return None


class _Session:
    latest = _Box()
    is_open = False
    measured_fps = 0.0
    tap = None


def _stage_drawing_over_a_replay(qapp, tmp_path, monkeypatch, running=True):
    stage = VideoStage(_Session(), _Run(running=running))
    stage.resize(200, 150)
    stage.show()
    stage.show_run()
    monkeypatch.setattr(stage, "_start_file_job", lambda video, job: True)
    assert stage.begin_draw(out_dir=str(tmp_path), video=r"C:\clips\rig.avi")
    return stage


def test_cancelling_a_drawing_returns_the_picture_to_the_running_replay(qapp, tmp_path, monkeypatch):
    stage = _stage_drawing_over_a_replay(qapp, tmp_path, monkeypatch)
    stage.draw_cancel_button.click()
    assert stage.mode == RUN, "the picture went to %r instead of back to the replay" % stage.mode


def test_the_drawings_frame_feed_ending_late_leaves_the_replay_on_screen(qapp, tmp_path, monkeypatch):
    stage = _stage_drawing_over_a_replay(qapp, tmp_path, monkeypatch)
    reported = []
    stage.job_finished.connect(lambda kind, payload: reported.append(kind))
    stage.draw_cancel_button.click()

    stage._on_file_job_finished({"frames": 12})       # the feed's own end, arriving afterwards

    assert stage.mode == RUN, "the feed ending took the picture off the replay"
    assert reported == ["draw"], "the feed ending was reported as a job of its own: %r" % reported


def test_with_no_run_a_finished_drawing_still_returns_to_the_camera(qapp, tmp_path, monkeypatch):
    stage = _stage_drawing_over_a_replay(qapp, tmp_path, monkeypatch, running=False)
    stage.draw_cancel_button.click()
    assert stage.mode == CAMERA


# =============================================================================================
# A replay obeys the "activity only" tickbox like a live run does
# =============================================================================================
class _PlanCatcher(_Run):
    """A RunController that records the plan it was handed and refuses to start."""

    def __init__(self):
        super().__init__(running=False)
        self.plans = []

    def start(self, plan):
        self.plans.append(plan)
        return False


@pytest.mark.parametrize("track", [True, False])
def test_a_replay_carries_the_tracking_tickbox_into_its_plan(window, monkeypatch, track):
    """REPORTED FROM THE RIG: "fly tracking tickbox does not seem to work - even with it selected
    the software draws fly tracks." The replay plan simply never carried the tickbox."""
    window.run = _PlanCatcher()
    window.session_bar.set_tracking(track)
    monkeypatch.setattr(window, "_pick_video", lambda title: "C:/clips/rig.avi")

    window._begin_replay(str(window.state["calib_dir"]))

    assert window.run.plans, "the replay was not started"
    assert window.run.plans[0].get("track_flies") is track,         "the replay plan says track_flies=%r with the tickbox at %r" % (
            window.run.plans[0].get("track_flies"), track)
