"""Every video operation happens in the window now. These are the claims that has to survive.

THE ONE THAT MATTERS MOST is `test_a_click_on_the_widget_lands_on_the_right_image_pixel...`. The
whole point of moving the vial selector out of its cv2 window is that the picture is now
LETTERBOXED and SCALED into whatever rectangle the layout gives it -- so between the operator's
mouse and the polygon that gets saved there is a transform that did not exist before. If it is
wrong, nothing complains: the bundle saves, the run starts, the CSV fills, and every vial is
measured over the wrong patch of tube. So it is tested against a frame at a size the widget must
scale, with real mouse events, by reading the coordinates back out of the saved calibration.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt                                        # noqa: E402
from PySide6.QtTest import QTest                                            # noqa: E402

from flygym_tracker.gui.preview import PreviewWidget                        # noqa: E402
from flygym_tracker.gui.video_jobs import (FileJobController, NoiseJob,     # noqa: E402
                                           PassiveJob)
from flygym_tracker.gui.video_stage import (CAMERA, DRAW, JOB, RUN,         # noqa: E402
                                            VideoStage, _job_message,
                                            _job_progress_line)


# =============================================================================================
# Helpers
# =============================================================================================
class FakeBox:
    """A `LatestFrame` with no lock and no counters -- enough for the stage's pull timer."""

    def __init__(self):
        self._frame = None
        self.stats = (0, 0)

    def put(self, image):
        self._frame = image

    def take(self):
        frame, self._frame = self._frame, None
        return frame


class FakeSession:
    """A `CameraSession` shaped exactly as the stage uses one, with a settable tap."""

    def __init__(self, is_open=False):
        self.latest = FakeBox()
        self.is_open = is_open
        self.measured_fps = 0.0
        self.tap = None
        self.detached = 0

    def attach_tap(self, job):
        if not self.is_open or self.tap is not None:
            return False
        self.tap = job
        return True

    def detach_tap(self):
        if self.tap is not None:
            self.detached += 1
        self.tap = None


def _frame(width=64, height=48, value=90):
    return np.full((height, width), value, dtype=np.uint8)


def _stage(qapp, session=None):
    stage = VideoStage(session or FakeSession(is_open=True))
    stage.resize(400, 300)
    stage.show()
    return stage


# =============================================================================================
# The coordinate transform -- the part that did not exist before the picture was letterboxed
# =============================================================================================
def test_a_click_on_the_letterbox_margin_is_not_a_vertex(qapp):
    """A click beside the picture is not a corner the operator meant to place. Clamping it to the
    frame edge would silently put a polygon corner where nobody clicked."""
    view = PreviewWidget()
    view.resize(200, 200)
    view.set_frame(_frame(100, 50))          # letterboxed top and bottom
    assert view.to_image(100, 2) is None, "a click above the picture became a vertex"
    assert view.to_image(100, 100) is not None, "a click ON the picture was rejected"


def test_widget_and_image_coordinates_are_exact_inverses(qapp):
    view = PreviewWidget()
    view.resize(320, 240)
    view.set_frame(_frame(64, 48))           # scaled up 5x by the fit
    for point in ((0.0, 0.0), (10.0, 7.0), (63.0, 47.0)):
        back = view.to_image(*_xy(view.to_widget(*point)))
        assert back == pytest.approx(point, abs=1e-6)


def _xy(qpointf):
    return (qpointf.x(), qpointf.y())


def test_a_click_on_the_widget_lands_on_the_right_image_pixel_in_the_saved_bundle(qapp, tmp_path):
    """END TO END, with real mouse events, through the scale the layout imposes.

    The frame is 64x48 inside a 400x300 widget, so every click is scaled by ~6.25 and offset by the
    letterbox. If the stage saved widget coordinates -- or forgot the offset, or inverted the
    scale -- the polygon would still save and nothing would complain, so this reads the numbers
    back out of `calibration.json` and compares them against the frame's own pixel grid.
    """
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(64, 48))
    stage._pull()                                    # a frame is on screen, so there is a mapping

    out = str(tmp_path / "bundle")
    assert stage.begin_draw(out_dir=out, n_vials=1, faces=("A", "B"))
    stage._pull()

    # Three corners chosen in IMAGE space, converted to widget space, and clicked as a real mouse.
    wanted = [(10.0, 8.0), (40.0, 8.0), (40.0, 36.0)]
    for x, y in wanted:
        point = stage.view.to_widget(x, y)
        QTest.mouseClick(stage.view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(int(round(point.x())), int(round(point.y()))))
    assert len(stage.draw_session.state.current) == 3, "the clicks did not become vertices"

    stage.draw_finish_vial_button.click()            # one vial wanted, so this also finishes
    qapp.processEvents()

    saved = json.loads((tmp_path / "bundle" / "calibration.json").read_text())
    face = saved["faces"]["A"] if "A" in saved.get("faces", {}) else list(saved["faces"].values())[0]
    polygon = (face["vials"] if isinstance(face, dict) else face)[0]["polygon"]
    got = [(float(px), float(py)) for px, py in polygon]
    # Whole-pixel rounding at both ends of the transform, hence the 1px tolerance -- but a
    # forgotten offset or an inverted scale is tens of pixels out and this catches it.
    for (gx, gy), (wx, wy) in zip(got, wanted):
        assert abs(gx - wx) <= 1 and abs(gy - wy) <= 1, "click %r saved as %r" % ((wx, wy), (gx, gy))


def test_the_saved_frame_size_is_the_cameras_not_the_widgets(qapp, tmp_path):
    """The bundle records the IMAGE size. Recording the widget's would make every later run
    compare polygons against a frame of a different shape."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(64, 48))
    stage._pull()
    out = str(tmp_path / "b")
    stage.begin_draw(out_dir=out, n_vials=1)
    stage._pull()
    for x, y in ((5, 5), (30, 5), (30, 30)):
        stage.draw_session.on_click(x, y)
    stage.draw_session.finish_vial()
    qapp.processEvents()
    saved = json.loads((tmp_path / "b" / "calibration.json").read_text())
    assert (saved["image_width"], saved["image_height"]) == (64, 48)


# =============================================================================================
# The drawing session
# =============================================================================================
def test_holding_the_picture_stops_the_frame_changing_but_not_the_clicking(qapp, tmp_path):
    """SPACE exists because the drum turns and clicking a moving tube is hopeless. A freeze that
    let the picture keep updating would be a freeze in name only."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(value=10))
    stage._pull()
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=4)
    stage._pull()

    stage.draw_freeze_button.click()
    held = stage.view._array.copy()
    session.latest.put(_frame(value=250))            # a very different frame arrives
    stage._pull()
    assert np.array_equal(stage.view._array, held), "the held picture was replaced"

    stage.draw_session.on_click(5, 5)                # and it is still a drawing surface
    assert len(stage.draw_session.state.current) == 1


def test_the_keys_are_the_same_keys_the_cv2_selector_had(qapp, tmp_path):
    """The keymap is `live_vial_selector.handle_key`, driven from Qt -- not a second keymap that
    can drift out of step with the one the operator learnt at this rig."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame())
    stage._pull()
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=4)
    stage._pull()
    draw = stage.draw_session

    for x, y in ((5, 5), (20, 5), (20, 20)):
        draw.on_click(x, y)
    QTest.keyClick(stage.view, Qt.Key.Key_Backspace)
    assert len(draw.state.current) == 2, "BACKSPACE did not undo a corner"
    draw.on_click(20, 20)
    QTest.keyClick(stage.view, Qt.Key.Key_Return)
    assert len(draw.state.polygons) == 1, "ENTER did not store the vial"
    QTest.keyClick(stage.view, Qt.Key.Key_Space)
    assert draw.state.frozen, "SPACE did not hold the picture"


def test_the_view_only_takes_the_keyboard_while_drawing(qapp):
    """A picture that always held focus would swallow keystrokes the settings pane is entitled to
    -- and this window goes out of its way to keep initial focus off anything that edits a camera
    setting."""
    stage = _stage(qapp)
    assert not stage.view.interactive
    stage.begin_draw(out_dir="ignored", n_vials=4)
    assert stage.view.interactive
    stage.show_camera()
    assert not stage.view.interactive


def test_drawing_nothing_and_finishing_writes_no_bundle(qapp, tmp_path):
    out = tmp_path / "empty"
    stage = _stage(qapp)
    stage.begin_draw(out_dir=str(out), n_vials=4)
    results = []
    stage.job_finished.connect(lambda kind, payload: results.append((kind, payload)))
    stage.draw_session.finish()
    qapp.processEvents()
    assert not out.exists(), "an empty session wrote a calibration bundle"
    assert results and results[0][1]["saved"] is False


def test_a_session_with_no_frame_says_so_instead_of_saving_polygons_it_cannot_mask(qapp, tmp_path):
    """The illumination mask and the overlay are built from the picture the polygons were drawn
    on. With no frame there is nothing to build them from, and the operator has to be told that --
    not handed a silent success."""
    stage = _stage(qapp)
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=4)
    for x, y in ((1, 1), (5, 1), (5, 5)):
        stage.draw_session.on_click(x, y)
    stage.draw_session.state.finish_vial()
    results = []
    stage.job_finished.connect(lambda kind, payload: results.append(payload))
    stage.draw_session.finish()
    qapp.processEvents()
    assert results and results[0]["saved"] is False
    assert "no frame" in results[0]["message"]


def test_the_camera_is_never_taken_just_because_a_video_job_was_asked_for(qapp):
    """USB3 Vision is exclusive. An app that grabs the device because somebody clicked a button is
    an app that blocks the rig."""
    session = FakeSession(is_open=False)
    stage = _stage(qapp, session)
    assert stage.begin_draw(out_dir="x", n_vials=4) is False
    assert stage.begin_noise(np.ones((4, 4), np.uint8)) is False
    assert stage.mode == CAMERA
    assert session.tap is None
    assert "open the camera" in stage.caption.text()


# =============================================================================================
# Measurements
# =============================================================================================
def test_the_in_window_noise_floor_is_the_same_number_as_the_command_line_one(qapp):
    """ONE implementation, not two. This is a threshold that seeds every activity reading the rig
    takes afterwards; the window and the CLI disagreeing about it would be undetectable in the
    output and wrong in both."""
    from flygym_tracker.pipeline import measure_noise

    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 40, size=(16, 16), dtype=np.uint8) for _ in range(12)]
    mask = np.ones((16, 16), dtype=np.uint8) * 255

    class Source:
        def __init__(self):
            self.i = 0

        def open(self):
            pass

        def read(self):
            if self.i >= len(frames):
                return None
            frame = type("F", (), {"image": frames[self.i]})()
            self.i += 1
            return frame

    reference = measure_noise(Source(), mask, n_frames=len(frames), k=5.0)

    job = NoiseJob(mask, n_frames=len(frames), k=5.0)
    for frame in frames:
        job.observe(frame)
    assert job.done
    got = job.result()
    for key in ("noise_mean", "noise_std", "suggested_pixel_threshold",
                "suggested_enter_threshold", "suggested_exit_threshold"):
        assert got[key] == pytest.approx(reference[key]), key
    assert got["n_frames"] == reference["n_frames"]


def test_a_camera_measurement_sees_every_frame_not_the_decimated_preview(qapp):
    """THE REASON THE TAP IS ON THE CAMERA THREAD AND NOT ON THE PREVIEW BOX.

    The preview is decimated to ~15 fps out of up to 88, deliberately. A noise floor is measured
    from |frame - previous frame|, so a job fed from the preview would be differencing frames 66 ms
    apart instead of 11 ms -- a different measurement, silently. This drives the real
    `CameraWorker` grab loop with a source that hands out known frames and asserts the job counted
    all of them while the preview counted fewer.
    """
    from flygym_tracker.gui.camera_worker import CameraWorker, LatestFrame

    n = 40
    frames = [np.full((8, 8), i, dtype=np.uint8) for i in range(n)]

    class Source:
        serial = "TEST"
        frame_size = (8, 8)

        def __init__(self):
            self.i = 0

        def open(self):
            pass

        def close(self):
            pass

        def current_values(self):
            return {}

        def ranges(self):
            return {}

        def read(self):
            if self.i >= n:
                return None
            frame = type("F", (), {"image": frames[self.i]})()
            self.i += 1
            return frame

    box = LatestFrame()
    worker = CameraWorker(lambda: Source(), box)
    job = NoiseJob(np.ones((8, 8), np.uint8) * 255, n_frames=n)
    worker.open()
    worker.set_tap(job)
    for _ in range(n + 2):                    # drive the self-rearming loop by hand
        worker._grab_once()

    shown, dropped = box.stats
    assert job.frames == n, "the measurement missed %d of %d frames" % (n - job.frames, n)
    assert dropped > 0, ("the preview did not decimate in this test, so it proves nothing about "
                         "the tap seeing more than the preview")


def test_a_measurement_that_raises_is_detached_rather_than_raising_once_per_frame(qapp):
    from flygym_tracker.gui.camera_worker import CameraWorker, LatestFrame

    class Exploding:
        done = False

        def observe(self, image):
            raise RuntimeError("boom")

    class Source:
        serial = "T"
        frame_size = (4, 4)

        def open(self):
            pass

        def close(self):
            pass

        def current_values(self):
            return {}

        def ranges(self):
            return {}

        def read(self):
            return type("F", (), {"image": np.zeros((4, 4), np.uint8)})()

    worker = CameraWorker(lambda: Source(), LatestFrame())
    failures = []
    worker.tap_failed.connect(failures.append)
    worker.open()
    worker.set_tap(Exploding())
    worker._grab_once()
    worker._grab_once()
    assert worker.tap is None, "a raising job stayed attached"
    assert len(failures) == 1, "it was reported %d times, not once" % len(failures)


def test_stopping_a_measurement_early_keeps_what_it_measured_and_says_how_much(qapp):
    """A stop is not a cancel: 60 frames of noise floor is a real measurement of 60 frames, and it
    reports the count it actually used rather than the count it was asked for."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    assert stage.begin_noise(np.ones((8, 8), np.uint8) * 255, n_frames=1000)
    rng = np.random.default_rng(3)
    for _ in range(9):
        session.tap.observe(rng.integers(0, 30, size=(8, 8), dtype=np.uint8))

    results = []
    stage.job_finished.connect(lambda kind, payload: results.append((kind, payload)))
    stage.stop_job()
    qapp.processEvents()
    assert results, "stopping produced no result at all"
    kind, payload = results[0]
    assert kind == "noise"
    assert payload["n_frames"] == 9
    assert "9 frame(s)" in payload["message"]
    assert session.detached == 1, "the camera was left feeding a job nobody owns"
    assert stage.mode == CAMERA


def test_two_measurements_cannot_run_on_one_camera(qapp):
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    assert stage.begin_noise(np.ones((4, 4), np.uint8))
    first = session.tap
    assert stage.begin_face_learning() is False
    assert session.tap is first, "the second job displaced the first"


# =============================================================================================
# Modes
# =============================================================================================
def test_a_run_is_watched_in_the_same_picture_as_everything_else(qapp):
    class FakeRun:
        latest = FakeBox()

    session = FakeSession()
    stage = VideoStage(session, FakeRun())
    stage.resize(200, 150)
    stage.show_run()
    assert stage.mode == RUN
    FakeRun.latest.put(_frame(32, 24, 123))
    stage._pull()
    assert stage.view.frame_size == (32, 24), "the run's frames did not reach the picture"


def test_the_mode_is_announced_so_the_window_can_block_a_second_job(qapp):
    stage = _stage(qapp)
    seen = []
    stage.mode_changed.connect(seen.append)
    stage.begin_noise(np.ones((4, 4), np.uint8))
    stage.show_camera()
    assert seen == [JOB, CAMERA]


# =============================================================================================
# What the operator is told -- pure, so it is testable without a widget
# =============================================================================================
def test_a_finished_noise_measurement_names_what_it_measured_over():
    """A suggested threshold with no frame count behind it is a number nobody can judge."""
    message = _job_message("noise", {"n_frames": 120, "n_pairs": 119,
                                     "suggested_pixel_threshold": 3.5,
                                     "suggested_enter_threshold": 6.0,
                                     "suggested_exit_threshold": 3.0})
    assert "120 frame(s)" in message and "119 pair(s)" in message and "3.500" in message


def test_an_unfinished_face_learning_says_what_the_data_would_look_like():
    """Declining or aborting this step is the option that quietly produces WRONG data: the run
    still starts, still fills a CSV, and records every face-B vial as face A."""
    message = _job_message("faces", {"complete": False, "learned": ["A"]})
    assert "cannot tell the faces apart" in message
    assert "recorded as one face" in message


def test_progress_counts_frames_and_never_invents_a_percentage():
    assert _job_progress_line("noise", {"frames": 12, "n_target": 100, "pairs": 11}) \
        == "12 of 100 frames   -   11 usable pair(s)"
    assert _job_progress_line("faces", {"status": "waiting for the drum to turn"}) \
        == "waiting for the drum to turn"


# =============================================================================================
# Reading a recording
# =============================================================================================
def test_a_file_job_reads_the_whole_recording_and_reports_once(qapp, pump):
    frames = [np.full((8, 8), i, dtype=np.uint8) for i in range(6)]

    class Source:
        def __init__(self):
            self.i = 0
            self.closed = False

        def open(self):
            pass

        def close(self):
            self.closed = True

        def read(self):
            if self.i >= len(frames):
                return None
            frame = type("F", (), {"image": frames[self.i]})()
            self.i += 1
            return frame

    controller = FileJobController()
    job = PassiveJob()
    done = []
    controller.finished.connect(done.append)
    assert controller.start(Source, job)
    pump(lambda: bool(done), timeout=5.0)
    assert done, "the file job never finished"
    assert job.frames == len(frames)
    controller.shutdown()


# =============================================================================================
# "It starts and never stops, and the Stop button does nothing" -- reported from the rig
# =============================================================================================
class ExplodingJob:
    """A job that raises, which is what `camera_worker` detaches a tap for."""

    done = False

    def observe(self, image):
        raise RuntimeError("the marker band could not be read")

    def snapshot(self):
        return {"frames": 0}

    def stop(self):
        self.done = True

    def result(self):
        return {}


def test_stop_always_gets_the_operator_out_even_when_the_job_already_died(qapp):
    """THE REPORTED BUG. Face learning is the job that legitimately runs forever -- it ends when
    the drum has shown every face, so a drum that is not turning never ends it. Stop is the only
    way out, and it used to `return` silently whenever it could not find a job to stop, which is
    exactly the state a job that RAISED leaves behind (camera_worker detaches it). The stage then
    sat in JOB mode with the last progress line still on screen, looking like it was running.
    """
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    assert stage.begin_face_learning()
    assert stage.mode == JOB

    session.tap = None                    # what camera_worker does to a job that raised
    stage.stop_job()
    qapp.processEvents()
    assert stage.mode == CAMERA, "Stop left the stage in a mode it cannot leave"


def test_a_job_that_dies_on_the_camera_thread_is_noticed_rather_than_shown_as_running(qapp):
    """`camera_worker` detaches a raising tap and reports it. Nothing was listening, so a dead
    measurement looked exactly like a live one."""
    from PySide6.QtCore import QObject, Signal

    class SignallingSession(FakeSession, QObject):
        tap_failed = Signal(str)
        tap_finished = Signal()

        def __init__(self):
            QObject.__init__(self)
            FakeSession.__init__(self, is_open=True)

    session = SignallingSession()
    stage = VideoStage(session)
    stage.resize(300, 200)
    results = []
    stage.job_finished.connect(lambda kind, payload: results.append((kind, payload)))
    assert stage.begin_face_learning()

    session.tap = None
    session.tap_failed.emit("the marker band could not be read")
    qapp.processEvents()
    assert stage.mode == CAMERA
    assert results and results[0][0] == "faces", "the failure was not reported as a faces job"
    assert results[0][1]["failed"] is True
    assert "marker band" in stage.caption.text()


def test_a_measurements_result_survives_the_caption_refresh(qapp):
    """SECOND BUG, same family. `_pull` rewrites the caption every 50 ms from the current mode, so
    a result written straight into the label lasted one pull and was replaced by the live camera
    line -- silently throwing away the only place the suggested thresholds are ever reported."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    assert stage.begin_noise(np.ones((8, 8), np.uint8) * 255, n_frames=1000)
    rng = np.random.default_rng(5)
    for _ in range(12):
        session.tap.observe(rng.integers(0, 30, size=(8, 8), dtype=np.uint8))
    stage.stop_job()
    qapp.processEvents()
    assert "pixel threshold" in stage.caption.text()

    for _ in range(5):                    # the pulls that used to wipe it
        stage._pull()
    assert "pixel threshold" in stage.caption.text(), \
        "the measurement's result was overwritten by the live camera line"


def test_starting_a_new_job_clears_the_previous_result(qapp):
    """A stale result beside a running measurement reads as that measurement's own output."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    stage.begin_noise(np.ones((8, 8), np.uint8) * 255, n_frames=1000)
    rng = np.random.default_rng(6)
    for _ in range(12):
        session.tap.observe(rng.integers(0, 30, size=(8, 8), dtype=np.uint8))
    stage.stop_job()
    qapp.processEvents()
    assert "pixel threshold" in stage.caption.text()

    stage.begin_face_learning()
    stage._pull()
    assert "pixel threshold" not in stage.caption.text()


# =============================================================================================
# Loading saved vial positions: visible AND editable
# =============================================================================================
def _square(x, y, w=20, h=30):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def test_saved_positions_open_on_the_picture_instead_of_behind_a_yes_no(qapp, tmp_path):
    """Reuse used to be all-or-nothing: keep them exactly as they are, or throw them away and
    re-click all sixteen. A bundle that was 15/16 right cost a whole clicking session, and nothing
    on screen said it was 15/16 right."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(200, 200))
    stage._pull()
    saved = [_square(10, 10), _square(50, 10), _square(90, 10)]
    assert stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=16, polygons=saved)
    assert stage.draw_session.state.polygons == saved, "the saved vials are not on the picture"
    assert stage.draw_session.state.editing


def test_a_loaded_session_does_not_close_itself(qapp, tmp_path):
    """It opens ALREADY COMPLETE -- all 16 vials are there -- so the "all drawn, we are finished"
    rule would end it before the operator could touch anything."""
    from flygym_tracker.live_vial_selector import SelectorState

    state = SelectorState(n_vials=3)
    state.load([_square(0, 0), _square(30, 0), _square(60, 0)])
    assert state.is_complete
    assert not state.done, "a loaded session ended itself before anything could be edited"
    state.finish_early()
    assert state.done


def test_dragging_a_corner_moves_that_corner_and_places_no_new_one(qapp, tmp_path):
    """THE POINT OF "EDITABLE". Without the press being handled before the click, every attempt to
    adjust a vial would instead add a vertex to it."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(200, 200))
    stage._pull()
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=16, polygons=[_square(40, 40)])
    draw = stage.draw_session

    draw.on_press(40, 40)                       # exactly on the first corner
    draw.on_click(40, 40)                       # the click that follows every press
    draw.on_drag(44, 47)
    draw.on_release(45, 48)

    assert draw.state.polygons[0][0] == [45, 48], "the corner did not move"
    assert len(draw.state.polygons[0]) == 4, "dragging added a vertex"
    assert draw.state.current == [], "dragging started a new vial"


def test_a_press_far_from_any_corner_still_places_a_new_vertex(qapp, tmp_path):
    """Editing must not cost the ability to draw the vials that are still missing."""
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(200, 200))
    stage._pull()
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=16, polygons=[_square(10, 10)])
    draw = stage.draw_session

    draw.on_press(150, 150)
    draw.on_click(150, 150)
    assert draw.state.current == [[150, 150]]
    assert len(draw.state.polygons) == 1, "the loaded vial was disturbed"


def test_start_over_throws_everything_away(qapp, tmp_path):
    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(200, 200))
    stage._pull()
    stage.begin_draw(out_dir=str(tmp_path / "b"), n_vials=16,
                     polygons=[_square(10, 10), _square(50, 10)])
    stage.draw_restart_button.click()
    assert stage.draw_session.state.polygons == []
    assert not stage.draw_session.state.editing


def test_editing_a_loaded_selection_saves_the_edited_shape(qapp, tmp_path):
    """End to end: load, nudge a corner, finish -- and the moved corner is what reaches disk."""
    import json

    session = FakeSession(is_open=True)
    stage = _stage(qapp, session)
    session.latest.put(_frame(200, 200))
    stage._pull()
    out = tmp_path / "b"
    stage.begin_draw(out_dir=str(out), n_vials=1, polygons=[_square(40, 40)])
    draw = stage.draw_session
    draw.on_press(40, 40)
    draw.on_release(46, 49)
    draw.finish()
    qapp.processEvents()

    saved = json.loads((out / "calibration.json").read_text())
    face = list(saved["faces"].values())[0]
    assert face["vials"][0]["polygon"][0] == [46, 49], "the edit did not reach the bundle"
