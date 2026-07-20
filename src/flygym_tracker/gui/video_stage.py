"""ONE picture in this window, and every video operation happens in it.

THE REQUEST THIS FILE ANSWERS, verbatim: *"I want that all the operations on videos are done in the
window that appears on the main GUI."* Before this, four of them were not. Drawing vial positions,
replaying a recording, measuring the noise floor and learning the drum faces each launched a CHILD
PROCESS with its own OpenCV window -- a window the operator had to find, position and close, that
could not see the settings pane beside it, that reported what it had done by printing to a console
nobody was looking at, and that wanted the same exclusive camera this window already held.

They are MODES of this widget now. The picture, its caption and its controls change; the window
does not, the settings pane beside it stays live, and there is exactly one place to look.

=================================================================================================
THE MODES, and where the frames come from in each. This is the whole design:

    CAMERA     the preview box CameraSession fills            just looking
    DRAW       the same box, or a file job's                  the operator's clicks are the work
    JOB        the camera TAP, or a file job's                noise floor / learning the faces
    RUN        the box the running pipeline fills             an experiment, or a replay of one

Every one of those is a `LatestFrame` -- the same one-slot mailbox, drained by the same timer. That
is what makes this one widget rather than four: the stage does not know or care whether the frames
it is showing come from a camera thread, a file thread or a pipeline thread, so a mode change is a
change of which box to pull from plus which controls to show.

=================================================================================================
WHY A ONE-SLOT BOX AND NOT A SIGNAL PER FRAME, once more, because it applies to all four.

Qt does not coalesce queued signals. Measured on this machine: 300 frame-sized payloads emitted at
a GUI thread that was not running its loop left 300 of 300 undelivered after 0.5 s, holding the
memory of every one. These sessions are watched for minutes (drawing) to days (a run), and a stall
-- a dragged window, a screen lock, a virus scan, a modal dialog -- would queue thousands of
frames. Pulling from a box means a stalled GUI shows a STALE frame and drops the rest, which is the
only failure mode here that cannot end an experiment.

=================================================================================================
THE CAMERA IS NEVER TAKEN BY A MODE CHANGE.

Switching to DRAW or to a noise measurement does NOT open the camera: USB3 Vision is exclusive, and
an app that grabs the device because somebody clicked a tab is an app that blocks the rig. A
camera-backed job asks `CameraSession.attach_tap`, which refuses unless the camera is already
streaming, and the mode says what to do about it instead of opening anything.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget,
                               QVBoxLayout, QWidget)

from flygym_tracker.gui.preview import PULL_INTERVAL_MS, PreviewWidget
from flygym_tracker.gui.video_jobs import FaceLearnJob, FileJobController, NoiseJob, PassiveJob

CAMERA = "camera"
DRAW = "draw"
JOB = "job"
RUN = "run"


def _bar() -> tuple:
    """A control strip: the widget and its layout. For strips of one or two things.

    The BUTTON strips use `_flow_bar` instead -- see `flow_layout` for the measurement: a
    QHBoxLayout's minimum width is the SUM of its children, and the drawing strip alone reported
    1082 px, forcing the window 246 px wider than the rig laptop's entire desktop.
    """
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    return widget, layout


def _flow_bar() -> tuple:
    """A strip of buttons that wraps onto a second line rather than widening the window."""
    from flygym_tracker.gui.flow_layout import flow_strip

    return flow_strip()


def _button(text: str, tip: str = "", role: str = "ghost") -> QPushButton:
    button = QPushButton(text)
    button.setProperty("role", role)
    if tip:
        button.setToolTip(tip)
    return button


class VideoStage(QWidget):
    """The picture, its caption, and whatever controls the current video operation needs."""

    #: The mode changed. The window uses it to enable/disable the things that need the picture.
    mode_changed = Signal(str)
    #: A video operation ended: ``(kind, payload)``. `kind` is "draw" | "noise" | "faces", and the
    #: payload always carries a `message` the window can put on screen verbatim.
    job_finished = Signal(str, dict)
    #: The operator asked to leave a mode that was not finished.
    cancelled = Signal(str)

    def __init__(self, session, run=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.session = session
        self.run = run
        #: Frames from a VIDEO FILE, for the jobs that read one. Owned here because these are the
        #: operations that are done IN the picture; the camera's box and the run's box belong to
        #: the objects that own those threads.
        self.files = FileJobController(self)
        self.files.progress.connect(self._on_job_progress)
        self.files.finished.connect(self._on_file_job_finished)
        self.files.failed.connect(self._on_file_job_failed)

        self._mode = CAMERA
        self._box = getattr(session, "latest", None)
        self._draw = None                     # the live VialDrawSession, in DRAW
        self._job_kind = ""                   # "noise" | "faces", in JOB
        self._job_note = ""
        self._build()
        self._show_mode(CAMERA)

        self._timer = QTimer(self)
        self._timer.setInterval(PULL_INTERVAL_MS)
        # A bound method of a GUI-thread object, never a lambda -- see `camera_session`.
        self._timer.timeout.connect(self._pull)
        self._timer.start()

    # -- construction ---------------------------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.view = PreviewWidget()
        self.view.clicked.connect(self._on_click)
        self.view.key_pressed.connect(self._on_key)
        layout.addWidget(self.view, 1)

        self.caption = QLabel("Camera not open - nothing is being read")
        self.caption.setProperty("role", "note")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.caption)

        self.bars = QStackedWidget()
        self.bars.addWidget(self._build_idle_bar())      # index 0 -- CAMERA and RUN
        self.bars.addWidget(self._build_draw_bar())      # index 1 -- DRAW
        self.bars.addWidget(self._build_job_bar())       # index 2 -- JOB
        layout.addWidget(self.bars)

    def _build_idle_bar(self) -> QWidget:
        widget, layout = _bar()
        self.idle_label = QLabel("")
        self.idle_label.setProperty("role", "note")
        layout.addWidget(self.idle_label, 1)
        return widget

    def _build_draw_bar(self) -> QWidget:
        """The keymap as BUTTONS. The keys still work and are still the same keys -- this is for
        an operator who has not learnt them, which on a shared rig is most of them."""
        widget, layout = _flow_bar()
        self.draw_freeze_button = _button(
            "Hold picture", "The drum turns; clicking a moving tube is hopeless. (SPACE)")
        self.draw_finish_vial_button = _button(
            "Vial done", "Store this polygon and start the next vial. (ENTER)", role="primary")
        self.draw_undo_point_button = _button("Undo point", "Remove the last corner. (BACKSPACE)")
        self.draw_undo_vial_button = _button(
            "Re-open last vial", "Re-open the previous vial for editing. (u)")
        self.draw_clear_button = _button("Clear vial", "Throw away this vial and start it. (c)")
        self.draw_done_button = _button(
            "Save and finish", "Keep the vials drawn so far and save the bundle. (q)")
        self.draw_cancel_button = _button(
            "Cancel", "Leave without writing anything.", role="danger")
        for button, slot in (
            (self.draw_freeze_button, self._draw_freeze),
            (self.draw_finish_vial_button, self._draw_finish_vial),
            (self.draw_undo_point_button, self._draw_undo_point),
            (self.draw_undo_vial_button, self._draw_undo_vial),
            (self.draw_clear_button, self._draw_clear),
            (self.draw_done_button, self._draw_done),
            (self.draw_cancel_button, self._draw_cancel),
        ):
            button.clicked.connect(slot)
            layout.addWidget(button)
        return widget

    def _build_job_bar(self) -> QWidget:
        widget, layout = _bar()
        self.job_label = QLabel("")
        self.job_label.setProperty("role", "note")
        layout.addWidget(self.job_label, 1)
        self.job_stop_button = _button("Stop", "End this measurement now.", role="danger")
        self.job_stop_button.clicked.connect(self.stop_job)
        layout.addWidget(self.job_stop_button)
        return widget

    # -- mode -----------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def draw_session(self):
        return self._draw

    def _show_mode(self, mode: str) -> None:
        self._mode = mode
        self.bars.setCurrentIndex({CAMERA: 0, RUN: 0, DRAW: 1, JOB: 2}.get(mode, 0))
        # ONLY DRAW TAKES THE KEYBOARD. A view that always held focus would swallow keystrokes the
        # settings pane is entitled to -- and this window goes out of its way to keep initial focus
        # off anything that edits a camera setting.
        self.view.set_interactive(mode == DRAW)
        self.view.set_overlay(self._draw.overlay if (mode == DRAW and self._draw) else None)
        self.mode_changed.emit(mode)

    def show_camera(self) -> None:
        """Back to just looking at the camera. Does NOT open or close it."""
        self._draw = None
        self._job_kind = ""
        self._box = getattr(self.session, "latest", None)
        self.view.placeholder = "No picture - the camera is not open"
        self._show_mode(CAMERA)

    def show_run(self) -> None:
        """Watch the running pipeline -- an experiment, or a replay of a recording.

        THIS IS WHY A REPLAY NO LONGER OPENS A MONITOR WINDOW. A replay is the same pipeline over a
        file, so it fills the same frame box, so watching it is the same view. There was never a
        reason for the recorded case to have a different surface than the live one, and having two
        meant the overlay an operator trusts was drawn by code they only see on one of the paths.
        """
        self._draw = None
        self._job_kind = ""
        self._box = getattr(self.run, "latest", None) if self.run is not None else None
        self.view.placeholder = "Waiting for the first frame of the run"
        self._show_mode(RUN)

    # -- drawing vial positions -------------------------------------------------------------------
    def begin_draw(self, *, out_dir: str, n_vials: int = 16, faces=("A", "B"),
                   video: Optional[str] = None) -> bool:
        """Draw vial positions, here, on the live camera or on a recording.

        Returns False (with a spoken caption) when there is nothing to draw ON: no video was given
        and the camera is not open. It does NOT open the camera -- see the module docstring.
        """
        from flygym_tracker.gui.vial_draw import VialDrawSession

        if video is None and not getattr(self.session, "is_open", False):
            self.caption.setText(
                "Nothing to draw on: open the camera first (the button in the bar at the top), or "
                "pick a recording to draw on. This window will not take the camera by itself - "
                "USB3 Vision allows one holder at a time.")
            return False
        label = ("FILE  %s  (recorded - not the camera)" % _basename(video) if video
                 else "CAMERA (live)")
        self._draw = VialDrawSession(n_vials=n_vials, face=list(faces)[0], source_label=label,
                                     out_dir=out_dir, faces=faces, parent=self)
        self._draw.changed.connect(self._on_draw_changed)
        self._draw.finished.connect(self._on_draw_finished)
        if video is not None:
            if not self._start_file_job(video, PassiveJob()):
                self._draw = None
                return False
            self._box = self.files.latest
        else:
            self._box = getattr(self.session, "latest", None)
        # THE PICTURE ALREADY ON SCREEN IS THE FIRST FRAME OF THE SESSION. Without this the
        # session waits for the NEXT frame before it has anything to build the illumination mask
        # from -- so an operator who opens the camera, sees the rig, presses the button and starts
        # clicking has a session that can only be saved if more frames happen to arrive. On a
        # recording that has reached its end, or a camera that stalls, none do, and several minutes
        # of clicking end in "no frame was ever received".
        if self.view._array is not None:
            self._draw.on_frame(self.view._array)
        self.view.placeholder = "Waiting for the first frame to draw on"
        self._show_mode(DRAW)
        self._on_draw_changed()
        return True

    def _on_draw_changed(self) -> None:
        if self._draw is None:
            return
        self.view.update()
        self._update_caption()
        self.draw_freeze_button.setText(
            "Release picture" if self._draw.state.frozen else "Hold picture")

    def _on_draw_finished(self, payload: dict) -> None:
        self.files.stop()
        self._draw = None
        self.show_camera()
        self.caption.setText(payload.get("message", ""))
        self.job_finished.emit("draw", payload)

    def _draw_freeze(self) -> None:
        if self._draw:
            self._draw.toggle_freeze()

    def _draw_finish_vial(self) -> None:
        if self._draw:
            self._draw.finish_vial()

    def _draw_undo_point(self) -> None:
        if self._draw:
            self._draw.undo_vertex()

    def _draw_undo_vial(self) -> None:
        if self._draw:
            self._draw.undo_vial()

    def _draw_clear(self) -> None:
        if self._draw:
            self._draw.clear_vial()

    def _draw_done(self) -> None:
        if self._draw:
            self._draw.finish()

    def _draw_cancel(self) -> None:
        if self._draw:
            self._draw.cancel()

    def _on_click(self, x: float, y: float) -> None:
        if self._mode == DRAW and self._draw is not None:
            self._draw.on_click(x, y)

    def _on_key(self, name: str) -> None:
        if self._mode == DRAW and self._draw is not None:
            self._draw.on_key(name)

    # -- measurements that consume frames ----------------------------------------------------------
    def begin_noise(self, illum_mask, *, n_frames: int = 100, k: float = 5.0,
                    video: Optional[str] = None) -> bool:
        """Measure the static-rig noise floor, in the window, with the picture visible.

        SEEING THE RIG WHILE IT IS MEASURED IS THE POINT of doing this here. The measurement is
        only valid on a STATIONARY rig, and the single most likely way to get a wrong answer is to
        measure while something is moving -- a hand in frame, the drum still settling. In a child
        process the operator saw a number appear; here they watch the thing being measured.
        """
        return self._begin_job("noise", NoiseJob(illum_mask, n_frames=n_frames, k=k), video)

    def begin_face_learning(self, *, n_faces: int = 2, face_names=("A", "B"),
                            video: Optional[str] = None) -> bool:
        """Learn one marker template per drum face while the drum turns."""
        return self._begin_job("faces", FaceLearnJob(n_faces=n_faces, face_names=face_names),
                               video)

    def _begin_job(self, kind: str, job, video: Optional[str]) -> bool:
        if video is not None:
            if not self._start_file_job(video, job):
                return False
            self._box = self.files.latest
        else:
            if not getattr(self.session, "is_open", False):
                self.caption.setText(
                    "This measurement needs frames: open the camera first, or pick a recording to "
                    "measure. This window will not take the camera by itself.")
                return False
            if not self.session.attach_tap(job):
                self.caption.setText("The camera is already busy with another measurement.")
                return False
            self._box = getattr(self.session, "latest", None)
        self._job_kind = kind
        self._job_note = ""
        self.view.placeholder = "Waiting for the first frame"
        self._show_mode(JOB)
        self._update_caption()
        return True

    def _start_file_job(self, video: str, job) -> bool:
        from flygym_tracker.frame_source import VideoFileSource

        if not self.files.start(lambda: VideoFileSource(video), job):
            self.caption.setText("Another recording is already being read in this window.")
            return False
        return True

    def stop_job(self) -> None:
        """End the current measurement now, keeping whatever it has already accumulated.

        A stop is not a cancel: a noise floor measured over 60 frames instead of 100 is a real
        measurement of 60 frames, and it reports the count it actually used.
        """
        if self.files.is_running:
            self.files.stop()
            return
        job = getattr(self.session, "tap", None)
        if job is None:
            return
        self.session.detach_tap()
        self._finish_camera_job(job)

    def _finish_camera_job(self, job) -> None:
        """A camera-backed job has ended. Its result is computed HERE, on the GUI thread, from the
        job object -- the accumulators are plain counters and the camera thread has let go of it."""
        kind = self._job_kind or "job"
        try:
            payload = dict(job.result() or {})
            payload.setdefault("message", _job_message(kind, payload))
        except Exception as exc:
            payload = {"message": "%s could not be completed: %s" % (kind, exc), "failed": True}
        self.show_camera()
        self.caption.setText(payload.get("message", ""))
        self.job_finished.emit(kind, payload)

    def _on_job_progress(self, snapshot: dict) -> None:
        self._job_note = _job_progress_line(self._job_kind, snapshot)
        self._update_caption()

    def _on_file_job_finished(self, result: dict) -> None:
        kind = self._job_kind or "job"
        payload = dict(result or {})
        if self._mode == DRAW:
            # The recording ran out while somebody was still clicking. The last frame stays on
            # screen and the session carries on -- losing a half-drawn face because a file ended
            # would be the cv2 selector's "end of video" behaviour, and it was right there too.
            self.caption.setText("End of the recording - the last frame is held. "
                                 "Carry on drawing, or save and finish.")
            return
        payload.setdefault("message", _job_message(kind, payload))
        self.show_camera()
        self.caption.setText(payload.get("message", ""))
        self.job_finished.emit(kind, payload)

    def _on_file_job_failed(self, message: str) -> None:
        kind = self._job_kind or "job"
        if self._mode == DRAW:
            self.caption.setText("That recording could not be read: %s" % message)
            return
        self.show_camera()
        self.caption.setText("%s failed: %s" % (kind, message))
        self.job_finished.emit(kind, {"message": message, "failed": True})

    # -- frames ------------------------------------------------------------------------------------
    def _pull(self) -> None:
        box = self._box
        if box is not None:
            frame = box.take()
            if frame is not None:
                # THE DRAWING SESSION IS TOLD FIRST. It keeps the frame the calibration will be
                # built from, and it declines the update while the picture is held -- so what is
                # measured for the illumination mask is the image the polygons were drawn on.
                if self._mode == DRAW and self._draw is not None:
                    self._draw.on_frame(frame)
                    if not self._draw.state.frozen:
                        self.view.set_frame(frame)
                else:
                    self.view.set_frame(frame)
        if self._mode == JOB:
            self._poll_camera_job()
        self._update_caption()

    def _poll_camera_job(self) -> None:
        """A camera-backed job reports through its own counters; nothing signals per frame."""
        job = getattr(self.session, "tap", None)
        if job is None:
            return
        self._job_note = _job_progress_line(self._job_kind, job.snapshot() or {})
        if job.done:
            self.session.detach_tap()
            self._finish_camera_job(job)

    def _update_caption(self) -> None:
        if self._mode == DRAW and self._draw is not None:
            self.caption.setText("%s   -   %s" % (self._draw.state.source_label,
                                                  self._draw.status()))
            return
        if self._mode == JOB:
            self.job_label.setText(self._job_note or "starting...")
            self.caption.setText(_JOB_CAPTIONS.get(self._job_kind, ""))
            return
        if self._mode == RUN:
            self.caption.setText("The run's own frames, as the pipeline sees them")
            return
        self.caption.setText(_camera_caption(self.session, self.view.frame_size))

    def shutdown(self) -> None:
        """Stop anything reading frames, before the window goes. Called from `closeEvent`."""
        self._timer.stop()
        try:
            self.session.detach_tap()
        except Exception:
            pass
        self.files.shutdown()


# =================================================================================================
# Captions -- pure, so what the operator is told is testable without a widget
# =================================================================================================
_JOB_CAPTIONS = {
    "noise": ("Measuring the noise floor. The rig must be STATIONARY for this to mean anything - "
              "if anything is moving in the picture, stop and start again."),
    "faces": ("Learning the drum faces. Turn the drum; each face is learned the first time it is "
              "shown and held still."),
}


def _basename(path) -> str:
    import os

    return os.path.basename(str(path)) if path else ""


def _job_progress_line(kind: str, snapshot: Dict[str, Any]) -> str:
    """One line of progress for a running measurement. Every figure is COUNTED by the job."""
    if kind == "faces":
        return str(snapshot.get("status") or "")
    frames = int(snapshot.get("frames") or 0)
    target = int(snapshot.get("n_target") or 0)
    if target:
        return "%d of %d frames   -   %d usable pair(s)" % (
            frames, target, int(snapshot.get("pairs") or 0))
    return "%d frames" % frames


def _job_message(kind: str, payload: Dict[str, Any]) -> str:
    """What a finished measurement says. It NAMES what it measured over, never just the answer.

    A suggested threshold with no frame count behind it is a number an operator cannot judge --
    and this one seeds every activity reading the rig takes afterwards.
    """
    if kind == "noise":
        try:
            return ("noise floor over %d frame(s), %d pair(s):  pixel threshold %.3f, "
                    "rotation enter %.3f / exit %.3f"
                    % (int(payload.get("n_frames") or 0), int(payload.get("n_pairs") or 0),
                       float(payload["suggested_pixel_threshold"]),
                       float(payload["suggested_enter_threshold"]),
                       float(payload["suggested_exit_threshold"])))
        except (KeyError, TypeError, ValueError):
            return "the noise measurement produced no usable result"
    if kind == "faces":
        learned = payload.get("learned") or []
        if payload.get("complete"):
            return "learned the marker band of face(s) %s - this bundle can identify faces now" \
                   % ", ".join(learned)
        return ("face learning stopped with %d of the faces learned (%s) - the bundle still "
                "cannot tell the faces apart, so everything would be recorded as one face"
                % (len(learned), ", ".join(learned) or "none"))
    return str(payload.get("message") or "")


def _camera_caption(session, frame_size) -> str:
    """The live preview's caption. Unchanged in meaning from `preview.PreviewPane`.

    "DELIVERED" IS STILL THE LOAD-BEARING WORD: counted from frames that arrived, and never the
    AcquisitionFrameRate setting or the camera's ResultingFrameRate read-back. On this rig's camera
    the frame-rate limiter is documented to disengage mid-stream while its registers still read
    back correct -- the two numbers the camera reports can both be right and both be wrong about
    what is happening. The counted one cannot.
    """
    if not getattr(session, "is_open", False):
        return "Camera not open - nothing is being read"
    shown, dropped = session.latest.stats
    bits = []
    width, height = frame_size
    if width and height:
        bits.append("%dx%d" % (width, height))
    delivered = session.measured_fps
    bits.append("camera delivering %.1f fps (measured)" % delivered if delivered > 0
                else "waiting for the first frame")
    total = shown + dropped
    if total:
        bits.append("showing %d of %d frames - %d not shown" % (shown, total, dropped))
    return "  -  ".join(bits)
