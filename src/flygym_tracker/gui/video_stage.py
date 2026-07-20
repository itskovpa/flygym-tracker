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
#: Marking WHERE THE MARKER BAND IS, so the detector stops inferring it every frame. See
#: `band_select` for the measurement that motivates it.
BAND = "band"


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
        # A CAMERA-BACKED JOB CAN DIE ON THE CAMERA THREAD, and until these were connected nothing
        # in the window found out: `camera_worker` detaches a raising tap and says so, into
        # nothing, leaving this widget showing a measurement that had already stopped. Bound
        # methods rather than lambdas -- a lambda's slot runs on the SENDER's thread, which here
        # is the one driving the SDK.
        for signal, slot in (("tap_failed", self._on_tap_failed),
                             ("tap_finished", self._on_tap_finished)):
            source = getattr(session, signal, None)
            if source is not None:
                source.connect(slot)

        self._mode = CAMERA
        self._box = getattr(session, "latest", None)
        self._draw = None                     # the live VialDrawSession, in DRAW
        self._band = None                     # the live BandSelectSession, in BAND
        self._job = None                      # the running FrameJob, in JOB -- see _poll_camera_job
        self._job_kind = ""                   # "noise" | "faces", in JOB
        self._job_note = ""
        self._tap_error = ""                  # why a camera-backed job ended, if it ended badly
        #: True once the RUN has actually delivered a frame. Until then the picture is the
        #: preview's last one and the caption must not claim otherwise -- see `_update_caption`.
        self._run_frames = False
        #: What the last video job produced, kept on screen until something else happens.
        #:
        #: BECAUSE THE CAPTION IS REWRITTEN EVERY 50 ms. `_pull` refreshes it from the current
        #: mode, so a result written straight into the label by a finishing job survived for one
        #: pull and was then replaced by the live camera line. That silently threw away the ONLY
        #: place the noise floor's suggested thresholds and the face-learning outcome are reported
        #: -- a measurement that appears to produce nothing at all. Same bug as `save_settings`
        #: calling `refresh_titles` after `set_status`; this is the version of it that eats a
        #: measurement rather than a confirmation.
        self._notice = ""
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
        self.view.pressed.connect(self._on_press)
        self.view.dragged.connect(self._on_drag)
        self.view.released.connect(self._on_release)
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
        self.bars.addWidget(self._build_band_bar())      # index 3 -- BAND
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
        self.draw_restart_button = _button(
            "Start over", "Throw away EVERY vial and draw all of them from nothing.")
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
            (self.draw_restart_button, self._draw_restart),
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

    def _build_band_bar(self) -> QWidget:
        widget, layout = _flow_bar()
        self.band_clear_button = _button("Clear", "Start the band again.")
        self.band_save_button = _button(
            "Save marker band", "Store these rows in the vial-positions folder, so every run and "
            "every face-learning session searches exactly here.", role="primary")
        self.band_cancel_button = _button("Cancel", "Leave the band as it was.", role="danger")
        for button, slot in ((self.band_clear_button, self._band_clear),
                             (self.band_save_button, self._band_save),
                             (self.band_cancel_button, self._band_cancel)):
            button.clicked.connect(slot)
            layout.addWidget(button)
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
        self.bars.setCurrentIndex({CAMERA: 0, RUN: 0, DRAW: 1, JOB: 2, BAND: 3}.get(mode, 0))
        # ONLY THE DRAWING MODES TAKE THE MOUSE. A view that always held focus would swallow
        # keystrokes the settings pane is entitled to -- and this window goes out of its way to
        # keep initial focus off anything that edits a camera setting.
        self.view.set_interactive(mode in (DRAW, BAND))
        overlay = None
        if mode == DRAW and self._draw is not None:
            overlay = self._draw.overlay
        elif mode == BAND and self._band is not None:
            overlay = self._band.overlay
        self.view.set_overlay(overlay)
        self.mode_changed.emit(mode)

    def show_camera(self) -> None:
        """Back to just looking at the camera. Does NOT open or close it.

        EVERY WAY OUT OF A JOB COMES THROUGH HERE, so this is where the job reference is dropped.
        Leaving a finished job attached would let the next poll collect its result a second time.
        """
        self._draw = None
        self._band = None
        self._job = None
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
        self._band = None
        self._job_kind = ""
        self._box = getattr(self.run, "latest", None) if self.run is not None else None
        self._run_frames = False
        self.view.placeholder = "Waiting for the first frame of the run"
        self._show_mode(RUN)

    # -- drawing vial positions -------------------------------------------------------------------
    def begin_draw(self, *, out_dir: str, n_vials: int = 16, faces=("A", "B"),
                   video: Optional[str] = None, polygons=None) -> bool:
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
        if polygons:
            # SAVED POSITIONS ARE SHOWN, NOT HIDDEN BEHIND A YES/NO. Reuse used to be all-or-
            # nothing -- keep them exactly, or re-click all sixteen -- so a bundle that was 15/16
            # right cost a whole clicking session, and nothing on screen said it was 15/16 right.
            self._draw.load(polygons)
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
        self._notice = payload.get("message", "")
        self._update_caption()
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

    def _draw_restart(self) -> None:
        if self._draw:
            self._draw.start_over()

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

    # -- marking where the marker band is ----------------------------------------------------------
    def begin_band(self, *, out_dir: str) -> bool:
        """Mark the marker band's rows on the live picture. Returns False if there is no picture.

        Needs a frame for the same reason drawing vials does: the rows are IMAGE rows, and a band
        drawn against no picture is a number the operator never actually saw on the rig.
        """
        from flygym_tracker.gui.band_select import BandSelectSession

        if self.view._array is None:
            self.caption.setText(
                "Nothing to mark the band on: open the camera first, or replay a recording. "
                "This window will not take the camera by itself.")
            return False
        width, height = self.view.frame_size
        self._band = BandSelectSession(out_dir=out_dir, frame_height=height, parent=self)
        self._band.changed.connect(self._on_band_changed)
        self._band.finished.connect(self._on_band_finished)
        self._band.on_frame(self.view._array)
        self._notice = ""
        self._show_mode(BAND)
        self._on_band_changed()
        return True

    def _on_band_changed(self) -> None:
        if self._band is None:
            return
        self.view.update()
        self._update_caption()
        self.band_save_button.setEnabled(self._band.has_band)

    def _on_band_finished(self, payload: dict) -> None:
        self._band = None
        self.show_camera()
        self._notice = payload.get("message", "")
        self._update_caption()
        self.job_finished.emit("band", payload)

    def _band_clear(self) -> None:
        if self._band:
            self._band.clear()

    def _band_save(self) -> None:
        if self._band:
            self._band.save()

    def _band_cancel(self) -> None:
        if self._band:
            self._band.cancel()

    def _on_press(self, x: float, y: float) -> None:
        if self._mode == BAND and self._band is not None:
            self._band.on_press(x, y)
        elif self._mode == DRAW and self._draw is not None:
            self._draw.on_press(x, y)

    def _on_drag(self, x: float, y: float) -> None:
        if self._mode == BAND and self._band is not None:
            self._band.on_drag(x, y)
        elif self._mode == DRAW and self._draw is not None:
            self._draw.on_drag(x, y)

    def _on_release(self, x: float, y: float) -> None:
        if self._mode == BAND and self._band is not None:
            self._band.on_release(x, y)
        elif self._mode == DRAW and self._draw is not None:
            self._draw.on_release(x, y)

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
                            video: Optional[str] = None, band_rows=None) -> bool:
        """Learn one marker template per drum face while the drum turns.

        `band_rows` is the operator's drawn marker band, if the bundle carries one. Learning MUST
        use the same window every later identification will use: profiles are only comparable to
        profiles extracted the same way.
        """
        return self._begin_job("faces", FaceLearnJob(n_faces=n_faces, face_names=face_names,
                                                     band_rows=band_rows), video)

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
        self._job = job
        self._job_kind = kind
        self._job_note = ""
        self._tap_error = ""
        self._notice = ""                     # the previous result is history now
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

        STOP ALWAYS GETS THE OPERATOR OUT. THE BUG THIS FIXES, reported from the rig: "I select
        Learn Drum faces, the algorithm starts and never stops, the button stop does not work."
        This method used to `return` silently when it could not find a job to stop -- and the one
        situation that produces exactly that is a job which RAISED: `camera_worker` detaches a
        raising tap (so it cannot throw once per frame for the rest of a session) and reports it,
        but nothing was listening, so the stage sat in JOB mode with the last progress line still
        on screen, looking like a measurement that was still running, and the only control offered
        did nothing at all.

        Face learning is where that is worst, because it is ALSO the job that legitimately runs
        forever: it ends when the drum has shown every face, and a drum that is not turning never
        ends it. So "it never stops" is normal and "stop does nothing" was the whole failure.

        There is now no path out of this method that leaves the stage in JOB mode.
        """
        if self.files.is_running:
            self.files.stop()
            return
        job = getattr(self.session, "tap", None)
        if job is not None:
            self.session.detach_tap()
            self._finish_camera_job(job)
            return
        # NOTHING TO STOP -- and the operator still asked to leave. Whatever became of the job
        # (it raised and detached itself, or it was never attached), sitting here is not an option.
        self.show_camera()
        self.caption.setText(
            self._job_note or "that measurement had already stopped - nothing was left running")

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
        self._notice = payload.get("message", "")
        self._update_caption()
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
        self._notice = payload.get("message", "")
        self._update_caption()
        self.job_finished.emit(kind, payload)

    def _on_file_job_failed(self, message: str) -> None:
        kind = self._job_kind or "job"
        if self._mode == DRAW:
            self.caption.setText("That recording could not be read: %s" % message)
            return
        self.show_camera()
        self._notice = "%s failed: %s" % (kind, message)
        self._update_caption()
        self.job_finished.emit(kind, {"message": message, "failed": True})

    # -- frames ------------------------------------------------------------------------------------
    def _pull(self) -> None:
        box = self._box
        if box is not None:
            frame = box.take()
            if frame is not None:
                if self._mode == RUN:
                    self._run_frames = True
                # THE DRAWING SESSION IS TOLD FIRST. It keeps the frame the calibration will be
                # built from, and it declines the update while the picture is held -- so what is
                # measured for the illumination mask is the image the polygons were drawn on.
                if self._mode == DRAW and self._draw is not None:
                    self._draw.on_frame(frame)
                    if not self._draw.state.frozen:
                        self.view.set_frame(frame)
                else:
                    if self._mode == BAND and self._band is not None:
                        # The band preview re-checks the strips against the CURRENT frame, so the
                        # "N strips found" readout tracks the live rig rather than one still.
                        self._band.on_frame(frame)
                    self.view.set_frame(frame)
        if self._mode == JOB:
            self._poll_camera_job()
        self._update_caption()

    def _poll_camera_job(self) -> None:
        """A camera-backed job reports through its own counters; nothing signals per frame.

        THE JOB IS HELD HERE, NOT READ BACK OFF THE SESSION. `camera_worker` clears its tap in BOTH
        endings -- when the job says it is done, and when the job raises -- so "the tap is gone"
        does not distinguish success from failure, and an earlier version of this method read the
        tap and treated a missing one as nothing to do at all. That is the state a raising job
        leaves behind, which is how the stage came to sit in JOB mode with a dead Stop button.
        """
        job = self._job
        if job is None:
            self.show_camera()
            self.caption.setText(self._tap_error
                                 or "that measurement stopped before it finished")
            return
        self._job_note = _job_progress_line(self._job_kind, job.snapshot() or {})
        if job.done:
            self.session.detach_tap()
            self._finish_camera_job(job)

    def _on_tap_failed(self, message: str) -> None:
        """The job raised on the camera thread; `camera_worker` already detached it."""
        self._tap_error = "that measurement stopped: %s" % message
        if self._mode != JOB:
            return
        kind = self._job_kind or "job"        # CAPTURED FIRST -- `show_camera` clears it
        self.show_camera()
        self._notice = self._tap_error
        self._update_caption()
        self.job_finished.emit(kind, {"message": self._tap_error, "failed": True})

    def _on_tap_finished(self) -> None:
        """The job reported `done` from the camera thread and was detached there. Nothing to do
        here: `_poll_camera_job` collects the result on the next pull, which is where the result
        is computed on the GUI thread rather than on the one driving the SDK."""
        return

    def _update_caption(self) -> None:
        if self._mode == DRAW and self._draw is not None:
            self.caption.setText("%s   -   %s" % (self._draw.state.source_label,
                                                  self._draw.status()))
            return
        if self._mode == BAND and self._band is not None:
            self.caption.setText(self._band.status())
            return
        if self._mode == JOB:
            self.job_label.setText(self._job_note or "starting...")
            self.caption.setText(_JOB_CAPTIONS.get(self._job_kind, ""))
            return
        if self._mode == RUN:
            # UNTIL THE RUN'S FIRST FRAME ARRIVES, WHAT IS ON SCREEN IS THE PREVIEW'S LAST ONE.
            # Leaving it captioned as the run's frames would be this program telling the operator
            # that a still from before the experiment started is the experiment -- the same class
            # of claim as calling a recording live, and the handover takes a second or two during
            # which the picture does not visibly change.
            self.caption.setText("The run's own frames, as the pipeline sees them"
                                 if self._run_frames else
                                 "handing the camera to the run - this is still the last preview "
                                 "frame, not the run")
            return
        live = _camera_caption(self.session, self.view.frame_size)
        # The notice FIRST: it is the result of something the operator asked for, and the
        # live line is a status they can read any time.
        self.caption.setText("%s   -   %s" % (self._notice, live) if self._notice else live)

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
        tail = ("the bundle still cannot tell the faces apart, so everything would be recorded "
                "as one face")
        # NAME THE ACTUAL OBSTACLE. Measured on the rig: the band was unreadable in 615 of 871
        # frames because the exposure was 10 ms (this band needs ~40 ms), and the only thing the
        # operator was told was that it was waiting for the drum -- so they kept turning it.
        unreadable = int(payload.get("unreadable") or 0)
        frames = int(payload.get("frames") or 0)
        if unreadable and frames and unreadable * 2 >= frames:
            return ("face learning stopped with %d face(s) learned (%s): THE MARKER BAND COULD NOT "
                    "BE READ in %d of %d frames - the picture is too dark for it. Raise the "
                    "exposure or the illumination until the two lit strips are clearly visible, "
                    "then run this again. %s"
                    % (len(learned), ", ".join(learned) or "none", unreadable, frames, tail))
        return ("face learning stopped with %d of the faces learned (%s) - %s"
                % (len(learned), ", ".join(learned) or "none", tail))
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
