"""One window, five bands, no navigation.

THE ORDER OF THE BANDS IS THE ORDER OF THE QUESTIONS an operator arrives with, top to bottom:

    1  CameraStatusBar   whose camera is it right now
    2  SessionBar        which experiment - config file, vial positions, results folder
    3  splitter          what am I imposing (left) / what do I see (right)
    4  ReadinessStrip    is anything going to stop this run
    5  ChangeBanner      what is unsaved, and the way to save it        (inside SettingsView)

There is no sidebar because there is nothing to navigate between: Stage 1 is settings and a camera,
and a nav rail for two things is ceremony. The settings and the preview are ADJACENT rather than on
separate pages because exposure and gain are tuned by looking at the picture -- a design that makes
you switch tabs to see the effect of the knob you are turning is the cv2 panel's problem again.

THE INITIAL FOCUS IS SET DELIBERATELY. Measured: Qt auto-focuses the first focusable widget of a
shown window, which here would be a settings spinbox. A stray keypress at 2 am would then edit a
camera setting before the operator has looked at anything. After `show()` the focus is moved to an
inert widget, and a test asserts that no spinbox holds focus.

TWO THINGS CONFIRM, AND ONLY TWO. Stopping another program (irreversible, and it may be the
operator's own work) and closing with unsaved changes or an open camera. Ordinary edits do not:
they are reversible, they are logged, and a surface that asks about everything trains an operator
to click past the question that mattered. Returning a row to "camera default" deliberately does NOT
confirm -- it is the action a tuning loop repeats, and its warning is a line on the row instead.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMessageBox, QSplitter, QVBoxLayout,
                               QWidget)

from flygym_tracker import camera_lock, readiness
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.camera_lock_dialog import CameraLockDialog, qt_confirm
from flygym_tracker.gui.camera_session import CLOSED, STREAMING, CameraSession
from flygym_tracker.gui.camera_status import CameraStatusBar
from flygym_tracker.gui.readiness_strip import ReadinessStrip
from flygym_tracker.gui.run_controller import RunController
from flygym_tracker.gui.run_panel import RunPanel
from flygym_tracker.gui.session_bar import SessionBar
from flygym_tracker.gui.settings_view import SettingsView
from flygym_tracker.gui.run_controller import DONE, FAILED, IDLE
from flygym_tracker.gui.video_stage import RUN as STAGE_RUN
from flygym_tracker.gui.video_stage import VideoStage
from flygym_tracker.settings_controller import (SettingsController, camera_block_reason,
                                                is_start_only)
from flygym_tracker.settings_model import build_app_settings

#: How often the status bar's delivered-fps figure is refreshed. A label, not a measurement: the
#: number itself is counted in the worker from frames that arrived, and nothing is asked of the
#: camera to produce it.
STATUS_REFRESH_MS = 500


class MainWindow(QMainWindow):
    """Stage 1: settings and camera. The run view lands in this window later, not beside it."""

    def __init__(self, *, config, config_path, state, root, camera_factory,
                 confirm=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("FlyGym v2 - settings and camera")
        self.config = config
        self.state = state
        self.root = root
        self._confirm = confirm            # injected in tests; None means "use a real dialog"

        self.session = CameraSession(camera_factory, parent=self)
        #: The experiment, in this window, on its own thread. It is given a CALLABLE for the
        #: preview camera's state rather than a boolean: USB3 Vision is exclusive, and whether the
        #: preview holds the handle changes while this window is up.
        self.run = RunController(camera_is_open=lambda: self.session.is_open, parent=self)
        model = build_app_settings(config)
        self.controller = SettingsController(
            model,
            # ONE provider, matching `pipeline.setting_block_reason`'s signature exactly. Stage 2
            # swaps the pipeline's bound method in here and nothing else changes.
            block_reason=self._block_reason,
            on_change=self._on_setting_change,
            config_path=config_path,
            camera_open=lambda: self.session.is_open,
        )
        self._build()
        self._connect()
        self.refresh_readiness()

    # -- construction --------------------------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.status_bar = CameraStatusBar()
        layout.addWidget(self.status_bar)

        self.session_bar = SessionBar(self.state)
        self.session_bar.set_camera_identity(
            _cfg(self.config, "source.camera.serial"), _cfg(self.config, "source.camera.index"))
        layout.addWidget(self.session_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.settings_view = SettingsView(self.controller)
        splitter.addWidget(self.settings_view)
        # THE PICTURE IS THE STAGE EVERY VIDEO OPERATION IS PERFORMED ON. Drawing vial positions,
        # replaying a recording, measuring the noise floor and learning the drum faces used to be
        # four child processes with four OpenCV windows; they are modes of this one widget now.
        self.stage = VideoStage(self.session, self.run)
        #: The old name, kept because it is what the window's own code and its tests call the
        #: picture. It is the same object -- there is only one picture in this window.
        self.preview = self.stage
        splitter.addWidget(self.stage)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        # Explicit initial sizes, not just stretch factors: stretch alone let the settings pane
        # open narrower than its content, which pushed the value controls off the right edge (seen
        # by rendering this window offscreen). The preview still gets the larger half.
        splitter.setSizes([600, 640])
        splitter.setCollapsible(0, False)
        layout.addWidget(splitter, 1)

        self.readiness_strip = ReadinessStrip()
        layout.addWidget(self.readiness_strip)

        # THE RUN BAND. What used to be a line of text saying those four jobs "are still in
        # run.bat" is now the controls themselves. The run happens in this window, on a worker
        # thread, with the settings pane above it still live -- which is the whole point.
        self.run_panel = RunPanel()
        layout.addWidget(self.run_panel)

        self.setCentralWidget(central)
        self.resize(*self._fitted_size(1180, 820))

    def _fitted_size(self, want_w: int, want_h: int) -> Tuple[int, int]:
        """`(w, h)` shrunk to fit this desktop, never enlarged past what was asked for.

        REGRESSION THIS EXISTS FOR, and it is the THIRD time on this project. A hard-coded size
        is a guess about a screen the developer cannot see. The cv2 selector once drew its frame
        130 px below the screen edge (the lower vial row, unclickable), and the cv2 settings panel
        was then placed 66 px off the right edge even though it fitted. Here 820 + the title bar
        came to 873 on an 852 px work area, putting the filter box under the taskbar.

        `availableGeometry` is the work area -- the screen minus the taskbar -- and the frame
        margins are the title bar and borders Qt adds AROUND the size passed to `resize`. Both are
        subtracted, because a window that merely FITS can still be positioned badly; Qt centres a
        window that fits, so leaving room is what makes the placement land on screen too.
        """
        try:
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                return want_w, want_h
            area = screen.availableGeometry()
            margins = self.frameGeometry().size() - self.geometry().size()
            chrome_w = max(0, margins.width())
            chrome_h = max(0, margins.height()) or 40      # before first show, margins read 0
            return (max(640, min(want_w, area.width() - chrome_w - 16)),
                    max(480, min(want_h, area.height() - chrome_h - 16)))
        except Exception:
            # Sizing is cosmetic; never let it stop the app opening.
            return want_w, want_h

    def _connect(self) -> None:
        self.status_bar.open_requested.connect(self.open_camera)
        self.status_bar.close_requested.connect(self.session.close)
        self.status_bar.free_requested.connect(self.show_camera_lock)

        # Bound methods of GUI-thread objects, never lambdas -- see `camera_session`.
        self.session.state_changed.connect(self._on_camera_state)
        self.session.written.connect(self._on_written)
        self.session.values_read.connect(self._on_values_read)

        self.settings_view.changed.connect(self.refresh_readiness)
        self.settings_view.save_requested.connect(self.save_settings)
        self.filter_box = self.settings_view.filter_box

        self.run_panel.start_requested.connect(self.start_run)
        self.run_panel.stop_requested.connect(self.run.stop)
        self.run_panel.tool_requested.connect(self._on_tool)
        self.stage.job_finished.connect(self._on_job_finished)
        self.stage.mode_changed.connect(self._on_stage_mode)
        self.run.state_changed.connect(self._on_run_state)
        self.run.progress.connect(self.run_panel.set_progress)
        self.run.setting_applied.connect(self._on_run_setting_applied)
        self.session_bar.config_changed.connect(self._on_config_changed)
        self.session_bar.calib_changed.connect(self._on_calib_changed)
        self.session_bar.output_changed.connect(self._on_output_changed)
        self.readiness_strip.fix_requested.connect(self._on_fix)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    def take_initial_focus(self) -> None:
        """Move focus off the first spinbox, AFTER `show()`. See the module docstring."""
        self.filter_box.setFocus(Qt.FocusReason.OtherFocusReason)

    # -- camera --------------------------------------------------------------------------------
    def open_camera(self) -> None:
        """Open on request only. The app NEVER takes the camera by itself -- USB3 Vision is
        exclusive, and an app that grabs the camera at launch is an app that blocks the rig."""
        self.session.open()

    def _on_camera_state(self, state: str, detail: str) -> None:
        self.status_bar.set_state(state, detail, measured_fps=self.session.measured_fps)
        if state == STREAMING:
            # The limits are now this sensor's rather than the datasheet's, and they can differ
            # enough that a spinbox built from the fallbacks offers a value the SDK rejects.
            self.settings_view.rebuild_camera_rows(self.config, self.session.source)
        elif state == CLOSED:
            self.preview.view.clear()
            self.settings_view.rebuild_camera_rows(self.config, None)
        self.settings_view.refresh()
        self.refresh_readiness()

    def _on_values_read(self, values: dict) -> None:
        """A camera has reported what it is doing: rows armed blind are now checkable."""
        self.controller.confirm_against_camera(values or {})
        self.settings_view.rebuild_camera_rows(self.config, self.session.source)
        self.refresh_readiness()

    def _on_written(self, key: str, ok: bool, confirmed, message: str) -> None:
        """Show the value the SDK actually took, not the one that was asked for.

        This is the closed loop: `confirmed` is read back from the camera AFTER the write, so a
        clamped exposure or a snapped width appears on screen as the number in force. A failed
        write says so; it is never silently dropped.
        """
        row = self.settings_view.rows.get(key)
        if not ok:
            if row is not None:
                row._show_notice("the camera refused it: %s" % message)
            return
        if confirmed is not None:
            self.controller.model.set(key, confirmed)
        if row is not None:
            row.refresh()
        # A start-only write changes what the other camera nodes will accept.
        self.session.refresh_ranges()
        self.session.read_values()

    def _block_reason(self, key: str):
        """Why `key` cannot be changed right now. ONE answer, whatever is going on.

        WHILE A RUN IS GOING, Width and Height are refused (invariant 3) -- they are fixed at
        StartGrabbing time, so applying one would mean stopping and restarting acquisition under an
        experiment that may have been recording for days: a gap in the series PLUS a frame-diff
        baseline reset, i.e. two incomparable regimes in one file with nothing marking the seam.

        THE REFUSAL IS NOT ENFORCED HERE. `TrackerPipeline.apply_setting` asks its own
        `setting_block_reason` as the backstop, so a change that somehow got past this one is still
        refused by the pipeline. This is what makes the row LOOK dead, which is the other half:
        a control that only refuses when pressed is a control the operator presses.

        The pipeline's own state is deliberately NOT consulted across the thread boundary. It lives
        on the run thread beside an open SDK handle, and a GUI-thread attribute read of a running
        acquisition object is the kind of thing that works until it does not.
        """
        if self.run.is_running and is_start_only(key):
            return ("the run is using this - %s is fixed when acquisition starts, and changing it "
                    "would restart the stream mid-experiment. Stop the run to change it." % key)
        return camera_block_reason(lambda: self.session.source)(key)

    def _on_setting_change(self, key: str, value) -> bool:
        """The router. THREE destinations, in order of what is actually happening right now.

        1. A RUN IS GOING -> `TrackerPipeline.apply_setting`, via the run thread's queue. That is
           the path that both APPLIES the change and LOGS it as a `setting_change` event
           (invariant 4). It covers camera keys AND algorithm keys, which is the whole live-tuning
           request: `activity.pixel_threshold` and the rotation knobs are re-read per frame, so
           nothing needs restarting for them either.
        2. NO RUN, BUT THE PREVIEW CAMERA IS OPEN -> the camera, so tuning against the live picture
           does what it looks like it does.
        3. NEITHER -> nothing to apply it to. It is in the model, and it goes to the config file on
           save; the row says "takes effect at next start", which is true and is the same thing the
           cv2 panel says.
        """
        if self.run.is_running:
            return self.run.apply_setting(key, value)
        if not key.startswith("source.camera."):
            return True                     # a config edit; there is no run to apply it to
        if not self.session.is_open:
            return True                     # nothing to send it to; it is in the model for saving
        return self.session.write(key, value,
                                  block_reason=self.controller.block_reason(key))

    def show_camera_lock(self) -> None:
        """Name what holds the camera, and offer to stop it. Nothing is stopped without a yes.

        `camera_lock.prompt_and_release` is deliberately NOT called: it is terminal-bound, so under
        a GUI it would find no stdin, print into nowhere and stop nothing while appearing to ask.
        """
        holders = camera_lock.find_camera_holders()
        confirm = self._confirm if self._confirm is not None else qt_confirm(self)
        dialog = CameraLockDialog(holders, confirm=confirm, parent=self)
        dialog.exec()
        self.settings_view.set_status(dialog.summary())
        self.refresh_readiness()

    def _refresh_status(self) -> None:
        self.status_bar.set_state(self.session.state, self.session.detail,
                                  measured_fps=self.session.measured_fps)

    # -- session paths -------------------------------------------------------------------------
    def _on_config_changed(self, path: str) -> None:
        """Load a different config. Unsaved edits to the current one are confirmed away first."""
        if not path or path == self.controller.config_path:
            return
        if self.controller.changed() and not self._ask_discard(
                "Switching config files will lose %d unsaved change(s)."
                % len(self.controller.changed())):
            self.session_bar.config_combo.setCurrentText(self.controller.config_path or "")
            return
        from flygym_tracker.config import load_config

        try:
            config = load_config(path=path)
        except Exception as exc:
            QMessageBox.warning(self, "That config could not be loaded", str(exc))
            self.session_bar.config_combo.setCurrentText(self.controller.config_path or "")
            return
        self.config = config
        self.state["config_path"] = path
        gui_state.remember_config(self.state, path)
        gui_state.save_state(self.root, self.state)
        self.session_bar.set_recent(self.state["recent_configs"])
        self.session_bar.set_camera_identity(_cfg(config, "source.camera.serial"),
                                             _cfg(config, "source.camera.index"))
        self.controller.model = build_app_settings(config)
        self.controller.config_path = path
        # The rows are built from the model, so a new model means a new view. Rebuilding the whole
        # settings pane is cheap (ten rows) and avoids a partial-update bug class entirely.
        self._replace_settings_view()
        self.refresh_readiness()

    def _replace_settings_view(self) -> None:
        old = self.settings_view
        new = SettingsView(self.controller)
        new.changed.connect(self.refresh_readiness)
        new.save_requested.connect(self.save_settings)
        splitter = old.parentWidget()
        while splitter is not None and not isinstance(splitter, QSplitter):
            splitter = splitter.parentWidget()
        if splitter is not None:
            splitter.replaceWidget(splitter.indexOf(old), new)
        old.deleteLater()
        self.settings_view = new
        # The filter box belongs to the pane, so a replaced pane brings a new one -- and the
        # window's handle has to follow it, or `take_initial_focus` would focus a deleted widget.
        self.filter_box = new.filter_box
        if self.session.is_open:
            new.rebuild_camera_rows(self.config, self.session.source)

    def _on_calib_changed(self, path: str) -> None:
        self.state["calib_dir"] = path
        gui_state.save_state(self.root, self.state)
        self.refresh_readiness()

    def _on_output_changed(self, path: str) -> None:
        self.state["output_dir"] = path
        gui_state.save_state(self.root, self.state)
        self.refresh_readiness()

    def _on_filter(self, text: str) -> None:
        """Kept as the window's entry point; the filter itself now lives in the settings pane, so
        the box can be sticky at the top of the column it filters."""
        self.settings_view._on_filter(text)

    # -- the run --------------------------------------------------------------------------------
    def start_run(self) -> None:
        """Begin an experiment in this window.

        THE PREVIEW CAMERA IS CLOSED FIRST, AND THE OPERATOR IS TOLD (invariant 5). USB3 Vision is
        exclusive: the preview handle and the run's handle cannot both exist. Closing it silently
        would make the live picture vanish with no explanation at the exact moment attention is
        highest, so it is confirmed -- this is one of the few things worth a question, because it
        is about to take the camera for hours or days.
        """
        if self.session.is_open:
            if not self._ask("The run needs the camera to itself - USB3 Vision allows one holder "
                             "at a time. Close the preview and start the run?"):
                return
            self.session.close()
        plan = {
            # The MODEL, not `self.config`. See `_config_for_run`: the config object is what was
            # on disk when the app launched, and nothing an operator does on screen writes back
            # into it -- so a run built from it measures at values the operator changed away from.
            "config": self._config_for_run(),
            "calib_dir": self.state.get("calib_dir"),
            "output_dir": self.state.get("output_dir") or getattr(
                getattr(self.config, "output", None), "dir", None),
            "config_path": self.controller.config_path,
        }
        if not self.run.start(plan):
            self.run_panel.set_run_state(self.run.state, self.run.detail)
            return
        # THE EXPERIMENT IS WATCHED IN THIS WINDOW. The preview camera has just been handed over,
        # so without this the picture would sit on the last frame the preview saw -- a still of the
        # rig, indistinguishable from a live one, for however many days the run lasts.
        self.stage.show_run()
        # THE ROUTER NOW HAS A RUN TO ROUTE TO. From here `_on_setting_change` sends camera AND
        # algorithm keys into `TrackerPipeline.apply_setting`, which applies them AND logs them as
        # `setting_change` events (invariant 4).
        self.settings_view.refresh()

    def _config_for_run(self):
        """The config a run should actually be measured with: the values that are ON SCREEN.

        REGRESSION THIS EXISTS FOR, and it is the worst kind this project can produce -- days of
        data that are quietly wrong while the screen says otherwise. `build_settings` seeds each
        Setting by COPYING out of the config object, `SettingsModel.set` only assigns to the
        Setting, and `save_settings_to_yaml` rewrites the FILE. Nothing ever wrote back into the
        live config. So a run built from `self.config` used whatever was on disk when the app
        launched. Measured: set pixel threshold 12 -> 25, save (the banner reads "no changes", the
        file holds 25), start the run -- and the pipeline measured at 12.0 for the whole run while
        the row showed 25. `run_meta.json` recorded 12.0 too, so nothing downstream could catch it.

        Overlaying the model is the fix rather than reloading the file, because it is also right
        for values that are set but NOT yet saved: what the operator can see is what gets measured,
        which is the only rule that cannot surprise them. `to_overrides()` returns exactly the
        nested shape `load_config` merges, and camera rows at "camera default" are absent from it,
        so invariant 1 is untouched -- an unset row still sends nothing.
        """
        from flygym_tracker.config import load_config

        overrides = self.controller.model.to_overrides()
        if not overrides:
            return self.config
        try:
            return load_config(path=self.controller.config_path, overrides=overrides)
        except Exception:
            # A run measured with stale settings is worse than one that does not start, but a
            # crash here is worse still: fall back and say so on the readiness strip.
            return self.config

    def _on_stage_mode(self, mode: str) -> None:
        """A video job has the picture, so the buttons that would start a SECOND one go grey.

        Two jobs at once is not a thing that can half-work: they would be two readers of one
        exclusive camera, or two videos interleaving frames into one box -- which on screen looks
        exactly like a corrupted recording.
        """
        from flygym_tracker.gui.video_stage import CAMERA

        self.run_panel.set_stage_busy(mode != CAMERA)

    def _on_run_state(self, state: str, detail: str) -> None:
        self.run_panel.set_run_state(state, detail)
        # A finished run gives the picture back to the camera preview, so the next thing the
        # operator does is not done against the last frame of the last experiment.
        if state in (DONE, FAILED, IDLE) and self.stage.mode == STAGE_RUN:
            self.stage.show_camera()
        # Width/Height must LOOK dead while the stream is running, not merely refuse when pressed
        # (invariant 3). The refusal itself is the pipeline's `setting_block_reason`; this is the
        # refresh that puts it on screen the moment the run starts and takes it off when it ends.
        self.settings_view.refresh()
        self.refresh_readiness()

    def _on_run_setting_applied(self, key: str, applied: bool) -> None:
        """The closed loop for a mid-run change: the row says whether it REACHED the pipeline.

        Without this, "I typed it" and "the run took it" look identical on screen, and the only
        record that the change landed is a line in events.csv nobody is reading while the run is
        going.
        """
        row = self.settings_view.rows.get(key)
        if row is None:
            return
        if applied:
            row.flash_applied()
            row._show_notice("")
        else:
            row._show_notice("this run could not take that change - it is stored for the next start")

    def _on_tool(self, action: str) -> None:
        """Every video job, IN THIS WINDOW.

        WHAT THIS METHOD USED TO BE. Four `subprocess.Popen` calls into `python -m
        flygym_tracker.cli`, each opening its own OpenCV window in its own process. That was not
        laziness -- it was working around a real measurement: a `QApplication` makes this process
        PER_MONITOR_AWARE, and `live_vial_selector.screen_view_limit` is built on the process
        staying DPI-UNAWARE, because an AUTOSIZE cv2 window is laid out at the frame's own pixel
        size and there is no other way to know whether that fits the desktop.

        THE WORKAROUND IS GONE BECAUSE THE CONSTRAINT IS. A letterboxed Qt widget inside a layout
        is given a rectangle and fits the frame into it; it cannot run off the screen edge at any
        DPI, so nothing here needs to measure the desktop and nothing needs a second process to
        stay unaware of it. What is left is what the operator asked for: one window.
        """
        if action == "free_camera":
            self.show_camera_lock()
            return
        calib = self.state.get("calib_dir") or "calib_faces"
        if action == "draw_vials":
            self._begin_draw(calib)
        elif action == "noise":
            self._begin_noise(calib)
        elif action == "learn_faces":
            self._begin_face_learning()
        elif action == "replay":
            self._begin_replay(calib)

    def _pick_video(self, title: str) -> Optional[str]:
        video, _ = QFileDialog.getOpenFileName(
            self, title, self.state.get("last_video") or "",
            "Video files (*.avi *.mp4 *.mkv);;All files (*)")
        if video:
            self.state["last_video"] = video
            gui_state.save_state(self.root, self.state)
        return video or None

    def _begin_draw(self, calib: str) -> None:
        """Draw vial positions on the picture in this window.

        THE "REUSE WHAT IS SAVED?" QUESTION IS A DIALOG NOW, not a terminal prompt. It used to be
        `input()` in a child process, which under a GUI means a question asked into a console the
        operator may not even have open -- and `prompt_reuse` defaults an unanswerable stdin to
        YES, so pressing the button could silently do nothing visible at all.
        """
        from flygym_tracker.calibration import saved_selection

        saved = saved_selection(calib)
        if saved is not None and not self._ask_redraw(saved):
            return
        # `calibration.VIALS_PER_FACE`, which is also the CLI's `--n-vials` default -- the drum's
        # geometry, not a preference, and it is not a config key precisely because it is the rig.
        from flygym_tracker.calibration import VIALS_PER_FACE

        if not self.stage.begin_draw(out_dir=calib, n_vials=VIALS_PER_FACE):
            self.refresh_readiness()

    def _ask_redraw(self, saved) -> bool:
        """True = draw again. False = keep what is saved.

        The DEFAULT differs by what was found, exactly as `live_vial_selector.prompt_reuse`
        documents: a hand-drawn bundle defaults to keeping (saying no costs a keystroke, a mistaken
        redraw costs a whole clicking session), while older AUTO-DETECTED boxes are named as such,
        because those are the ones known to sit crookedly on the tubes.
        """
        from flygym_tracker.live_vial_selector import _format_saved_time

        text = ("%s already holds %d vial position(s) (%s, saved %s)."
                % (self.state.get("calib_dir") or "the calibration folder", saved.n_vials,
                   "drawn by hand" if saved.hand_drawn
                   else "AUTO-DETECTED boxes, known to sit crookedly on the tubes",
                   _format_saved_time(saved.created)))
        if self._confirm is not None:
            return bool(self._confirm(text + " Draw them again?"))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Vial positions already exist")
        box.setText(text)
        box.setInformativeText("Drawing again replaces them.")
        keep = box.addButton("Keep them", QMessageBox.ButtonRole.RejectRole)
        redraw = box.addButton("Draw again", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(keep if saved.hand_drawn else redraw)
        box.exec()
        return box.clickedButton() is redraw

    def _begin_noise(self, calib: str) -> None:
        """Measure the noise floor here, watching the rig it is being measured on."""
        mask, problem = _illum_mask(calib)
        if mask is None:
            QMessageBox.warning(self, "The noise floor cannot be measured yet", problem)
            return
        video = None
        if not self.session.is_open:
            video = self._pick_video("Measure the noise floor on which recording?")
            if video is None:
                self.stage.caption.setText(
                    "The noise floor needs frames: open the camera, or pick a recording.")
                return
        self.stage.begin_noise(mask, k=float(_cfg(self.config, "activity.k") or 5.0),
                               video=video)

    def _begin_face_learning(self) -> None:
        video = None
        if not self.session.is_open:
            video = self._pick_video("Learn the drum faces from which recording?")
            if video is None:
                self.stage.caption.setText(
                    "Learning the faces needs frames: open the camera, or pick a recording.")
                return
        self.stage.begin_face_learning(video=video)

    def _begin_replay(self, calib: str) -> None:
        """Replay a recording through the IDENTICAL pipeline, watched in this window.

        A replay is not a different program from a run -- it is the same pipeline with a file
        instead of a sensor, which is what makes it worth anything as a check. So it goes through
        `RunController` like a run does, fills the same frame box, and is watched in the same view.
        The `source_factory` is the only difference, and it is built ON THE RUN THREAD (the plan
        carries a callable, not an open file) so nothing opens a video on the GUI thread.
        """
        video = self._pick_video("Replay which recording?")
        if video is None:
            return
        plan = {
            "config": self._config_for_run(),
            "calib_dir": calib,
            "output_dir": self.state.get("output_dir") or getattr(
                getattr(self.config, "output", None), "dir", None),
            "config_path": self.controller.config_path,
            "source_factory": _video_source_factory(video),
            "replay_of": video,
        }
        if not self.run.start(plan):
            self.run_panel.set_run_state(self.run.state, self.run.detail)
            return
        self.stage.show_run()
        self.settings_view.refresh()

    def _on_job_finished(self, kind: str, payload: dict) -> None:
        """A video job ended. Say what it produced, and put a measurement where it belongs.

        THE NOISE FLOOR IS OFFERED INTO THE SETTINGS, NOT WRITTEN TO THE CONFIG FILE. Its whole
        output is three suggested thresholds, and the old command printed them to a console for the
        operator to retype. They land on the rows instead, as UNSAVED changes -- so they are
        visible, comparable against what was there, revertable, and saved only when the operator
        says so. Writing them straight to disk would be a measurement silently redefining what
        every later activity reading means.
        """
        self.settings_view.set_status(payload.get("message", ""))
        if kind == "noise" and not payload.get("failed"):
            self._offer_noise_thresholds(payload)
        self.settings_view.refresh()
        self.refresh_readiness()

    def _offer_noise_thresholds(self, payload: dict) -> None:
        suggested = {
            "activity.pixel_threshold": payload.get("suggested_pixel_threshold"),
            "rotation.enter_threshold": payload.get("suggested_enter_threshold"),
            "rotation.exit_threshold": payload.get("suggested_exit_threshold"),
        }
        applied = []
        for key, value in suggested.items():
            if value is None:
                continue
            try:
                result = self.controller.commit(key, float(value))
            except Exception:
                continue
            # `.moved`, NOT the result object. `commit` returns a `CommitResult` for every call,
            # including a refusal and a no-op, and every one of those is truthy -- so testing the
            # object would report a threshold as "put on the row" when the row had refused it.
            if getattr(result, "moved", False):
                applied.append(key)
        if applied:
            self.settings_view.set_status(
                "%s  -  %d suggested threshold(s) put on the rows, UNSAVED. Check them, then save."
                % (payload.get("message", ""), len(applied)))

    def _ask(self, text: str) -> bool:
        if self._confirm is not None:
            return bool(self._confirm(text))
        return QMessageBox.question(
            self, "Start the run?", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    # -- saving + readiness ---------------------------------------------------------------------
    def save_settings(self) -> None:
        """Write the changed rows. Confirms ONLY if a value was chosen with no camera to check it.

        An ordinary save is not confirmed: it rewrites only what changed and keeps every comment.
        The confirm exists for the one case that can quietly ruin the next experiment -- see
        `settings_controller.arming_plan`.
        """
        result = self.controller.save(confirm=self._confirm_unverified)
        self.settings_view.set_status(result.message)
        self.settings_view.refresh_titles()
        self.refresh_readiness()

    def _confirm_unverified(self, warning: str) -> bool:
        if self._confirm is not None:
            return bool(self._confirm(warning))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Save values no camera has confirmed?")
        box.setText(warning)
        write = box.addButton("Write them anyway", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is write

    def refresh_readiness(self) -> None:
        labels = {s.key: s.label for s in self.controller.model.settings}
        self.readiness_strip.set_readiness(readiness.evaluate(
            config_path=self.controller.config_path,
            calib_dir=self.state.get("calib_dir"),
            output_dir=self.state.get("output_dir"),
            camera_state=self.session.state,
            camera_detail=self.session.detail,
            never_checked=self.controller.never_checked(),
            labels=labels,
            n_changed=len(self.controller.changed()),
        ))

    def _on_fix(self, action: str) -> None:
        """Wire a readiness fix button to the thing that fixes it."""
        if action == "open_camera":
            self.open_camera()
        elif action == "free_camera":
            self.show_camera_lock()
        elif action == "save":
            self.save_settings()
        elif action == "pick_config":
            self.session_bar._browse_config()
        elif action == "pick_calib":
            self.session_bar.calib_field._pick()
        elif action == "pick_output":
            self.session_bar.output_field._pick()
        elif action == "draw_vials":
            # It is a button now, so the fix is the button rather than a paragraph telling the
            # operator which run.bat menu entry to type.
            self._on_tool("draw_vials")

    # -- closing --------------------------------------------------------------------------------
    def _ask_discard(self, text: str) -> bool:
        if self._confirm is not None:
            return bool(self._confirm(text))
        answer = QMessageBox.question(
            self, "Discard unsaved changes?", text + "\n\nDiscard them?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        """Refuse to finish until the camera is released, and ask about unsaved work first.

        LEAKING AN EXCLUSIVE USB3 HANDLE IS WHAT CREATES THE NEXT SESSION'S "camera is busy" -- with
        no window on screen to explain it, which is precisely the failure `camera_lock` exists to
        diagnose. So the shutdown is ordered and synchronous rather than left to interpreter exit.
        """
        # A RUN IN PROGRESS IS THE THIRD THING WORTH CONFIRMING. It is not reversible and it may be
        # days of irreplaceable samples; closing the window ends it. The other two (stopping
        # another program, closing with unsaved changes) were already here, and the list stays
        # short on purpose -- a surface that asks about everything trains an operator to click past
        # the question that mattered.
        if self.run.is_running and not self._ask_discard(
                "An experiment is running. Closing this window stops it and closes its files."):
            event.ignore()
            return
        changed = self.controller.changed()
        if changed:
            decision = self._ask_close_with_changes(len(changed))
            if decision == "cancel":
                event.ignore()
                return
            if decision == "save":
                self.save_settings()
        self._status_timer.stop()
        # THE RUN GOES DOWN FIRST, and synchronously. It holds the exclusive camera handle and an
        # open logger; abandoning the thread at interpreter exit would leave a partial bin
        # unflushed and the handle leaked -- which is how the NEXT session's "camera is busy"
        # appears, with no window on screen to explain it.
        self.run.shutdown()
        # BEFORE the camera session: the stage may hold a tap on the camera thread and a thread of
        # its own reading a video. Tearing the camera down under an attached job would have the
        # grab loop calling into a job whose owner has gone.
        self.stage.shutdown()
        self.session.shutdown()
        gui_state.save_state(self.root, self.state)
        event.accept()

    def _ask_close_with_changes(self, n: int) -> str:
        """"save" | "discard" | "cancel"."""
        if self._confirm is not None:
            return "discard" if self._confirm("close with %d unsaved change(s)" % n) else "cancel"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Unsaved settings")
        box.setText("%d setting change(s) have not been written to %s."
                    % (n, self.controller.config_path or "the config file"))
        box.setInformativeText("A run started now would use the file's old values.")
        save = box.addButton("Save and close", QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton("Close without saving", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is save:
            return "save"
        if clicked is discard:
            return "discard"
        return "cancel"


def _video_source_factory(video: str):
    """A callable that opens `video`, built to run ON THE RUN THREAD.

    A module-level function rather than a closure defined in the method: the plan dict is handed
    across a thread boundary, and a closure over `self` would carry the whole window -- and every
    widget in it -- into an object the run thread holds for the length of the run.
    """
    def factory():
        from flygym_tracker.frame_source import VideoFileSource

        return VideoFileSource(video)

    return factory


def _illum_mask(calib_dir: str):
    """``(mask, None)`` for the first face's illumination mask, or ``(None, why not)``.

    The noise floor is measured INSIDE this mask -- the lit part of the picture -- so without it
    there is no measurement to make. Every failure is a sentence naming the folder, because all of
    them are fixed by drawing the vials or by pointing at a different bundle.
    """
    import cv2

    from flygym_tracker.calibration import load_calibration

    if not calib_dir:
        return None, "No calibration folder is chosen, so there is no lit area to measure inside."
    try:
        calib = load_calibration(calib_dir)
    except Exception as exc:
        return None, ("%s does not hold a calibration bundle yet (%s). Draw the vial positions "
                      "first - that is what writes the illumination mask." % (calib_dir, exc))
    if not calib.faces:
        return None, "%s holds a calibration with no faces in it." % calib_dir
    face = "A" if "A" in calib.faces else sorted(calib.faces)[0]
    path = calib.faces[face].illum_mask_path
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "The illumination mask for face %s could not be read (%s)." % (face, path)
    return mask, None


def _cfg(config, path: str):
    """A dotted config lookup that tolerates a missing branch (returns None)."""
    node = config
    for part in path.split("."):
        try:
            node = getattr(node, part)
        except Exception:
            return None
    return node
