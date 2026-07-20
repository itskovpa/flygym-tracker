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

import os
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QDockWidget, QFileDialog, QMainWindow, QMessageBox, QSplitter,
                               QVBoxLayout, QWidget)

from flygym_tracker import camera_lock, readiness
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.camera_lock_dialog import CameraLockDialog, qt_confirm
from flygym_tracker.gui.camera_session import (CLOSED, CLOSING, ERROR_BUSY, ERROR_OTHER,
                                               OPENING, STREAMING, CameraSession)
from flygym_tracker.gui.camera_status import CameraStatusBar
from flygym_tracker.gui.behaviour_series import BehaviourSeries
from flygym_tracker.gui.plot_dock import BehaviourPlotDock
from flygym_tracker.gui.readiness_strip import ReadinessStrip
from flygym_tracker.gui.results_panel import ResultsPanel
from flygym_tracker.gui.run_controller import RunController
from flygym_tracker.gui.run_panel import RunPanel
from flygym_tracker.gui.session_bar import SessionBar
from flygym_tracker.gui.settings_view import SettingsView
from flygym_tracker.gui.run_controller import DONE, FAILED, IDLE, RUNNING, STARTING
from flygym_tracker.gui.video_stage import RUN as STAGE_RUN
from flygym_tracker.gui.video_stage import VideoStage
from flygym_tracker.settings_controller import (SettingsController, camera_block_reason,
                                                is_start_only)
from flygym_tracker.settings_model import build_app_settings

#: How often the status bar's delivered-fps figure is refreshed. A label, not a measurement: the
#: number itself is counted in the worker from frames that arrived, and nothing is asked of the
#: camera to produce it.
STATUS_REFRESH_MS = 500

#: How long `start_run` waits for the preview camera to actually release the device before giving
#: up on a clean handover. Generous -- a close that takes longer than this has gone wrong, and the
#: honest thing then is to say so rather than to start a run on a camera somebody else still holds.
HANDOVER_TIMEOUT_MS = 6000

#: How long a video job waits for the camera it asked to be opened. Enumerating and configuring a
#: USB3 Vision device is not instant; a wait longer than this means something is wrong, and saying
#: so beats a button that appears to have done nothing.
CAMERA_OPEN_TIMEOUT_MS = 8000


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
        #: A run was asked for while the preview still held the camera. See `start_run`.
        self._pending_start = False
        #: A video job waiting for the camera to finish opening. See `with_camera`.
        self._camera_then = None
        self._camera_why = ""
        #: Every behaviour row of this run, shared by every open plot dock. ONE STORE: a dock
        #: opened at hour 40 draws the whole run rather than only what arrives after it opened.
        self.behaviour = BehaviourSeries()
        self._plot_docks = {}
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

        # THE SETTINGS ARE A DOCK, NOT A COLUMN, and that is a screen-space decision the rig owner
        # made: the settings column was "occupying too much screen space with a lot of dead space".
        # Ten rows of controls held a 560 px column open beside a picture that wants every pixel,
        # and most of that column was empty below the last row.
        #
        # A QDockWidget rather than a hand-rolled collapsible: it closes to nothing, it FLOATS AS A
        # REAL SEPARATE WINDOW -- which is what was asked for, so the settings can sit on a second
        # monitor while the rig picture fills this one -- Qt handles the drag and the re-dock, and
        # `toggleViewAction` is the single source of truth for "is it showing", so the button and
        # the dock can never disagree about it.
        self.settings_view = SettingsView(self.controller)
        self.settings_dock = QDockWidget("Settings", self)
        self.settings_dock.setObjectName("settings")
        self.settings_dock.setWidget(self.settings_view)
        self.settings_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                                           | Qt.DockWidgetArea.RightDockWidgetArea)
        self.settings_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                                       | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                                       | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.settings_dock)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        # THE PICTURE IS THE STAGE EVERY VIDEO OPERATION IS PERFORMED ON. Drawing vial positions,
        # replaying a recording, measuring the noise floor and learning the drum faces used to be
        # four child processes with four OpenCV windows; they are modes of this one widget now.
        self.stage = VideoStage(self.session, self.run)
        #: The old name, kept because it is what the window's own code and its tests call the
        #: picture. It is the same object -- there is only one picture in this window.
        self.preview = self.stage
        splitter.addWidget(self.stage)
        # THE MEASUREMENT, BESIDE THE PICTURE. Until now the window showed frames, fps and a
        # brightness strip -- all of which say the machine is running, none of which is a number
        # that reaches the results. It is HIDDEN until a run starts: with nothing measuring, an
        # empty table of result columns is a promise the window cannot keep, and it would be
        # taking width from the picture that the picture needs for drawing vials.
        self.results = ResultsPanel()
        self.results.setVisible(False)
        splitter.addWidget(self.results)
        # Two panes now that the settings have their own dock: the picture and, once a run starts,
        # the measurement. The PICTURE keeps the larger share -- it is the pane that cannot be read
        # by scrolling.
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([760, 420])
        self._splitter = splitter
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
        # BOTH DIRECTIONS, so the button follows the dock however the dock was closed -- its own
        # X, a drag, or this button. One of them driving the other only would leave the button
        # showing "on" over a dock that is not there.
        self.status_bar.settings_button.toggled.connect(self.settings_dock.setVisible)
        self.settings_dock.visibilityChanged.connect(self._on_settings_visibility)
        self.status_bar.settings_button.setChecked(True)

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
        self.run_panel.plot_requested.connect(self.show_plot)
        self.stage.job_finished.connect(self._on_job_finished)
        self.stage.mode_changed.connect(self._on_stage_mode)
        self.run.state_changed.connect(self._on_run_state)
        self.run.progress.connect(self.run_panel.set_progress)
        self.run.progress.connect(self.results.set_progress)
        self.run.progress.connect(self._on_run_progress)
        self.run.bin_done.connect(self.results.add_bin)
        self.run.behaviour_done.connect(self._on_behaviour_rows)
        self.run.setting_applied.connect(self._on_run_setting_applied)
        self.session_bar.config_changed.connect(self._on_config_changed)
        self.session_bar.calib_changed.connect(self._on_calib_changed)
        self.session_bar.output_changed.connect(self._on_output_changed)
        self.readiness_strip.fix_requested.connect(self._on_fix)

        # The handover watchdog. A single-shot timer rather than a wait: the GUI thread must stay
        # responsive while the camera thread releases the device, or "seamless" becomes "frozen".
        self._handover_timer = QTimer(self)
        self._handover_timer.setSingleShot(True)
        self._handover_timer.timeout.connect(self._handover_timed_out)

        # The other watchdog: a job asked for the camera and the camera never arrived.
        self._camera_timer = QTimer(self)
        self._camera_timer.setSingleShot(True)
        self._camera_timer.timeout.connect(self._camera_open_timed_out)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(STATUS_REFRESH_MS)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    def _on_settings_visibility(self, visible: bool) -> None:
        button = self.status_bar.settings_button
        if button.isChecked() != visible:
            button.blockSignals(True)          # not a request to change the dock; it already did
            button.setChecked(visible)
            button.blockSignals(False)

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
        # THE HANDOVER COMPLETES HERE. `start_run` asked the preview to close and left; this is the
        # camera thread reporting that the device is actually released, which is the first moment a
        # run can legally open it.
        if state == CLOSED and self._pending_start:
            self._start_run_now()
        # A JOB IS WAITING FOR THIS CAMERA. Both endings are handled: streaming runs it, and any
        # error path tells it why it will not run rather than leaving it pending forever.
        if self._camera_then is not None:
            if state == STREAMING:
                self._camera_ready()
            elif state in (ERROR_BUSY, ERROR_OTHER):
                self._camera_failed(detail or state)
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
        # The pane lives in the dock now, so the swap is one call rather than a walk up the
        # parents looking for a splitter that no longer holds it.
        self.settings_dock.setWidget(new)
        old.setParent(None)
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
        """Begin an experiment in this window. ONE CLICK, no question.

        WHAT THIS USED TO ASK, and why it is gone. USB3 Vision is exclusive, so the preview handle
        and the run's handle cannot both exist; the preview has to close first. That was put behind
        a confirm on the grounds that the picture vanishing unexplained at the moment attention is
        highest would be alarming. In practice the operator has just pressed "Start experiment" --
        handing the camera over IS what they asked for, and a dialog that only ever has one sensible
        answer trains people to click past dialogs that matter.

        THE PROMPT WAS ALSO HIDING A RACE, which is the part that had to be built rather than
        deleted. `session.close()` only POSTS the close to the camera thread: `is_open` goes false
        at once, but the SDK handle is released later, on that thread. The old code closed and
        started the run in the same breath -- correct only because a human took a second or two to
        read the dialog and click Yes. Remove the dialog and the run would routinely try to open a
        camera the preview had not finished releasing, and the SDK's answer to that is a
        culprit-free 0x80000203.

        So the start is SEQUENCED: close the preview, remember that a run is wanted, and start it
        from `_on_camera_state` when the camera reports CLOSED.
        """
        if self.session.is_open or self.session.state in (OPENING, CLOSING):
            self._pending_start = True
            self.stage.show_run()          # the picture switches now, not after the handover
            self.run_panel.set_run_state(STARTING, "handing the camera to the run")
            self._handover_timer.start(HANDOVER_TIMEOUT_MS)
            self.session.close()
            return
        self._start_run_now()

    def _start_run_now(self) -> None:
        """Build the plan and hand it to the run thread. The camera is free by now."""
        self._pending_start = False
        self._handover_timer.stop()
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
        self.results.clear()
        self.behaviour.clear()
        for dock in self._plot_docks.values():
            dock.refresh()
        self._show_run_vials()
        if not self.run.start(plan):
            self.run_panel.set_run_state(self.run.state, self.run.detail)
            self.stage.show_camera()       # the run did not begin; stop implying it did
            return
        # THE EXPERIMENT IS WATCHED IN THIS WINDOW. The preview camera has just been handed over,
        # so without this the picture would sit on the last frame the preview saw -- a still of the
        # rig, indistinguishable from a live one, for however many days the run lasts.
        self.stage.show_run()
        # THE ROUTER NOW HAS A RUN TO ROUTE TO. From here `_on_setting_change` sends camera AND
        # algorithm keys into `TrackerPipeline.apply_setting`, which applies them AND logs them as
        # `setting_change` events (invariant 4).
        self.settings_view.refresh()

    #: Width shares once the results pane appears: settings, picture, results. THE PICTURE KEEPS
    #: THE LARGEST SHARE because it is where the run is actually watched -- and it is the pane that
    #: cannot be read by scrolling. Measured without this: showing the pane left the splitter at
    #: [560, 320, 600], i.e. the picture squeezed to its 320 px minimum while the results table --
    #: which scrolls perfectly well at half that -- took the most room on screen.
    RUN_WIDTH_SHARES = (0.60, 0.40)

    def _share_width_with_results(self) -> None:
        """Re-proportion the three panes the first time the results appear.

        Set EXPLICITLY rather than left to Qt. A newly shown pane is given whatever the stretch
        factors and minimum sizes happen to produce, which here was the picture at its minimum --
        so the pane that matters most shrank to make room for the one that scrolls.
        """
        sizes = self._splitter.sizes()
        if len(sizes) != len(self.RUN_WIDTH_SHARES):
            return
        total = sum(sizes) or self._splitter.width()
        if total <= 0:
            return
        self._splitter.setSizes([max(300, int(total * share)) for share in self.RUN_WIDTH_SHARES])

    def _handover_timed_out(self) -> None:
        """The preview never reported CLOSED. Say so; do not start a run on a held camera.

        Starting anyway would meet the SDK's culprit-free 0x80000203 from inside a worker thread --
        the exact failure `camera_lock` exists to diagnose, arriving with no diagnosis.
        """
        if not self._pending_start:
            return
        self._pending_start = False
        self.stage.show_camera()
        self.run_panel.set_run_state(
            IDLE, "the preview camera did not release in %.0f s, so the run was not started - "
                  "try Free the camera..." % (HANDOVER_TIMEOUT_MS / 1000.0))
        self.refresh_readiness()

    def _show_run_vials(self) -> None:
        """Put the bundle's vial shapes on the picture for the run to be watched through."""
        from flygym_tracker.calibration import load_calibration
        from flygym_tracker.gui.vial_overlay import overlay_from_calibration

        try:
            calib = load_calibration(self.state.get("calib_dir") or "calib_faces")
        except Exception:
            self.stage.set_run_overlay(None)
            return
        self.stage.set_run_overlay(overlay_from_calibration(calib))

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
        # THE RESULTS PANE APPEARS WITH THE RUN AND STAYS AFTER IT ENDS. Hiding it the moment the
        # run finishes would take the last bins off the screen at the exact moment somebody wants
        # to read them -- and the run's final, partial bin is flushed at the very end.
        if state in (STARTING, RUNNING) and not self.results.isVisible():
            self.results.setVisible(True)
            self._share_width_with_results()
        # A finished run gives the picture back to the camera preview, so the next thing the
        # operator does is not done against the last frame of the last experiment.
        if state in (DONE, FAILED, IDLE) and self.stage.mode == STAGE_RUN:
            self.stage.show_camera()
        # Width/Height must LOOK dead while the stream is running, not merely refuse when pressed
        # (invariant 3). The refusal itself is the pipeline's `setting_block_reason`; this is the
        # refresh that puts it on screen the moment the run starts and takes it off when it ends.
        self.settings_view.refresh()
        self.refresh_readiness()

    def _on_behaviour_rows(self, payload: dict) -> None:
        """Completed dwells arrived: store them and redraw whatever plots are open."""
        if not self.behaviour.add(payload.get("rows") or []):
            return
        for dock in self._plot_docks.values():
            dock.refresh()

    def show_plot(self, field: str) -> None:
        """Open (or raise) the dock for one behavioural parameter.

        RAISED RATHER THAN DUPLICATED: two docks of the same parameter would be two identical
        graphs the operator then has to tell apart, and closing one would look like it had failed
        to close the other.
        """
        dock = self._plot_docks.get(field)
        if dock is None:
            dock = BehaviourPlotDock(self.behaviour, field, self)
            self._plot_docks[field] = dock
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            # TABBED WITH ITS SIBLINGS rather than stacked: four parameters split vertically would
            # leave each grid too short to read, and the operator asked for a graph per parameter.
            others = [d for key, d in self._plot_docks.items() if key != field and d.isVisible()]
            if others:
                self.tabifyDockWidget(others[-1], dock)
        dock.show()
        dock.raise_()
        dock.refresh()

    def _on_run_progress(self, payload: dict) -> None:
        """Tint the vial outlines on the picture by what each vial is reporting."""
        self.stage.set_run_activity(payload.get("vial_results") or {})

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
        elif action == "mark_band":
            self._begin_band(calib)
        elif action == "noise":
            self._begin_noise(calib)
        elif action == "learn_faces":
            self._begin_face_learning()
        elif action == "replay":
            self._begin_replay(calib)

    # -- getting the camera for a job -------------------------------------------------------------
    def with_camera(self, then, *, why: str) -> None:
        """Run `then()` once the LIVE camera is streaming, opening it if it is not.

        THE RIG OWNER'S RULE, verbatim: "as soon as I click draw vial positions or learn drum faces
        it needs to do the necessary steps for the operation, always prioritize live camera for all
        measurements." Before this, pressing Draw vial positions with the camera closed did
        NOTHING VISIBLE -- the job refused, wrote a sentence into the caption under the picture, and
        the operator was left to work out that a different button in a different band had to be
        pressed first. The button did not do its job; it described a precondition.

        THE OPEN IS ASYNCHRONOUS, which is why this is a continuation and not two lines. `open()`
        posts to the camera thread and returns; the device is not usable until that thread reports
        STREAMING. Calling the job straight after would have it ask a camera that is not there yet
        -- the same race `start_run` had, in the other direction.

        WHY THIS DOES NOT VIOLATE "the app never takes the camera by itself". That rule is about
        LAUNCH: an app that grabs an exclusive device the moment it opens is an app that blocks the
        rig. Here the operator has pressed a button whose whole meaning is "do this to the rig
        now", so taking the camera IS what was asked for. It is still never taken without a click.
        """
        if self.session.is_open:
            then()
            return
        if self.run.is_running:
            # The run owns the camera and must keep it. Saying so is the useful answer; opening
            # would fail with the SDK's culprit-free error, and stopping the run to draw vials
            # would end an experiment to change its calibration.
            self.stage.caption.setText(
                "the experiment has the camera - stop the run first, then %s" % why)
            return
        self._camera_then = then
        self._camera_why = why
        self.stage.caption.setText("opening the camera to %s..." % why)
        self._camera_timer.start(CAMERA_OPEN_TIMEOUT_MS)
        self.session.open()

    def _camera_ready(self) -> None:
        """The camera reached STREAMING. Run whatever was waiting for it."""
        self._camera_timer.stop()
        then, self._camera_then = self._camera_then, None
        if then is not None:
            then()

    def _camera_failed(self, detail: str) -> None:
        """The camera did not open. Say why, in terms of the job that wanted it."""
        self._camera_timer.stop()
        why, self._camera_then, self._camera_why = self._camera_why, None, ""
        if not why:
            return
        self.stage.caption.setText(
            "could not open the camera to %s: %s   -   try Free the camera..." % (why, detail))
        self.refresh_readiness()

    def _camera_open_timed_out(self) -> None:
        self._camera_failed("it did not start streaming in %.0f s"
                            % (CAMERA_OPEN_TIMEOUT_MS / 1000.0))

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

        # SAVED POSITIONS ARE LOADED AND SHOWN, NOT OFFERED BEHIND A YES/NO. Reuse used to be
        # all-or-nothing: keep them exactly as they are, or throw them away and re-click all
        # sixteen. There was no way to SEE what was saved, so a bundle that was 15/16 right cost a
        # whole clicking session to correct -- and nothing on screen said it was 15/16 right.
        # Now they open on the picture with draggable corners; "Start over" is still one button.
        saved = saved_selection(calib)
        polygons = saved.polygons if saved is not None else None
        # `calibration.VIALS_PER_FACE`, which is also the CLI's `--n-vials` default -- the drum's
        # geometry, not a preference, and it is not a config key precisely because it is the rig.
        from flygym_tracker.calibration import VIALS_PER_FACE

        def draw():
            if not self.stage.begin_draw(out_dir=calib, n_vials=VIALS_PER_FACE,
                                         polygons=polygons):
                self.refresh_readiness()

        self.with_camera(draw, why="draw the vial positions")

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

    def _begin_band(self, calib: str) -> None:
        """Mark where the marker band is, and store it beside the vial positions.

        IT GOES IN THE VIAL-POSITIONS BUNDLE, which is the rig owner's own call and the right one:
        the band's location is a fact about this rig's geometry, exactly like the vial polygons,
        and it belongs with them rather than in a config file that travels between rigs. Anything
        that loads the bundle -- a run, a replay, a face-learning session -- picks it up without
        being told (`calibration.marker_detector_from_calibration`).
        """
        import os

        if not os.path.isfile(os.path.join(calib, "calibration.json")):
            QMessageBox.warning(
                self, "There is nowhere to save the marker band yet",
                "%s does not hold a calibration bundle. Draw the vial positions first - the "
                "marker band is stored beside them." % calib)
            return
        self.stage.begin_band(out_dir=calib)

    def _begin_noise(self, calib: str) -> None:
        """Measure the noise floor here, watching the rig it is being measured on."""
        mask, problem = _illum_mask(calib)
        if mask is None:
            QMessageBox.warning(self, "The noise floor cannot be measured yet", problem)
            return
        # The noise floor is a property of THIS rig as it stands right now -- its illumination, its
        # exposure, its sensor. Measuring it from a recording answers the question for the rig as
        # it was when that clip was taken, which is not the question being asked.
        self.with_camera(
            lambda: self.stage.begin_noise(mask, k=float(_cfg(self.config, "activity.k") or 5.0)),
            why="measure the noise floor")

    def _begin_face_learning(self) -> None:
        # THE LIVE CAMERA IS THE POINT of this step: it watches the drum turn NOW. Asking which
        # recording to use before even trying the camera had it backwards.
        self.with_camera(
            lambda: self.stage.begin_face_learning(band_rows=self._band_rows()),
            why="learn the drum faces")

    def _band_rows(self):
        """The marker band the operator drew, if this bundle carries one. None means "guess it".

        Read fresh from the bundle rather than cached: the band can be redrawn between a run and a
        learning session, and a stale copy would have learning read a different region than every
        identification afterwards.
        """
        from flygym_tracker.calibration import calibration_band_rows, load_calibration

        calib = self.state.get("calib_dir") or "calib_faces"
        try:
            return calibration_band_rows(load_calibration(calib))
        except Exception:
            return None

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
        self.results.clear()
        self.behaviour.clear()
        for dock in self._plot_docks.values():
            dock.refresh()
        self._show_run_vials()
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
        if kind == "faces" and payload.get("complete"):
            self._save_face_templates(payload)
        self.settings_view.refresh()
        self.refresh_readiness()

    def _save_face_templates(self, payload: dict) -> None:
        """Write what face learning just learned into the calibration bundle.

        WITHOUT THIS THE STEP DID NOTHING. It watched the drum, learned a template per face, said
        so on screen -- and dropped the templates when the job object went out of scope. The next
        run would still start, still fill a CSV, and still record every face-B vial as face A,
        which is the exact bug `face_learning` exists to close and the reason its own docstring
        calls declining this step "the option that quietly produces wrong data".

        The CLI has always done this (`live_vial_selector.learn_faces_for_bundle`); this is the
        same call on the same bundle, so a session started from the window and one started from a
        terminal leave the same thing on disk.
        """
        from datetime import datetime

        from flygym_tracker.calibration import attach_face_templates

        calib = self.state.get("calib_dir") or "calib_faces"
        detector = payload.get("detector")
        if detector is None:
            return
        try:
            written = attach_face_templates(
                calib, detector,
                extra={"band_learned": datetime.now().isoformat(timespec="seconds"),
                       "band_dwells": len(payload.get("dwells") or [])})
        except Exception as exc:
            # NAME THE CONSEQUENCE, not just the error. The operator has just spent 10-20 s turning
            # the drum and is entitled to know that the run they are about to start still cannot
            # tell the faces apart.
            self.settings_view.set_status(
                "the drum faces were learned but could NOT be saved to %s (%s) - a run started now "
                "would still record every vial as one face" % (calib, exc))
            return
        self.settings_view.set_status(
            "face templates saved to %s for face(s) %s - this bundle can identify faces now"
            % (calib, ", ".join(written)))

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
        redirected = self._redirect_away_from_the_template()
        result = self.controller.save(confirm=self._confirm_unverified)
        # The order of these two no longer matters -- `SettingsView` holds the status text as state
        # and re-renders it, precisely so no caller has to know. It used to matter and be wrong:
        # `refresh_titles` rewrote the change line from the change COUNT alone, wiping the save
        # result off the screen every time.
        self.settings_view.refresh_titles()
        self.settings_view.set_status(
            "%s  -  %s" % (result.message, redirected) if redirected and result.saved
            else result.message)
        self.refresh_readiness()

    def _redirect_away_from_the_template(self) -> str:
        """Send a save aimed at a SHIPPED config to this machine's own copy instead.

        THE PROBLEM THIS SOLVES. `config/flygym_rig.yaml` is in version control: it is what a fresh
        clone gets, and it is asserted to leave every camera field null so an untouched install
        imposes nothing on the sensor. But it was also the file the app opened and saved into, so
        an ordinary afternoon of tuning rewrote the shipped default to whatever the last operator
        was trying -- silently, and visible only in a diff nobody reads before an experiment.

        REDIRECTING RATHER THAN REFUSING, because refusing punishes the operator for a filing
        decision they did not make and should not have to think about. The values are theirs and
        they asked for them to be kept; what changes is only WHICH FILE keeps them, and the status
        line says so by name. `config/flygym_rig.local.yaml` layers on top of the template
        (`config.load_config`), so nothing is lost and the template stays a template.

        Returns a sentence to append to the save message, or "" when nothing was redirected.
        """
        from flygym_tracker.config import is_tracked_template, local_config_path

        path = self.controller.config_path
        if not is_tracked_template(path):
            return ""
        local = str(local_config_path(path))
        self.controller.config_path = local
        self.state["config_path"] = local
        gui_state.remember_config(self.state, local)
        gui_state.save_state(self.root, self.state)
        self.session_bar.set_recent(self.state["recent_configs"])
        self.session_bar.config_combo.setCurrentText(local)
        return ("written to this rig's own %s, not the shipped template - the template stays as a "
                "fresh clone gets it, and your values layer on top of it"
                % os.path.basename(local))

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
