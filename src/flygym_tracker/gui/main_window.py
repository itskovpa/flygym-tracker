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
from flygym_tracker.gui.preview import PreviewPane
from flygym_tracker.gui.readiness_strip import ReadinessStrip
from flygym_tracker.gui.run_controller import RunController
from flygym_tracker.gui.run_panel import RunPanel, launch_cli_tool
from flygym_tracker.gui.session_bar import SessionBar
from flygym_tracker.gui.settings_view import SettingsView
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
        self.preview = PreviewPane(self.session)
        splitter.addWidget(self.preview)
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

    def _on_run_state(self, state: str, detail: str) -> None:
        self.run_panel.set_run_state(state, detail)
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
        """Run one of `run.bat`'s old menu entries. cv2 tools go to a CHILD PROCESS -- see
        `run_panel.launch_cli_tool` for the DPI measurement that makes that mandatory."""
        if action == "free_camera":
            self.show_camera_lock()
            return
        config_path = self.controller.config_path or "config/flygym_rig.yaml"
        calib = self.state.get("calib_dir") or "calib_faces"
        if action == "draw_vials":
            args = ["select-vials", "--out", calib, "--config", config_path]
        elif action == "noise":
            args = ["noise", "--calib", calib, "--config", config_path]
        elif action == "replay":
            video, _ = QFileDialog.getOpenFileName(
                self, "Replay which recording?", "", "Video files (*.avi *.mp4 *.mkv);;All files (*)")
            if not video:
                return
            args = ["replay", "--video", video, "--calib", calib, "--config", config_path,
                    "--monitor"]
        else:
            return
        try:
            launch_cli_tool(args, cwd=self.root)
        except Exception as exc:
            # A tool that silently does not open is indistinguishable from one that is slow to
            # start, and the operator's next move is to press the button again.
            QMessageBox.warning(self, "That tool could not be started", str(exc))
            return
        self.settings_view.set_status("started: %s" % " ".join(args[:2]))

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


def _cfg(config, path: str):
    """A dotted config lookup that tolerates a missing branch (returns None)."""
    node = config
    for part in path.split("."):
        try:
            node = getattr(node, part)
        except Exception:
            return None
    return node
