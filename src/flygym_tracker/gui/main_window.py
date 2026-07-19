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
from PySide6.QtWidgets import (QLabel, QLineEdit, QMainWindow, QMessageBox, QSplitter, QVBoxLayout,
                               QWidget)

from flygym_tracker import camera_lock, readiness
from flygym_tracker.gui import gui_state
from flygym_tracker.gui.camera_lock_dialog import CameraLockDialog, qt_confirm
from flygym_tracker.gui.camera_session import CLOSED, STREAMING, CameraSession
from flygym_tracker.gui.camera_status import CameraStatusBar
from flygym_tracker.gui.preview import PreviewPane
from flygym_tracker.gui.readiness_strip import ReadinessStrip
from flygym_tracker.gui.session_bar import SessionBar
from flygym_tracker.gui.settings_view import SettingsView
from flygym_tracker.settings_controller import SettingsController, camera_block_reason
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
        model = build_app_settings(config)
        self.controller = SettingsController(
            model,
            # ONE provider, matching `pipeline.setting_block_reason`'s signature exactly. Stage 2
            # swaps the pipeline's bound method in here and nothing else changes.
            block_reason=camera_block_reason(lambda: self.session.source),
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

        # The four things run.bat's menu offers that this stage does not. ONE LINE OF TEXT, not a
        # row of disabled buttons: four dead controls are four things to try before reading the
        # small print, and they make the app look like it lost features rather than not having
        # grown them yet.
        self.elsewhere = QLabel(
            "Starting a run, drawing vial positions, replaying a recording and measuring the noise "
            "floor are still in run.bat.")
        self.elsewhere.setProperty("role", "note")
        self.elsewhere.setWordWrap(True)
        layout.addWidget(self.elsewhere)

        #: The inert widget that takes the initial focus. It filters the settings list, so the
        #: worst a stray keystroke can do here is hide some rows.
        self.filter_box = QLineEdit()
        self.filter_box.setPlaceholderText("Type to filter the settings...")
        self.filter_box.setClearButtonEnabled(True)
        self.filter_box.textChanged.connect(self._on_filter)
        layout.addWidget(self.filter_box)

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

    def _on_setting_change(self, key: str, value) -> bool:
        """The router. Stage 1 has no pipeline, so a camera row goes to the camera and nothing else
        goes anywhere -- it is a config edit that takes effect at the next run.

        Returning False for a non-camera key is what puts "not applied to this run - takes effect
        at next start" on the row, which is true and is the same thing the cv2 panel says.
        """
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
        """Hide rows whose label or help does not mention `text`. Hiding only -- never disabling.

        A hidden row still holds its value and is still saved: filtering is a way to find a setting
        in a list, not a way to exclude one from the file.
        """
        needle = (text or "").strip().lower()
        for key, row in self.settings_view.rows.items():
            haystack = "%s %s %s" % (row.setting.label, row.setting.help, key)
            row.setVisible(not needle or needle in haystack.lower())

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
            QMessageBox.information(
                self, "Drawing vial positions",
                "Vial drawing is still the cv2 tool. Start it from run.bat, option [2], or:\n\n"
                "    python -m flygym_tracker.cli select-vials --out %s --config %s"
                % (self.state.get("calib_dir") or "calib_faces",
                   self.controller.config_path or "config/flygym_rig.yaml"))

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
        changed = self.controller.changed()
        if changed:
            decision = self._ask_close_with_changes(len(changed))
            if decision == "cancel":
                event.ignore()
                return
            if decision == "save":
                self.save_settings()
        self._status_timer.stop()
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
