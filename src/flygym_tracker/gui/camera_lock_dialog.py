"""A window over `camera_lock`, which already does the thinking. This adds no policy.

WHY THIS IS A FRONT END AND NOT A REIMPLEMENTATION. `camera_lock` is invariant 5's implementation
and it is careful in ways that are easy to lose by rewriting: `find_camera_holders` has no side
effects at all, never nominates this process or any ancestor of it (so it cannot offer to kill
run.bat or itself), matches an explicit list of programs rather than guessing, and never nominates
a bare `python.exe` unless its command line names this package -- because ending a scientist's
unrelated analysis job would be a far worse outcome than a camera that stays busy another minute.
Every one of those decisions stays over there. This file supplies a body of text and a button.

`prompt_and_release` IS NOT CALLED, and that is the one thing this module must not do. It is
terminal-bound: `input()`, `sys.stdin.isatty()`, `print`. Under a GUI it would find no terminal,
print into nowhere, and stop nothing -- while looking like it had asked. So the shape is
replicated, not the function: report -> confirm -> `release_camera` -> report what was stopped.

THE CONFIRM IS AN INJECTED CALLABLE, exactly as `release_camera(holders, confirm=...)` requires --
it has no default, on purpose, so no call can reach a kill by accident. In production it reaches a
real dialog; in tests it is a stub, and nothing can block a headless run.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QMessageBox, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from flygym_tracker import camera_lock


class CameraLockDialog(QDialog):
    """Shows what holds the camera and, only if confirmed, stops it.

    Args:
        holders: what `find_camera_holders` nominated. Passed IN rather than looked up here, so a
            test can supply a fixture process table and this class never touches the real OS.
        confirm: ``(holders) -> bool``. Required, no default (see the module docstring).
        stop: injected so a test never terminates anything real.
    """

    def __init__(self, holders: Sequence[camera_lock.CameraHolder], *,
                 confirm: Callable[[Sequence[camera_lock.CameraHolder]], bool],
                 stop: Callable[[int], bool] = camera_lock.stop_process,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("What is holding the camera")
        self.holders = list(holders)
        self._confirm = confirm
        self._stop = stop
        self.stopped: List[camera_lock.CameraHolder] = []

        layout = QVBoxLayout(self)
        intro = QLabel("Only one program at a time may use this camera.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # `report()` VERBATIM. It is already written for a non-programmer -- it explains what a
        # headless Bonsai is and why the rig looks idle while the camera is held -- and rewording
        # it here would produce a second explanation to keep in step with the first.
        body = QPlainTextEdit(camera_lock.report(self.holders))
        body.setReadOnly(True)
        body.setMinimumSize(560, 220)
        layout.addWidget(body)

        buttons = QDialogButtonBox()
        self.stop_button = buttons.addButton("Stop them and free the camera",
                                             QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Leave them alone", QDialogButtonBox.ButtonRole.RejectRole)
        # Nothing to stop means nothing to press: with no holders the dialog is purely
        # informational, and the report already says what to try instead.
        self.stop_button.setEnabled(bool(self.holders))
        self.stop_button.setDefault(False)
        self.stop_button.setAutoDefault(False)
        buttons.accepted.connect(self._on_stop)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_stop(self) -> None:
        """Hand the decision to `release_camera`, which is the only thing that may kill anything."""
        self.stopped = camera_lock.release_camera(self.holders, confirm=self._confirm,
                                                  stop=self._stop)
        self.accept()

    def summary(self) -> str:
        """What actually happened, for the status bar. Never claims more than was done."""
        if not self.stopped:
            return "Nothing was stopped."
        names = ", ".join("%s (PID %d)" % (h.what or h.name, h.pid) for h in self.stopped)
        return "Stopped %d program(s): %s. The camera should be free now." % (len(self.stopped),
                                                                             names)


def qt_confirm(parent: Optional[QWidget]) -> Callable[[Sequence[camera_lock.CameraHolder]], bool]:
    """The production `confirm`: a modal that NAMES what will be stopped and defaults to No.

    One of only two irreversible actions in this app that confirm (the other is closing with
    unsaved changes). Ending a process cannot be undone, and the operator may be looking at their
    own work, so the default button is the harmless one.
    """
    def confirm(holders: Sequence[camera_lock.CameraHolder]) -> bool:
        listing = "\n".join("    PID %d  %s" % (h.pid, h.what or h.name) for h in holders)
        box = QMessageBox(parent)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Stop these programs?")
        box.setText("This will END %d running program(s):\n\n%s" % (len(holders), listing))
        box.setInformativeText(
            "Anything they were doing is lost. If one of them is your own work, close it yourself "
            "instead.")
        yes = box.addButton("Stop them", QMessageBox.ButtonRole.DestructiveRole)
        no = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(no)
        box.exec()
        return box.clickedButton() is yes

    return confirm
