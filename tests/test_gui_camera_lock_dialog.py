"""Invariant 5's front end: nothing is stopped unless a confirmation says so.

`camera_lock` is already careful -- `find_camera_holders` has no side effects, never nominates this
process or an ancestor, and never nominates a bare `python.exe` unless its command line names this
package. These tests check that the dialog does not undo any of that: the real OS is never queried,
`prompt_and_release` (which is terminal-bound and would silently stop nothing under a GUI) is never
called, and `stop` is only reached through `release_camera` with a confirmation that said yes.
"""
from __future__ import annotations

import pytest

from flygym_tracker import camera_lock
from flygym_tracker.gui.camera_lock_dialog import CameraLockDialog

#: A fixture process table -- the real OS is never asked anything in this file.
PROCESSES = [
    {"pid": 1234, "ppid": 1, "name": "Bonsai.exe",
     "cmdline": "Bonsai.exe workflow.bonsai --start --no-editor"},
    {"pid": 2345, "ppid": 1, "name": "MVS.exe", "cmdline": "MVS.exe"},
    {"pid": 3456, "ppid": 1, "name": "python.exe", "cmdline": "python analyse_my_other_data.py"},
]


@pytest.fixture
def holders():
    return camera_lock.find_camera_holders(processes=PROCESSES)


def test_the_fixture_table_nominates_only_the_two_camera_programs(holders):
    """The unrelated python job must never be nominated: ending a scientist's own analysis would be
    a far worse outcome than a camera that stays busy another minute."""
    assert sorted(h.pid for h in holders) == [1234, 2345]


def test_the_headless_bonsai_is_listed_first_and_flagged(holders):
    """It is the one the operator CANNOT deal with themselves -- no window, no taskbar entry -- so
    it is what the prompt should be about."""
    assert holders[0].pid == 1234
    assert holders[0].headless is True


def test_a_confirm_that_says_no_stops_nothing(qapp, holders):
    stopped_pids = []
    dialog = CameraLockDialog(holders, confirm=lambda h: False,
                              stop=lambda pid: stopped_pids.append(pid) or True)
    dialog._on_stop()
    assert stopped_pids == []
    assert dialog.stopped == []
    assert dialog.summary() == "Nothing was stopped."


def test_a_confirm_that_says_yes_stops_exactly_the_nominated_pids(qapp, holders):
    stopped_pids = []
    dialog = CameraLockDialog(holders, confirm=lambda h: True,
                              stop=lambda pid: stopped_pids.append(pid) or True)
    dialog._on_stop()
    assert sorted(stopped_pids) == [1234, 2345]
    assert "PID 1234" in dialog.summary() or "1234" in dialog.summary()


def test_the_confirm_is_shown_the_holders_it_is_being_asked_about(qapp, holders):
    """"Are you sure?" is not a question anyone can answer. The list is the content of the
    decision."""
    seen = []
    CameraLockDialog(holders, confirm=lambda h: seen.append(list(h)) or False,
                     stop=lambda pid: True)._on_stop()
    assert seen and sorted(h.pid for h in seen[0]) == [1234, 2345]


def test_the_body_is_camera_locks_own_report_verbatim(qapp, holders):
    """It already explains, for a non-programmer, what a headless Bonsai is and why the rig looks
    idle while the camera is held. A second wording here would be a second thing to keep in step."""
    from PySide6.QtWidgets import QPlainTextEdit

    dialog = CameraLockDialog(holders, confirm=lambda h: False)
    text = dialog.findChild(QPlainTextEdit).toPlainText()
    assert text == camera_lock.report(holders)
    assert "no window" in text


def test_with_nothing_holding_the_camera_there_is_nothing_to_press(qapp):
    """The dialog is then purely informational, and the report already says what to try instead."""
    dialog = CameraLockDialog([], confirm=lambda h: True, stop=lambda pid: True)
    assert dialog.stop_button.isEnabled() is False
    from PySide6.QtWidgets import QPlainTextEdit

    assert "Nothing was found holding the camera" in dialog.findChild(QPlainTextEdit).toPlainText()


def test_the_dialog_never_calls_the_terminal_bound_prompt(qapp, holders, monkeypatch):
    """`prompt_and_release` uses `input()` and `sys.stdin.isatty()`. Under a GUI it would find no
    terminal, print into nowhere and stop nothing -- while looking like it had asked."""
    def explode(*args, **kwargs):
        raise AssertionError("the GUI called prompt_and_release")

    monkeypatch.setattr(camera_lock, "prompt_and_release", explode)
    dialog = CameraLockDialog(holders, confirm=lambda h: True, stop=lambda pid: True)
    dialog._on_stop()
    assert len(dialog.stopped) == 2


def test_finding_holders_has_no_side_effects_and_never_touches_the_real_os(monkeypatch):
    """`find_camera_holders` is pure observation. If it ever shells out when handed a table, this
    fails rather than the rig meeting a 30-second PowerShell call mid-experiment."""
    def explode(*args, **kwargs):
        raise AssertionError("the process table was queried from the OS")

    monkeypatch.setattr(camera_lock, "_powershell_processes", explode)
    assert len(camera_lock.find_camera_holders(processes=PROCESSES)) == 2


def test_stop_is_only_ever_reached_through_release_camera(qapp, holders, monkeypatch):
    """There must be no path to a kill that skips the confirmation, so the dialog is checked to go
    through the function that REQUIRES one."""
    called = {}

    def fake_release(items, confirm, stop=None):
        called["confirm"] = confirm
        return []

    monkeypatch.setattr(camera_lock, "release_camera", fake_release)
    CameraLockDialog(holders, confirm=lambda h: True, stop=lambda pid: True)._on_stop()
    assert callable(called["confirm"])
