"""Tests for finding -- and refusing to over-eagerly kill -- whatever holds the camera.

NOTHING HERE TERMINATES A REAL PROCESS. The process table is injected and `stop` is a stub, so
the kill path is exercised without any possibility of ending something on the machine running
the suite.

The safety properties matter more than the feature: this is the one module that can destroy a
scientist's running work, so most of what is asserted here is what it must NOT do.
"""
from __future__ import annotations

import os

import pytest

from flygym_tracker import camera_lock as CL


def _proc(pid, name, cmdline="", ppid=1):
    return {"pid": pid, "ppid": ppid, "name": name, "cmdline": cmdline}


#: A realistic table: the invisible Bonsai that actually caused this, plus innocent bystanders.
TABLE = [
    _proc(4, "System"),
    _proc(100, "explorer.exe", "C:\\Windows\\explorer.exe"),
    _proc(8624, "Bonsai.exe",
          '"C:\\...\\Bonsai2.9\\Bonsai.exe" live_test.bonsai --start --no-editor'),
    _proc(28344, "Bonsai.exe",
          '"C:\\...\\Bonsai.exe" --start --no-editor --no-boot "C:\\...\\live_test.bonsai"'),
    _proc(555, "MVS.exe", '"C:\\Program Files (x86)\\MVS\\MVS.exe"'),
    _proc(777, "python.exe", 'python.exe -m flygym_tracker.cli select-vials --out calib_faces'),
    _proc(888, "python.exe", 'python.exe train_my_network.py --epochs 400'),
    _proc(999, "chrome.exe", "chrome.exe"),
]


# =========================================================================================
# What it finds
# =========================================================================================
def test_it_finds_the_invisible_headless_bonsai():
    """The case this exists for: a Bonsai with no window, so nothing on screen to close."""
    holders = CL.find_camera_holders(processes=TABLE)
    pids = [h.pid for h in holders]
    assert 8624 in pids and 28344 in pids
    headless = [h for h in holders if h.headless]
    assert {h.pid for h in headless} == {8624, 28344}
    assert any("no window" in r for r in next(h for h in holders if h.pid == 8624).reasons)


def test_headless_holders_are_listed_first():
    """They are the ones the operator cannot deal with by hand, so they lead the report."""
    holders = CL.find_camera_holders(processes=TABLE)
    assert holders[0].headless


def test_it_finds_the_mvs_viewer_and_our_own_leftovers():
    pids = [h.pid for h in CL.find_camera_holders(processes=TABLE)]
    assert 555 in pids          # MVS Viewer
    assert 777 in pids          # a flygym-tracker session that did not exit


# =========================================================================================
# What it must NEVER touch
# =========================================================================================
def test_an_unrelated_python_job_is_never_nominated():
    """Ending a scientist's training run to free a camera would be far worse than a busy camera."""
    pids = [h.pid for h in CL.find_camera_holders(processes=TABLE)]
    assert 888 not in pids


@pytest.mark.parametrize("pid", [4, 100, 999])
def test_unrelated_programs_are_never_nominated(pid):
    assert pid not in [h.pid for h in CL.find_camera_holders(processes=TABLE)]


def test_it_never_nominates_itself():
    table = TABLE + [_proc(os.getpid(), "python.exe",
                           "python.exe -m flygym_tracker.cli free-camera")]
    assert os.getpid() not in [h.pid for h in CL.find_camera_holders(processes=table)]


def test_it_never_nominates_the_shell_it_was_launched_from():
    """run.bat's cmd.exe is an ancestor; offering to kill it would end the session mid-answer."""
    me, shell, grandparent = os.getpid(), 4242, 4243
    table = [
        _proc(me, "python.exe", "python.exe -m flygym_tracker.cli free-camera", ppid=shell),
        _proc(shell, "Bonsai.exe", "Bonsai.exe --start --no-editor", ppid=grandparent),
        _proc(grandparent, "Bonsai.exe", "Bonsai.exe --start", ppid=0),
        _proc(8624, "Bonsai.exe", "Bonsai.exe live_test.bonsai --start --no-editor"),
    ]
    pids = [h.pid for h in CL.find_camera_holders(processes=table)]
    assert shell not in pids and grandparent not in pids
    assert pids == [8624]       # the genuinely unrelated one is still found


def test_extra_excluded_pids_are_honoured():
    pids = [h.pid for h in CL.find_camera_holders(processes=TABLE, exclude_pids=[8624, 555])]
    assert 8624 not in pids and 555 not in pids


# =========================================================================================
# Stopping: only ever with an explicit yes
# =========================================================================================
def test_nothing_is_stopped_without_confirmation():
    killed = []
    holders = CL.find_camera_holders(processes=TABLE)
    assert CL.release_camera(holders, confirm=lambda _h: False, stop=killed.append) == []
    assert killed == []


def test_confirming_stops_exactly_what_was_shown():
    killed = []

    def stop(pid):
        killed.append(pid)
        return True

    holders = CL.find_camera_holders(processes=TABLE)
    stopped = CL.release_camera(holders, confirm=lambda _h: True, stop=stop)
    assert killed == [h.pid for h in holders]
    assert [h.pid for h in stopped] == [h.pid for h in holders]


def test_a_process_that_could_not_be_stopped_is_not_reported_as_stopped():
    holders = CL.find_camera_holders(processes=TABLE)
    stopped = CL.release_camera(holders, confirm=lambda _h: True, stop=lambda _p: False)
    assert stopped == []


def test_confirm_is_shown_the_actual_holders():
    seen = {}
    holders = CL.find_camera_holders(processes=TABLE)
    CL.release_camera(holders, confirm=lambda h: seen.update(n=len(h)) or False)
    assert seen["n"] == len(holders)


def test_an_empty_list_never_asks_anything():
    CL.release_camera([], confirm=lambda _h: pytest.fail("must not ask about nothing"))


# =========================================================================================
# The prompt
# =========================================================================================
def test_prompt_defaults_to_no(monkeypatch):
    """Opt-in, not opt-out: ENTER must not end anything."""
    monkeypatch.setattr(CL, "_powershell_processes", lambda: TABLE)
    killed = []
    assert CL.prompt_and_release(input_fn=lambda _p: "", stop=killed.append) == 0
    assert killed == []


def test_prompt_stops_them_on_an_explicit_yes(monkeypatch):
    monkeypatch.setattr(CL, "_powershell_processes", lambda: TABLE)
    killed = []

    def stop(pid):
        killed.append(pid)
        return True

    n = CL.prompt_and_release(input_fn=lambda _p: "y", stop=stop)
    assert n == 4 and sorted(killed) == [555, 777, 8624, 28344]


def test_prompt_stops_nothing_when_there_is_no_terminal(monkeypatch):
    """An unattended run must not silently start terminating things."""
    monkeypatch.setattr(CL, "_powershell_processes", lambda: TABLE)
    monkeypatch.setattr(CL.sys, "stdin", None)
    killed = []
    assert CL.prompt_and_release(stop=killed.append) == 0
    assert killed == []


@pytest.mark.parametrize("boom", [EOFError, KeyboardInterrupt])
def test_an_interrupted_prompt_stops_nothing(monkeypatch, boom):
    monkeypatch.setattr(CL, "_powershell_processes", lambda: TABLE)
    killed = []

    def ask(_p):
        raise boom()

    assert CL.prompt_and_release(input_fn=ask, stop=killed.append) == 0
    assert killed == []


# =========================================================================================
# Reporting + error recognition
# =========================================================================================
def test_the_report_names_pids_and_explains_the_invisible_ones():
    text = CL.report(CL.find_camera_holders(processes=TABLE))
    assert "8624" in text and "Bonsai" in text
    assert "no window" in text
    assert "MVS Viewer" in text


def test_the_report_says_so_when_nothing_holds_the_camera():
    text = CL.report([])
    assert "Nothing was found" in text
    assert "unplug" in text.lower()          # the remaining thing to try


@pytest.mark.parametrize("message", [
    "MV_CC_OpenDevice failed (ret=0x80000203)",
    "camera may already be in use by another application",
    "MV_E_ACCESS_DENIED",
])
def test_the_busy_error_is_recognised(message):
    assert CL.looks_like_busy_error(message)


@pytest.mark.parametrize("message", [
    "MV_CC_OpenDevice failed (ret=0x80000001)", "no camera found", "",
])
def test_other_errors_are_not_mistaken_for_a_busy_camera(message):
    assert not CL.looks_like_busy_error(message)


def test_a_process_table_that_cannot_be_read_yields_no_nominations(monkeypatch):
    """If PowerShell is missing or blocked, report nothing rather than guessing."""
    monkeypatch.setattr(CL, "_powershell_processes", lambda: [])
    assert CL.find_camera_holders() == []


def test_stop_process_treats_an_already_gone_process_as_success(monkeypatch):
    class Done:
        returncode, stdout, stderr = 128, "", 'ERROR: The process "123" not found.'

    monkeypatch.setattr(CL.subprocess, "run", lambda *a, **k: Done())
    assert CL.stop_process(123) is True


def test_stop_process_reports_failure_when_the_kill_is_refused(monkeypatch):
    class Denied:
        returncode, stdout, stderr = 1, "", "Access is denied."

    monkeypatch.setattr(CL.subprocess, "run", lambda *a, **k: Denied())
    assert CL.stop_process(123) is False
