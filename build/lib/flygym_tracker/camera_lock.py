"""Find -- and, on request, stop -- whatever else is holding the USB3 Vision camera.

USB3 Vision access is EXCLUSIVE: exactly one process may have the camera open, and the second
one gets ``MV_CC_OpenDevice failed (ret=0x80000203)``. That error names no culprit, which is the
whole problem, because the usual culprit is INVISIBLE:

    Bonsai.exe ... --start --no-editor        <- headless, no window, no taskbar entry

A leftover headless Bonsai (or a crashed selector, or the MVS Viewer minimised somewhere) keeps
the rig hostage with nothing on screen to close. This module answers "who has it?" and offers to
end them.

SAFETY. Killing processes is not undoable, so this module is deliberately conservative:

  * it only ever NOMINATES processes -- `find_camera_holders` has no side effects at all;
  * it never nominates this process, its parent, or any ancestor (the launching shell, run.bat);
  * it matches a small, explicit list of programs known to take this camera, and reports the full
    command line of each so the operator can recognise their own work before agreeing;
  * `release_camera` requires an explicit confirmation callback -- there is no "kill everything"
    path that can be reached by accident.

It never kills a bare `python.exe`: only one whose command line actually mentions this package.
Terminating an unrelated Python job the scientist had running would be a far worse outcome than
a camera that stays busy for another minute.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

#: Programs known to open this rig's camera, and how to describe them to a non-programmer.
#: Matched against the executable name, case-insensitively.
KNOWN_HOLDERS: Dict[str, str] = {
    "mvs.exe": "the MVS Viewer",
    "mvsviewer.exe": "the MVS Viewer",
    "bonsai.exe": "a Bonsai workflow",
    "bonsai64.exe": "a Bonsai workflow",
}
#: A Python process is only ever nominated if its command line mentions one of these -- i.e. it
#: is one of OUR OWN leftovers, not somebody's unrelated analysis script.
PYTHON_MARKERS = ("flygym_tracker", "flygym-tracker", "select-vials", "select_vials_live")
PYTHON_NAMES = ("python.exe", "pythonw.exe", "py.exe")

#: The SDK's "device already in use" return code, as it appears in the error text.
BUSY_CODES = ("0x80000203", "MV_E_ACCESS_DENIED")


@dataclass
class CameraHolder:
    """A process that plausibly has the camera open. Nominated, never yet touched."""
    pid: int
    name: str
    cmdline: str = ""
    what: str = ""                      # human description, e.g. "a Bonsai workflow"
    headless: bool = False              # runs with no window -> the operator cannot close it
    reasons: List[str] = field(default_factory=list)

    def describe(self) -> str:
        """One line an operator can act on without knowing what a PID is."""
        bits = ["PID %d  %s" % (self.pid, self.what or self.name)]
        if self.headless:
            bits.append("[no window - cannot be closed by hand]")
        line = "  ".join(bits)
        if self.cmdline:
            line += "\n        " + _shorten(self.cmdline, 150)
        return line


def _shorten(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def looks_like_busy_error(error: object) -> bool:
    """True if `error` is the SDK's exclusive-access complaint (and so worth offering a fix for)."""
    text = str(error)
    return any(code.lower() in text.lower() for code in BUSY_CODES) or (
        "already" in text.lower() and "use" in text.lower())


# ------------------------------------------------------------------------------------------
# Process enumeration
# ------------------------------------------------------------------------------------------
def _powershell_processes() -> List[dict]:
    """Every process as ``{pid, ppid, name, cmdline}``, via CIM. Empty list if unavailable.

    PowerShell rather than a third-party module: the rig install is numpy/opencv/pandas/pyyaml/
    openpyxl and adding `psutil` for one diagnostic is not worth a new dependency on a machine
    that runs unattended experiments.
    """
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | "
        "ConvertTo-Json -Compress -Depth 2"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0 or not out.stdout.strip():
        return []
    try:
        data = json.loads(out.stdout)
    except ValueError:
        return []
    if isinstance(data, dict):          # a single match is not wrapped in a list
        data = [data]
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            rows.append({
                "pid": int(item.get("ProcessId") or 0),
                "ppid": int(item.get("ParentProcessId") or 0),
                "name": str(item.get("Name") or ""),
                "cmdline": str(item.get("CommandLine") or ""),
            })
        except (TypeError, ValueError):
            continue
    return rows


def _ancestors(pid: int, by_pid: Dict[int, dict], limit: int = 32) -> set:
    """`pid` and every parent above it -- the processes that must never be nominated.

    Without this the tool could offer to kill the shell it is running in (run.bat), or itself.
    """
    seen, current = set(), int(pid)
    while current and current not in seen and len(seen) < limit:
        seen.add(current)
        row = by_pid.get(current)
        if not row:
            break
        current = int(row.get("ppid") or 0)
    return seen


def find_camera_holders(exclude_pids: Sequence[int] = (),
                        processes: Optional[List[dict]] = None) -> List[CameraHolder]:
    """Processes that plausibly hold the camera. PURE OBSERVATION -- nothing is stopped here.

    Args:
        exclude_pids: extra pids to leave alone, on top of this process and its ancestors.
        processes: injected process table (for tests); queried from the OS when omitted.

    Returns:
        Nominations, headless ones first -- those are the ones the operator cannot deal with
        themselves, so they are what the prompt should be about.
    """
    rows = _powershell_processes() if processes is None else list(processes)
    by_pid = {int(r.get("pid") or 0): r for r in rows}
    protected = set(int(p) for p in exclude_pids)
    protected |= _ancestors(os.getpid(), by_pid)

    holders: List[CameraHolder] = []
    for row in rows:
        pid, name = int(row.get("pid") or 0), str(row.get("name") or "")
        cmdline, lowered = str(row.get("cmdline") or ""), name.lower()
        if pid <= 0 or pid in protected:
            continue

        if lowered in KNOWN_HOLDERS:
            headless = "--no-editor" in cmdline or "--start" in cmdline
            reasons = ["%s can open this camera" % KNOWN_HOLDERS[lowered]]
            if headless:
                reasons.append("started headless (--no-editor/--start): it has no window to close")
            holders.append(CameraHolder(pid=pid, name=name, cmdline=cmdline,
                                        what=KNOWN_HOLDERS[lowered], headless=headless,
                                        reasons=reasons))
        elif lowered in PYTHON_NAMES and any(m in cmdline for m in PYTHON_MARKERS):
            holders.append(CameraHolder(
                pid=pid, name=name, cmdline=cmdline, what="a leftover flygym-tracker session",
                headless=False,
                reasons=["a previous run of this software that did not shut down cleanly"]))

    holders.sort(key=lambda h: (not h.headless, h.pid))
    return holders


# ------------------------------------------------------------------------------------------
# Stopping them
# ------------------------------------------------------------------------------------------
def stop_process(pid: int) -> bool:
    """End one process. True if it is gone afterwards."""
    try:
        out = subprocess.run(["taskkill", "/PID", str(int(pid)), "/F"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    if out.returncode == 0:
        return True
    # 128 == "no such process": it exited on its own between listing and killing, which is the
    # outcome we wanted anyway.
    return out.returncode == 128 or "not found" in (out.stdout + out.stderr).lower()


def release_camera(holders: Sequence[CameraHolder],
                   confirm: Callable[[Sequence[CameraHolder]], bool],
                   stop: Callable[[int], bool] = stop_process) -> List[CameraHolder]:
    """Offer to stop `holders`; stop them only if `confirm` says yes. Returns those stopped.

    `confirm` is required and has no default: there must be no way to reach a kill from a plain
    call. `stop` is injected so tests never terminate anything real.
    """
    if not holders or not confirm(holders):
        return []
    return [h for h in holders if stop(h.pid)]


def report(holders: Sequence[CameraHolder]) -> str:
    """The block of text shown before asking to stop anything."""
    if not holders:
        return ("Nothing was found holding the camera.\n"
                "If it is still busy, the holder may be running as another Windows user, or the\n"
                "camera may need unplugging and plugging back in.")
    lines = ["These programs can hold the camera open:", ""]
    for holder in holders:
        lines.append(holder.describe())
        for reason in holder.reasons:
            lines.append("        - " + reason)
        lines.append("")
    if any(h.headless for h in holders):
        lines.append("The ones marked [no window] are running invisibly - there is nothing on")
        lines.append("screen to close, which is why the camera looks free but is not.")
        lines.append("")
    return "\n".join(lines)


def prompt_and_release(exclude_pids: Sequence[int] = (),
                       input_fn: Optional[Callable[[str], str]] = None,
                       stop: Callable[[int], bool] = stop_process) -> int:
    """Show what holds the camera, ask, and stop them. Returns how many were stopped.

    Answering is deliberately opt-IN (``[y/N]``): ending a process the scientist is using would
    be worse than a camera that stays busy. With no terminal to ask on, nothing is stopped.
    """
    holders = find_camera_holders(exclude_pids=exclude_pids)
    print(report(holders))
    if not holders:
        return 0

    def confirm(items: Sequence[CameraHolder]) -> bool:
        ask = input_fn if input_fn is not None else input
        if input_fn is None and not (sys.stdin is not None and sys.stdin.isatty()):
            print("(no terminal to confirm on - nothing was stopped)")
            return False
        try:
            answer = ask("Stop %d program(s) and free the camera? [y/N]: " % len(items))
        except (EOFError, KeyboardInterrupt):
            print("(cancelled - nothing was stopped)")
            return False
        return str(answer).strip().lower().startswith("y")

    stopped = release_camera(holders, confirm=confirm, stop=stop)
    if stopped:
        print("\nStopped %d program(s):" % len(stopped))
        for holder in stopped:
            print("  PID %d  %s" % (holder.pid, holder.what or holder.name))
        print("\nThe camera should be free now.")
    elif holders:
        print("\nNothing was stopped.")
    return len(stopped)
