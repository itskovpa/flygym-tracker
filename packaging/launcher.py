"""Entry point for the built app: start the window, and if it cannot start, SAY SO.

WHY THIS IS NOT JUST `from flygym_tracker.gui import main`. The build is windowed -- no console --
because a console flashing behind a scientific instrument's window looks broken. The cost of that
is that anything failing before Qt is up has nowhere to print: the customer double-clicks the icon,
nothing happens, and there is no message, no log and no traceback. That is the single worst support
case a shipped application can have, and it is the one that a missing DLL or an unreadable config
produces.

So the whole startup sits inside a guard that writes a log next to the user's own data and puts a
native message box on screen naming the log. A message box via `ctypes` rather than Qt, because the
failure being reported may well be that Qt did not load.
"""
from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime


def _crash_log_path() -> str:
    """Beside the operator's own files, not inside the install (which is read-only)."""
    try:
        from flygym_tracker import paths

        return str(paths.ensure_user_data_root() / "startup-error.log")
    except Exception:
        return os.path.join(os.path.expanduser("~"), "flygym-tracker-startup-error.log")


def _report(exc: BaseException) -> None:
    detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log = _crash_log_path()
    try:
        with open(log, "a", encoding="utf-8") as f:
            f.write("\n%s  FlyGym Tracker failed to start\n%s\n" % (datetime.now().isoformat(),
                                                                    detail))
    except OSError:
        log = "(the log could not be written either)"

    message = (
        "FlyGym Tracker could not start.\n\n"
        "%s: %s\n\n"
        "The full details are in:\n%s\n\n"
        "If this mentions MvImport or MVS, the HikRobot MVS software is not installed -- the "
        "camera needs it, and it is a separate download. Everything else in the program works "
        "without it, including replaying a recording."
        % (type(exc).__name__, exc, log))
    try:
        import ctypes

        # 0x10 = MB_ICONERROR. Via ctypes, not Qt: the thing that failed may BE Qt.
        ctypes.windll.user32.MessageBoxW(None, message, "FlyGym Tracker", 0x10)
    except Exception:
        sys.stderr.write(message + "\n")


def main() -> int:
    try:
        from flygym_tracker.gui import main as gui_main

        return int(gui_main(sys.argv[1:]) or 0)
    except BaseException as exc:            # noqa: BLE001 - the whole point is that nothing escapes
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        _report(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
