"""Always-on crash logging, so a crash on any machine leaves a report to send back.

WHAT THIS IS FOR, AND WHY IT IS NOT THE DEBUG HARNESS. `packaging/debug_run.py` proved what it
takes to catch these crashes: the app closes itself with a native access violation, a windowed
build prints nothing, and only `faulthandler` turns that into a readable per-thread stack. But a
harness is a separate script a customer would have to know to run. This is the same capture, ON in
EVERY build, writing where the operator can find it -- so "it crashed on the other PC" becomes "here
is the file" instead of a fresh debugging trip.

WHERE IT WRITES. `<user data>/logs/session_<stamp>.log`, beside the results, never inside the
install (which is read-only). Each launch is its own file; the last few are kept and the rest are
pruned, so a machine that runs for months does not accumulate a heap of them.

WHAT EACH LOG CONTAINS.
  * a header naming the version, the OS, the CPU count, the MVS SDK and camera runtime it found --
    which answers "what is different about that machine" before a single crash;
  * `faulthandler`'s dump if the process dies of a native fault (the access-violation case);
  * any unhandled Python exception, on the main thread or a worker;
  * a one-line "clean shutdown" marker at the end -- whose ABSENCE on the previous log is how the
    next launch knows the last run crashed rather than being closed.

`collect_report()` zips the logs together with a fresh system report into one file to attach.
"""
from __future__ import annotations

import atexit
import faulthandler
import os
import platform
import sys
import threading
import traceback
from datetime import datetime
from typing import Optional

#: How many session logs to keep. Enough to cover "it crashed, I reopened it, then collected the
#: report" plus a couple more, without letting a long-lived install hoard them.
KEEP_SESSIONS = 12

_log = None            # the open session-log file, kept for the process lifetime
_log_path = None
_installed = False


def log_dir():
    from flygym_tracker import paths

    return paths.ensure_user_data_root() / "logs"


def install(app_version: str = "") -> Optional[str]:
    """Turn on crash logging for this process. Returns the log path, or None if it could not start.

    Idempotent and NEVER RAISES: diagnostics failing to start is not a reason to stop the app from
    starting. A machine whose data folder cannot be written simply gets no log -- which is exactly
    the machine that also cannot save results, so the operator has a bigger message coming anyway.
    """
    global _log, _log_path, _installed
    if _installed:
        return _log_path
    _installed = True
    try:
        directory = log_dir()
        directory.mkdir(parents=True, exist_ok=True)
        _prune(directory)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _log_path = str(directory / ("session_%s.log" % stamp))
        _log = open(_log_path, "a", buffering=1, encoding="utf-8")   # line-buffered: survives a kill
    except Exception:
        _log = None
        return None

    _write_header(app_version)

    # THE HEADLINE TOOL. On Windows this installs a handler for EXCEPTION_ACCESS_VIOLATION as well
    # as the POSIX fatal signals, so the 0xc0000005 that closes the app silently instead prints
    # "Windows fatal exception: access violation" and every thread's Python stack into this file.
    try:
        faulthandler.enable(file=_log, all_threads=True)
    except Exception:
        pass

    _install_python_hooks()

    # THE CLEAN-SHUTDOWN MARKER. If the process dies of a native fault this line never runs, so its
    # absence on a past log is how `previous_session_crashed` knows the last run did not exit
    # cleanly -- which is what lets the next launch offer to send the report.
    atexit.register(_mark_clean_exit)
    return _log_path


def write(message: str) -> None:
    """Append a timestamped line to the session log. Safe to call before `install` (a no-op then)."""
    if _log is None:
        return
    try:
        _log.write("%s  %s\n" % (datetime.now().strftime("%H:%M:%S.%f")[:-3], message))
    except Exception:
        pass


def _write_header(app_version: str) -> None:
    from flygym_tracker import paths

    lines = [
        "=" * 78,
        "FlyGym Tracker session log",
        "version        %s" % (app_version or "unknown"),
        "started        %s" % datetime.now().isoformat(),
        "frozen build   %s" % paths.is_frozen(),
        "python         %s" % sys.version.split()[0],
        "platform       %s" % platform.platform(),
        "machine        %s" % platform.machine(),
        "cpu count      %s" % (os.cpu_count() or "?"),
        "data folder    %s" % paths.user_data_root(),
    ]
    lines.extend(_camera_environment())
    lines.append("=" * 78)
    for line in lines:
        write(line)


def _camera_environment() -> list:
    """What the machine offers the camera -- the answer to "why does it work here and not there"."""
    out = ["-- camera environment --"]
    try:
        from flygym_tracker.frame_source import mvs_sdk_candidates

        found = False
        for path in mvs_sdk_candidates():
            here = os.path.isfile(os.path.join(path, "MvCameraControl_class.py"))
            out.append("  MVS SDK %s  %s" % ("FOUND " if here else "absent", path))
            found = found or here
        if not found:
            out.append("  MVS SDK: not found in any candidate -> rig camera cannot open")
        override = os.environ.get("MVS_PYTHON_SDK")
        if override:
            out.append("  MVS_PYTHON_SDK=%s" % override)
    except Exception as exc:
        out.append("  (could not probe the MVS SDK: %r)" % exc)
    return out


def _install_python_hooks() -> None:
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        write("UNHANDLED EXCEPTION:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
        try:
            previous(exc_type, exc, tb)
        except Exception:
            pass

    sys.excepthook = hook

    if hasattr(threading, "excepthook"):
        def thook(args):
            name = args.thread.name if args.thread else "?"
            write("UNHANDLED EXCEPTION in thread %s:\n%s" % (
                name, "".join(traceback.format_exception(
                    args.exc_type, args.exc_value, args.exc_traceback))))
        threading.excepthook = thook


def _mark_clean_exit() -> None:
    write("clean shutdown")


def _prune(directory) -> None:
    try:
        logs = sorted(directory.glob("session_*.log"))
        for stale in logs[:-KEEP_SESSIONS]:
            try:
                stale.unlink()
            except OSError:
                pass
    except Exception:
        pass


# =============================================================================================
# After a crash: did the last run die, and can the operator hand us one file that says why
# =============================================================================================
def previous_session_crashed() -> Optional[str]:
    """The path of the most recent PAST log that has no clean-shutdown marker, or None.

    "Most recent past" excludes this session's own log, which of course has no marker yet.
    """
    try:
        logs = sorted(log_dir().glob("session_*.log"))
    except Exception:
        return None
    others = [p for p in logs if str(p) != (_log_path or "")]
    if not others:
        return None
    last = others[-1]
    try:
        text = last.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # A native fault leaves the faulthandler dump but never the marker; a clean close writes it.
    if "clean shutdown" in text:
        return None
    return str(last)


def collect_report(destination=None) -> Optional[str]:
    """Zip every session log plus a fresh system report into ONE file to send. Returns its path.

    The whole point is that the operator attaches a single file rather than hunting for logs, so
    the default lands it on the Desktop where they will find it, falling back to the data folder.
    """
    import zipfile

    from flygym_tracker import paths

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = "flygym-diagnostics_%s.zip" % stamp
    if destination is None:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base = desktop if os.path.isdir(desktop) else str(paths.ensure_user_data_root())
        destination = os.path.join(base, name)
    try:
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("system_report.txt", _system_report())
            for log in sorted(log_dir().glob("session_*.log")):
                try:
                    bundle.write(str(log), arcname="logs/" + log.name)
                except OSError:
                    continue
        return destination
    except Exception:
        return None


def _system_report() -> str:
    from flygym_tracker import paths

    lines = [
        "FlyGym Tracker diagnostics",
        "collected      %s" % datetime.now().isoformat(),
        "frozen build   %s" % paths.is_frozen(),
        "python         %s" % sys.version.split()[0],
        "platform       %s" % platform.platform(),
        "machine        %s" % platform.machine(),
        "cpu count      %s" % (os.cpu_count() or "?"),
        "executable     %s" % sys.executable,
        "data folder    %s" % paths.user_data_root(),
    ]
    try:
        from flygym_tracker import __version__

        lines.insert(1, "version        %s" % __version__)
    except Exception:
        pass
    lines.extend(_camera_environment())
    # Enumerate what is attached RIGHT NOW -- rig camera present or not, webcams, and any error.
    try:
        from flygym_tracker.frame_source import list_cameras_with_error

        cameras, error = list_cameras_with_error(include_uvc=False)
        lines.append("-- cameras seen now --")
        for camera in cameras:
            lines.append("  %s" % camera.label)
        if not cameras:
            lines.append("  (no rig camera detected)")
        if error:
            lines.append("  rig enumeration error: %s" % str(error).splitlines()[0])
    except Exception as exc:
        lines.append("  (could not enumerate cameras: %r)" % exc)
    return "\n".join(lines) + "\n"
