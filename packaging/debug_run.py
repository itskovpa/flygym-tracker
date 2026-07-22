"""Launch the app with everything that turns a silent native crash into a readable report.

WHY THIS EXISTS. The app has been closing itself "after some time", leaving nothing behind but a
Windows event-log line naming `MVGigEVisionSDK.dll_unloaded` and an access violation. That is a
NATIVE crash -- a background thread inside the HikRobot SDK running code in a DLL that has already
been unloaded -- and an ordinary `try/except` cannot see it, because control never returns to
Python. `faulthandler` is the one tool that does: on Windows it catches the access violation and
prints the Python stack of EVERY thread at the instant of the fault, which is what says which
thread was in which SDK call when the DLL went away.

Run it INSTEAD of the normal launcher while debugging:

    python packaging/debug_run.py

Everything it captures goes to `debug_crash.log` next to this file AND to the console.
"""
from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "debug_crash.log")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

_log = open(LOG_PATH, "a", buffering=1, encoding="utf-8")   # line-buffered: survives a hard crash


def stamp(msg: str) -> None:
    line = "%s  %s" % (datetime.now().strftime("%H:%M:%S.%f")[:-3], msg)
    _log.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    stamp("=" * 78)
    stamp("debug_run start  pid=%d  python=%s" % (os.getpid(), sys.version.split()[0]))

    # THE HEADLINE TOOL. On Windows this installs a handler for EXCEPTION_ACCESS_VIOLATION as well
    # as the POSIX fatal signals, so the 0xc0000005 that has been killing the app silently now
    # prints "Windows fatal exception: access violation" followed by every thread's Python stack.
    faulthandler.enable(file=_log, all_threads=True)

    # A HEARTBEAT of all-thread stacks. If the crash is preceded by a hang, or if the fatal handler
    # cannot run (a truly wedged process), the LAST periodic dump before the app dies still shows
    # what each thread was doing seconds earlier. 15 s is frequent enough to catch the approach and
    # rare enough not to bury the log.
    faulthandler.dump_traceback_later(15, repeat=True, file=_log)

    # PYTHON-LEVEL failures too, in case the crash is not native after all: an unhandled exception
    # on any thread, including the camera worker and the tracking pool.
    def hook(exc_type, exc, tb):
        stamp("UNHANDLED EXCEPTION:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))

    sys.excepthook = hook
    if hasattr(threading, "excepthook"):
        def thook(args):
            stamp("UNHANDLED THREAD EXCEPTION in %s:\n%s" % (
                args.thread.name if args.thread else "?",
                "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))))
        threading.excepthook = thook

    # Qt's own warnings and fatals, which otherwise vanish -- a "QObject::killTimer: Timers cannot
    # be stopped from another thread" here would be a smoking gun for a cross-thread teardown.
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def qt_handler(mode, context, message):
            kind = {QtMsgType.QtDebugMsg: "Qt.debug", QtMsgType.QtInfoMsg: "Qt.info",
                    QtMsgType.QtWarningMsg: "Qt.WARN", QtMsgType.QtCriticalMsg: "Qt.CRIT",
                    QtMsgType.QtFatalMsg: "Qt.FATAL"}.get(mode, "Qt")
            stamp("%s: %s" % (kind, message))

        qInstallMessageHandler(qt_handler)
    except Exception as exc:
        stamp("could not install the Qt message handler: %r" % exc)

    stamp("launching the window; interact normally and let it run until it crashes")
    try:
        from flygym_tracker.gui import main as gui_main

        code = int(gui_main(sys.argv[1:]) or 0)
        stamp("the window closed normally with exit code %d" % code)
        return code
    except BaseException as exc:                 # noqa: BLE001
        stamp("main() raised: %r\n%s" % (exc, traceback.format_exc()))
        return 1
    finally:
        faulthandler.cancel_dump_traceback_later()   # cancel the repeating heartbeat timer
        stamp("debug_run end")
        _log.flush()
        time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(main())
