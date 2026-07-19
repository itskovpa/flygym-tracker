"""The desktop app: settings and camera in one window (Stage 1).

NOTHING IS IMPORTED FROM PySide6 AT MODULE SCOPE, here or in `flygym_tracker/__init__.py`. The rig
runs unattended experiments from the CLI, and `import flygym_tracker` must keep working on a
machine where Qt is not installed -- a missing GUI toolkit is not a reason for a headless overnight
run to fail to start. `main()` imports Qt when it is actually asked to open a window, and says
something useful if that fails.
"""
from __future__ import annotations

__all__ = ["main"]

#: The pip name to suggest when Qt is missing. Spelled once, so the message and the install
#: instruction cannot drift.
QT_PACKAGE = "PySide6"

_MISSING_QT = (
    "The FlyGym settings app needs %s, which is not installed on this machine.\n"
    "\n"
    "    python -m pip install %s\n"
    "\n"
    "Everything else keeps working without it: `flygym-tracker run`, `replay`, `noise` and the\n"
    "`settings` command all run from the terminal and need no Qt at all."
) % (QT_PACKAGE, QT_PACKAGE)


def main(argv=None) -> int:
    """Open the app. Returns a process exit code.

    Deliberately a thin re-export of `flygym_tracker.gui.app.main`: importing THIS package must not
    import Qt, so the real entry point lives one module down and is reached only when called.

    THE `try` COVERS THE CALL, NOT JUST THE IMPORT, and that is a real bug rather than a
    belt-and-braces flourish. `app.main` imports PySide6 INSIDE ITSELF -- deliberately, so that
    high-DPI attributes are set before the QApplication exists -- so on a machine with no Qt the
    `from ... import main` line succeeds and the ImportError arrives from `_main(argv)` instead.
    Guarding only the import printed a raw traceback at a scientist, which is the exact outcome the
    message below exists to prevent. Caught by
    `tests/test_settings_model_isolation.py::test_asking_the_gui_to_open_without_qt_...`.
    """
    import sys

    try:
        from flygym_tracker.gui.app import main as _main

        return _main(argv)
    except ImportError as exc:  # Qt missing, or a broken install of it
        if not _mentions_qt(exc):
            raise                    # some other import failed; do not disguise it as "no Qt"
        print(_MISSING_QT, file=sys.stderr)
        print("\n(underlying import error: %s)" % exc, file=sys.stderr)
        return 2


def _mentions_qt(exc: ImportError) -> bool:
    """Is this ImportError about Qt, or about something else that must not be mislabelled?

    An unrelated broken import inside the app reported as "install PySide6" would send someone off
    installing a package they already have, which is worse than the traceback.
    """
    name = (getattr(exc, "name", "") or "").split(".")[0]
    return name == QT_PACKAGE or QT_PACKAGE.lower() in str(exc).lower()
