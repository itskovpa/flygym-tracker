"""Bootstrap: a QApplication, the saved state, a config, and one window.

TWO THINGS HAPPEN IN A FIXED ORDER HERE, AND BOTH ARE LOAD-BEARING.

1. HIGH-DPI ATTRIBUTES BEFORE THE QApplication IS CONSTRUCTED. After construction they are
   ignored, silently.

2. THE cv2 TOOLS ARE LAUNCHED AS SUBPROCESSES, NEVER IN-PROCESS. MEASURED on this machine:
   process DPI awareness is 0 (UNAWARE) before `QApplication(...)` and 2 (PER_MONITOR_AWARE)
   after -- with the real windows platform plugin; under `offscreen` it stays 0, which is why no
   test can catch this. `live_vial_selector.screen_view_limit` depends on the process staying
   UNAWARE: it deliberately does not call `SetProcessDPIAware`, because `SM_CXFULLSCREEN` then
   reports the desktop in the same coordinate space the OpenCV window is laid out in. Its docstring
   documents the regression it exists to fix, on a 2880x1800 panel at 200% scaling -- this
   machine's configuration -- where the bottom rows of every frame fell below the screen edge:
   "not visible, and impossible to click... exactly the part of the tube the operator most needs to
   enclose."

   Constructing a QApplication in the same process makes that regression twice as bad in the ROI
   editor, which is Stage 2's core tool and stays in use from the CLI regardless. So the rule is
   decided NOW, while it is one line: the remaining cv2 tools are started with
   `python -m flygym_tracker.cli ...` in a child process. Cheap today, expensive after the ROI
   editor is wired in. It also sidesteps any future cv2/Qt symbol clash.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from flygym_tracker.config import DEFAULT_CONFIG_PATH, ensure_local_config, load_config
from flygym_tracker.gui import gui_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flygym-tracker gui",
        description="Settings and camera for the FlyGym v2 rig, in one window.")
    parser.add_argument("--config", default=None,
                        help="Config YAML to edit (default: whatever was open last time).")
    parser.add_argument("--state-dir", default=None,
                        help="Where gui_state.json lives (default: the repo root).")
    return parser


def _repo_root() -> str:
    """The folder `gui_state.json` sits in: the repo, not the user profile.

    Resolved from this file (``<repo>/src/flygym_tracker/gui/app.py`` -> three parents up) rather
    than from the working directory, because a shortcut on the desktop starts the app in whatever
    folder Windows feels like.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def camera_factory_from_config(config):
    """A callable that builds the `HikCameraSource` this config asks for. Touches no hardware.

    Deliberately the SAME construction the CLI does (`cli._camera_source_from_config`), imported
    lazily so that importing this module does not drag cv2 in behind the CLI.
    """
    def factory():
        from flygym_tracker.cli import _camera_source_from_config

        return _camera_source_from_config(config)

    return factory


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    # BEFORE the QApplication exists -- see the module docstring.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from flygym_tracker.gui import theme

    app.setApplicationName("FlyGym v2 Tracker")
    # FONT FIRST, THEN STYLESHEET, and both through one call so a second entry point cannot do
    # only half of it. The font carries a real POINT size: the old global `font-size: 13px` left
    # every widget reporting `pointSize() == -1`, which is the precondition for the
    # `QFont::setPointSize: Point size <= 0 (-1)` line the operator saw on every launch.
    theme.apply(app)

    root = args.state_dir or _repo_root()
    state = gui_state.load_state(root)
    config_path = args.config or state.get("config_path") or ""
    if not args.config:
        # THIS MACHINE'S OWN config file, created empty on first launch. Only when the operator did
        # not name one explicitly: `--config` is someone saying exactly which file they mean, and
        # silently substituting a different one is how a run gets measured with values nobody
        # chose. See `config.ensure_local_config`.
        config_path = ensure_local_config(config_path or None)

    config, error = _load(config_path)
    if config is None:
        # Fall back to the packaged defaults rather than refusing to start. An operator whose
        # config file has a typo in it still needs a window to fix it FROM, and a dialog on a black
        # screen with no application behind it is where a support call starts.
        config, _ = _load(str(DEFAULT_CONFIG_PATH))
        QMessageBox.warning(None, "That config could not be loaded",
                            "%s\n\nThe packaged defaults are shown instead. Choose a different "
                            "config file at the top of the window." % error)
        config_path = ""

    state["config_path"] = config_path
    if config_path:
        gui_state.remember_config(state, config_path)

    from flygym_tracker.gui.main_window import MainWindow

    window = MainWindow(config=config, config_path=config_path, state=state, root=root,
                        camera_factory=camera_factory_from_config(config))
    window.show()
    # AFTER show(): Qt auto-focuses the first focusable widget when the window is shown, and
    # measured, that is the first settings spinbox. See `main_window`.
    window.take_initial_focus()
    return app.exec()


def _load(path: str):
    if not path:
        return (load_config(), None)
    try:
        return (load_config(path=path), None)
    except Exception as exc:
        return (None, str(exc))


if __name__ == "__main__":
    sys.exit(main())
