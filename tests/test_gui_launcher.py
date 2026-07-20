"""How the app is reached, and the one thing that keeps run.bat and the app from disagreeing.

Qt-free by construction: none of this may import PySide6, because `--print-paths` is called by
`run.bat` on every trip round the menu, on machines that may not have Qt installed at all.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

from flygym_tracker.cli import build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_BAT = os.path.join(ROOT, "run.bat")


def test_the_cli_has_a_gui_subcommand():
    args = build_parser().parse_args(["gui", "--config", "x.yaml"])
    assert args.handler.__name__ == "_cmd_gui"
    assert args.config == "x.yaml"


def test_print_paths_reports_the_apps_own_state_and_needs_no_qt(tmp_path, monkeypatch):
    """`run.bat` calls this on every menu redraw. If it imported Qt it would fail on exactly the
    machines that most need the menu to keep working."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "from importlib.abc import MetaPathFinder\n"
         "class B(MetaPathFinder):\n"
         "    def find_spec(self, name, path=None, target=None):\n"
         "        if name.split('.')[0] == 'PySide6':\n"
         "            raise ImportError('blocked')\n"
         "        return None\n"
         "sys.meta_path.insert(0, B())\n"
         "from flygym_tracker.cli import main\n"
         "raise SystemExit(main(['gui', '--print-paths']))\n"],
        capture_output=True, text=True, timeout=120, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    names = dict(line.split("=", 1) for line in out.stdout.strip().splitlines())
    assert set(names) == {"CONFIG", "CALIB", "OUTDIR"}
    assert names["CALIB"]


def test_run_bat_keeps_no_paths_of_its_own():
    """THE REGRESSION THIS PREVENTS. The batch file used to carry CONFIG/CALIB/OUTDIR in a header
    block the operator edited by hand. With the app owning them, a second copy here would mean
    choosing an output folder in the app and still getting results somewhere else -- the results
    appearing exactly where the operator had just said not to.

    It used to read them back with `gui --print-paths` so the menu and the app agreed. There is no
    menu now, so there is nothing to keep in agreement: the app reads its own saved state.
    """
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    for stale in ('set "CONFIG=', 'set "CALIB=', 'set "OUTDIR=', 'set "BIN_SECONDS='):
        assert stale not in text, "run.bat still carries a path of its own: %s" % stale
    assert "--bin-seconds" not in text


def test_run_bat_goes_straight_into_the_app_with_nothing_to_choose():
    """It used to make the app the DEFAULT choice on a menu ([A], first in the list, taken on
    Enter). Being the default was the best a menu could do; not having a menu is better, and it is
    what "no terminal prompt anywhere in the app" means for the file that starts it."""
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    assert "flygym_tracker.cli gui" in text
    assert 'set /p CH=' not in text, "run.bat still asks the operator to choose"
    assert ":menu" not in text


def test_every_old_menu_entry_still_exists_somewhere_the_operator_can_reach_it(qapp, tmp_path):
    """THE ORIGINAL CONCERN SURVIVES THE REDESIGN, and it is a real one: "the menu is the
    operator's mental model of this program. Losing an entry to a redesign is how a rig owner
    concludes the update broke something."

    So the assertion moved rather than being dropped. The five jobs are no longer `goto` labels in
    a batch file; they are buttons in the window, and this checks that ALL of them made the trip.
    Deleting this test because the labels are gone would have been the exact failure it was written
    to catch.
    """
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow
    from PySide6.QtWidgets import QPushButton

    window = MainWindow(config=load_config(path="config/flygym_rig.yaml"),
                        config_path="config/flygym_rig.yaml", state=gui_state.default_state(),
                        root=str(tmp_path), camera_factory=lambda: None,
                        confirm=lambda text: False)
    try:
        labels = " | ".join(b.text() for b in window.findChildren(QPushButton))
        for job in ("Start experiment", "Draw vial positions", "Replay a recording",
                    "Measure noise floor", "Free the camera"):
            assert job in labels, "the app has no way to %r" % job
    finally:
        window.run.shutdown()
        window.session.shutdown()


def test_run_bat_never_blocks_the_app_on_an_opencv_gui_build():
    """The app draws with Qt, and now that EVERY video job happens in its own window it does not
    open an OpenCV window at all -- not even indirectly through a child process. Refusing to start,
    or nagging about a headless OpenCV, would be a support call caused by nothing.

    Tested as BEHAVIOUR rather than by matching a sentence: what matters is that the launcher does
    not branch on `has_gui_support`, whatever wording explains it.
    """
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    code = "\n".join(line for line in text.splitlines()
                     if not line.strip().upper().startswith("REM"))
    assert "has_gui_support" not in code, \
        "run.bat still gates on an OpenCV GUI build the app no longer needs"
    assert "pip uninstall" not in code, \
        "run.bat still offers to reinstall OpenCV for a window it no longer opens"


def test_pyproject_ships_a_gui_entry_point_and_keeps_qt_optional():
    text = io.open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8").read()
    assert "flygym-tracker-gui" in text
    assert "[project.gui-scripts]" in text, \
        "a console entry point would leave a black console window behind the app"
    # Qt must NOT be a hard dependency: an overnight run must not fail to start because a GUI
    # toolkit is missing.
    deps = text.split("dependencies = [")[1].split("]")[0]
    assert "PySide6" not in deps
    assert 'gui = ["PySide6' in text
