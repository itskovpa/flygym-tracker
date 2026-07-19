"""How the app is reached, and the one thing that keeps run.bat and the app from disagreeing.

Qt-free by construction: none of this may import PySide6, because `--print-paths` is called by
`run.bat` on every trip round the menu, on machines that may not have Qt installed at all.
"""
from __future__ import annotations

import io
import os
import re
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


def test_run_bat_reads_the_paths_from_the_app_instead_of_keeping_its_own():
    """THE REGRESSION THIS PREVENTS. The batch file used to carry CONFIG/CALIB/OUTDIR in a header
    block the operator edited by hand. With the app owning them, a second copy here would mean
    choosing an output folder in the app and still getting results somewhere else from the menu's
    "Start experiment" -- the results appearing exactly where the operator had just said not to."""
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    assert "gui --print-paths" in text
    # The batch VARIABLE must be gone, not every mention of the name -- the comment explaining why
    # it is gone names it, and asserting on the name alone would forbid the explanation.
    assert 'set "BIN_SECONDS=' not in text, \
        "bin size is a measurement parameter and belongs in the config YAML, not the launcher"
    assert "--bin-seconds" not in text


def test_run_bat_makes_the_app_the_default_choice():
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    # A NEW LETTER, not a renumbering. `test_cli.py` asserts that [1]..[5] keep their meanings --
    # they are what is written on the note stuck to the rig -- so the app becomes the obvious way
    # in by being first in the list and the default on Enter, not by taking someone else's number.
    assert 'if not defined CH set "CH=A"' in text
    assert 'if /I "%CH%"=="A" goto app' in text
    order = re.findall(r"^echo\s+\[([^\]]+)\]", text, re.MULTILINE)
    assert order[0] == "A", "the app must be the first entry the operator reads"


def test_run_bat_still_offers_every_old_menu_entry():
    """The menu is the operator's mental model of this program. Losing an entry to a redesign is
    how a rig owner concludes the update broke something."""
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    for label in ("goto run", "goto selectvials", "goto replay", "goto noise", "goto freecam",
                  "goto settings"):
        assert label in text, "run.bat lost %r" % label


def test_run_bat_no_longer_gates_the_settings_surface_on_an_opencv_gui_build():
    """The Qt app draws with Qt. Refusing to open it because OpenCV was installed headless would be
    a support call caused by nothing."""
    text = io.open(RUN_BAT, encoding="utf-8", errors="replace").read()
    assert "does not use OpenCV to draw" in text


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
