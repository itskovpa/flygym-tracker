"""The refactor's one load-bearing claim, asserted rather than described.

`settings_panel` imports cv2, `gui_support.require_gui` and `live_vial_selector` (which drags
`calibration` behind it). A Qt settings dialog that inherited all of that would refuse to open on a
machine whose OpenCV is the headless build -- for a window OpenCV is not drawing. That is a support
call on a rig a customer paid for, caused by nothing.

So the value layer moved to `settings_model`, and this file is what keeps it there: it imports the
whole GUI-facing stack with cv2 and PySide6 blocked out of `sys.modules`, and fails if anything
reaches for either. A future edit that innocently imports a drawing helper breaks a test instead of
a customer's machine.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

#: Run in a CHILD PROCESS, not with a monkeypatched `sys.meta_path`. cv2 is already imported by the
#: time this test runs (other tests import it), so blocking it in-process would prove nothing --
#: the module object is still in `sys.modules` and any `import cv2` finds it there.
#:
#: The body is spliced with `replace`, not `%`, because the blocker itself contains a `%s`.
#:
#: `find_spec` IS THE ONLY PROTOCOL THAT WORKS HERE. The first version of this blocker used the
#: legacy `find_module`/`load_module` pair, which Python REMOVED in 3.12 -- so on this machine
#: (3.14.3) it was silently a no-op and every test below passed without blocking anything. The
#: `test_asking_the_gui_to_open` case is what exposed it: instead of failing it HUNG, because
#: PySide6 imported fine and the app reached `app.exec()`. A guard that cannot fail is worse than
#: no guard, so the self-check below asserts the blocker actually blocks before anything else runs.
_BLOCKER = '''
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    BLOCKED = ("cv2", "PySide6")

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(name + " is blocked: the GUI value layer must not need it")
        return None

sys.meta_path.insert(0, Blocker())

# Self-check: prove the blocker blocks, so a broken blocker cannot make these tests pass silently.
try:
    import cv2
except ImportError:
    pass
else:
    raise SystemExit("BLOCKER FAILED: cv2 imported anyway")

__BODY__
print("OK")
'''

#: The repo root, so the child runs where `config/flygym_rig.yaml` is and finds `src/`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_isolated(body: str):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(_ROOT, "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", _BLOCKER.replace("__BODY__", body)],
                          capture_output=True, text=True, timeout=120, cwd=_ROOT, env=env)


@pytest.mark.parametrize("module", [
    "flygym_tracker.settings_model",
    "flygym_tracker.settings_controller",
    "flygym_tracker.readiness",
    "flygym_tracker.frame_source",
    "flygym_tracker.camera_lock",
    "flygym_tracker.config",
    "flygym_tracker.gui.gui_state",
])
def test_the_value_layer_imports_with_no_opencv_and_no_qt_installed(module):
    """Each of these is reachable from the Qt app's startup path, so none of them may need cv2."""
    out = _run_isolated("import %s" % module)
    assert out.returncode == 0, "%s failed to import without cv2/PySide6:\n%s" % (module,
                                                                                 out.stderr)


def test_the_whole_settings_model_can_be_built_and_saved_with_no_opencv():
    """Not just importable -- USABLE. This is the actual customer scenario: a broken OpenCV install
    and an operator who needs to look at their settings, which is exactly when they need to."""
    out = _run_isolated(
        "from flygym_tracker.config import load_config\n"
        "from flygym_tracker.settings_model import build_app_settings\n"
        "from flygym_tracker.settings_controller import SettingsController\n"
        "model = build_app_settings(load_config(path='config/flygym_rig.yaml'))\n"
        "c = SettingsController(model)\n"
        "assert c.commit('activity.pixel_threshold', 15.0).moved is True\n"
        "assert c.group_title('Camera').startswith('Camera - 5 of 5')\n")
    assert out.returncode == 0, out.stderr
    assert "OK" in out.stdout


def test_importing_flygym_tracker_still_works_without_qt():
    """The rig runs unattended experiments from the CLI. A missing GUI toolkit is not a reason for
    an overnight run to fail to start, so `import flygym_tracker` must not reach for PySide6 --
    which means `gui/__init__.py` must not import Qt at module scope either."""
    out = _run_isolated("import flygym_tracker\nimport flygym_tracker.gui\n")
    assert out.returncode == 0, out.stderr


def test_asking_the_gui_to_open_without_qt_explains_itself_instead_of_a_traceback():
    """A scientist meeting an ImportError traceback learns nothing actionable."""
    out = _run_isolated(
        "import flygym_tracker.gui as g\n"
        "code = g.main([])\n"
        "assert code == 2, code\n")
    assert out.returncode == 0, out.stderr
    assert "pip install PySide6" in out.stderr
    assert "keeps working without it" in out.stderr


def test_settings_panel_still_exports_everything_it_always_did():
    """The 216 existing tests, the CLI and the monitor all import these from `settings_panel`. The
    split moved a dependency, not an API."""
    from flygym_tracker import settings_panel as SP

    for name in ("Setting", "SettingsModel", "coerce", "build_settings", "build_camera_settings",
                 "save_settings_to_yaml", "apply_overrides_to_yaml_text", "format_value",
                 "format_hint", "format_bound", "DEFAULT_TEXT", "CAMERA_ROWS",
                 "CAMERA_PANEL_CAPS", "format_yaml_value", "COLOR_DEFAULT", "COLOR_VALUE"):
        assert hasattr(SP, name), "settings_panel stopped exporting %s" % name


def test_the_app_and_the_cv2_panel_derive_their_colours_from_the_same_numbers():
    """The bug this prevents is concrete: `COLOR_VALUE` is a BGR triple, so (0, 235, 255) renders
    AMBER. Read as RGB -- which is what hand-copying it into a stylesheet does -- the same numbers
    are CYAN. A hex literal here would have made the app cyan where the panel is amber, with a
    comment in both places claiming they matched. So the tuple is CONVERTED, in code."""
    from flygym_tracker.gui import theme
    from flygym_tracker.settings_model import COLOR_DEFAULT, COLOR_VALUE

    assert theme.bgr_to_hex(COLOR_VALUE) == theme.IMPOSED
    assert theme.bgr_to_hex(COLOR_DEFAULT) == theme.DEFAULT_GREEN
    assert theme.IMPOSED == "#FFEB00", "the panel draws amber; the app must not drift to cyan"
    assert theme.bgr_to_hex((0, 0, 255)) == "#FF0000"      # the conversion itself
