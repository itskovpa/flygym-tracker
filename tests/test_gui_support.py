"""Guards the preflight that turns a cryptic cv2.imshow failure into an actionable message.

Regression: `opencv-python-headless` (GUI: NONE) was installed on the rig machine, so the
calibration wizard died with "The function is not implemented. Rebuild the library with Windows,
GTK+ 2.x or Cocoa support" — a traceback that tells a biologist nothing. The whole test suite passed
because every interactive path is deliberately headless-safe in tests, so nothing caught it.
"""
import pytest

from flygym_tracker import gui_support


def test_gui_backend_is_reported_as_a_string():
    assert isinstance(gui_support.gui_backend(), str)


def test_has_gui_support_agrees_with_the_backend():
    backend = gui_support.gui_backend()
    assert gui_support.has_gui_support() == (backend not in ("NONE", "UNKNOWN", ""))


@pytest.mark.parametrize("backend,expected", [
    ("NONE", False), ("UNKNOWN", False), ("", False),
    ("WIN32UI", True), ("GTK3", True), ("COCOA", True), ("QT5", True),
])
def test_backend_values_classify_correctly(monkeypatch, backend, expected):
    monkeypatch.setattr(gui_support, "gui_backend", lambda: backend)
    assert gui_support.has_gui_support() is expected


def test_require_gui_raises_actionable_message_when_headless(monkeypatch):
    monkeypatch.setattr(gui_support, "gui_backend", lambda: "NONE")
    monkeypatch.setattr(gui_support, "headless_package_installed", lambda: True)
    with pytest.raises(SystemExit) as exc:
        gui_support.require_gui("The ROI editor")
    msg = str(exc.value)
    # names the feature, the cause, and the exact fix commands
    assert "The ROI editor" in msg
    assert "opencv-python-headless" in msg
    assert "pip uninstall -y opencv-python-headless" in msg
    assert "pip install opencv-python" in msg


def test_require_gui_is_silent_when_a_backend_exists(monkeypatch):
    monkeypatch.setattr(gui_support, "gui_backend", lambda: "WIN32UI")
    gui_support.require_gui("The ROI editor")  # must not raise


def test_diagnosis_still_helps_when_headless_package_is_absent(monkeypatch):
    monkeypatch.setattr(gui_support, "gui_backend", lambda: "NONE")
    monkeypatch.setattr(gui_support, "headless_package_installed", lambda: False)
    msg = gui_support.gui_diagnosis("The live monitor")
    assert "The live monitor" in msg
    assert "pip install opencv-python" in msg
