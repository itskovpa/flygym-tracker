"""Qt fixtures for the GUI tests. ~30 lines, which is why pytest-qt is not a dependency.

pytest-qt IS NOT INSTALLED AND MUST NOT BE ADDED. Measured on this machine (PySide6 6.11.0 /
Qt 6.11.0 / Python 3.14.3): `QT_QPA_PLATFORM=offscreen` gives a working `QApplication` in 0.006 s
after a 0.12 s import, and `PySide6.QtTest` -- which ships in the wheel -- provides everything
`qtbot` wraps. Verified working offscreen in this project: `QTest.keyClick` (a spinbox stepped
5 -> 6), `QTest.keyClicks` (typed "5000" into a box), `QTest.mouseClick`, and `QSignalSpy`
(`.count()` == 1, `.at(0)` == [5.0] -- note PySide6's QSignalSpy has no `__len__`, so use
`.count()`). Adding a dependency to a machine that runs unattended multi-day experiments, in
exchange for a fixture that is thirty lines, is not a trade worth making.

THE ROOT `conftest.py` (which puts `src/` on the path) IS UNCHANGED and still does that job. This
one only adds Qt, and only for the tests that ask for it: `importorskip` keeps the other 894 tests
running on a machine with no PySide6 at all.
"""
import os

# BEFORE any PySide6 import, including the ones inside the fixtures below. Setting it afterwards
# has no effect: the platform plugin is chosen when QGuiApplication is first constructed, and by
# then a test run on a machine with no display has already failed.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session -- Qt permits exactly one per process."""
    from PySide6.QtWidgets import QApplication

    yield QApplication.instance() or QApplication([])


@pytest.fixture
def pump(qapp):
    """Run the event loop until `until()` is true. The app does its camera work on a thread, so a
    queued slot has not run by the time the call that posted it returns."""
    from flygym_tracker.gui.qt_compat import pump as _pump

    def run(until=None, timeout=2.0):
        return _pump(qapp, until, timeout)

    return run


@pytest.fixture(autouse=True)
def _no_widget_leak(qapp):
    """Close every top-level widget after each test.

    Not politeness: a leaked window keeps its `CameraSession` alive, which keeps a `QThread`
    running, and a later test that asserts on thread identity or on what a fake camera was told
    then sees another test's worker. Failures like that are ordering-dependent and cost hours.
    """
    yield
    for widget in list(qapp.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    qapp.processEvents()
