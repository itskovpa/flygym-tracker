"""The four Qt idioms this app invents, each one made of a measurement. Nothing else goes here.

Kept small on purpose: a "utils" module in a GUI grows into a second, undocumented framework. Every
item below exists because Qt's default behaviour is wrong for a surface that sits over a live
sensor, and each one names the measurement that says so.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class _NoWheelSpin:
    """Mixin: the mouse wheel never edits an UNFOCUSED spinbox, and still edits a focused one.

    THE HAZARD, MEASURED ON THIS MACHINE (PySide6 6.11 / Qt 6.11, offscreen). A plain
    `QDoubleSpinBox` sitting at 50.0, not focused, with the pointer over it: one wheel notch takes
    it to 49.0 and the event comes back `accepted=True`. In a scrolling settings page that is
    mostly spinboxes, an operator scrolling the page at 2 am changes whichever camera setting
    happens to be under the pointer -- which is invariant 2's original near-miss (a stray
    interaction imposing a value on a live recording) rebuilt out of Qt's defaults.

    WHY `ev.ignore()` AND NOT AN EVENT FILTER RETURNING True. Both stop the edit. Measured
    difference: an event filter returning True stops the wheel editing a FOCUSED box as well
    (50.0 -> 50.0 with the box focused) and, per Qt's documented filter semantics, consumes the
    event so the enclosing `QScrollArea` never sees it -- i.e. the settings page stops scrolling by
    wheel over most of its own surface, on a panel whose predecessor was already called unusable.
    `ignore()` measured 50.0 -> 50.0 unfocused with `accepted=False` (Qt's documented "not mine,
    pass it on") and 50.0 -> 49.0 focused, so deliberate editing survives.

    WHAT COULD NOT BE MEASURED, STATED PLAINLY: whether the ignored event actually reaches the
    enclosing scroll area. Offscreen, the scrollbar stayed at 0 in every variant tried INCLUDING
    the unguarded one, so the null result says nothing about the guard -- `QApplication.sendEvent`
    does not reproduce the platform's propagation walk. The semantics are documented and
    `accepted=False` is the right signal; the scroll behaviour needs one look on a real display.
    """

    def wheelEvent(self, event) -> None:
        if not self.hasFocus():
            event.ignore()
            return
        super().wheelEvent(event)


class NoWheelDoubleSpinBox(_NoWheelSpin, QDoubleSpinBox):
    pass


class NoWheelSpinBox(_NoWheelSpin, QSpinBox):
    pass


@contextmanager
def no_signals(widget):
    """Set a widget's value without the write-back re-entering the edit funnel.

    NOT HYGIENE. Measured: `QDoubleSpinBox.setValue(5.0)` called from code emits `valueChanged`
    (`QSignalSpy` recorded exactly one emission, `[5.0]`). The commit loop ends by writing the
    model's coerced value back into the widget, so without this the write-back arrives as a fresh
    operator edit -- and on a camera row that is a second SDK write and a second `setting_change`
    event for one keystroke.
    """
    blocker = QSignalBlocker(widget)
    try:
        yield widget
    finally:
        del blocker


def pump(app, until=None, timeout: float = 2.0, interval: float = 0.001) -> bool:
    """Run the event loop until `until()` is true, or `timeout` seconds pass. True if it became true.

    Tests need this because the app deliberately does its camera work on another thread: a queued
    slot has not run when the call that posted it returns. `app.exec()` would never return, and
    `QTest.qWait` is not available in every PySide6 build, so this is the one loop.
    """
    end = time.monotonic() + float(timeout)
    while True:
        app.processEvents()
        if until is None or until():
            return True
        if time.monotonic() >= end:
            return bool(until is None or until())
        time.sleep(interval)
