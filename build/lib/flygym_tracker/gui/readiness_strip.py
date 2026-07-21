"""The "can I start" strip: one line per check, a plain sentence, and the button that fixes it.

Renders `flygym_tracker.readiness`, which is where the checks and their wording live. This file
owns no policy at all -- it turns `Check`s into rows and emits the `fix_action` of whichever row's
button was pressed.

NEVER HIDDEN, EVEN WHEN EVERYTHING PASSES. A strip that appears only when something is wrong is a
strip nobody has read before, so the first time it appears it is unfamiliar -- at the moment it is
most needed. All ticks is information too: it is the thing an operator looks at before walking away
from a rig for three days.

BUT WHAT PASSES IS COLLAPSED TO ONE LINE, AND THAT IS A MEASUREMENT, NOT A PREFERENCE. Rendered as
six full rows this strip took 210 px of an 880 px window -- 24% of the height, 35 px per short
sentence -- while the camera picture beside it got 312 px. On a rig where the whole point of the
window is looking at the picture (exposure and gain are tuned by eye, and vial polygons are drawn on
it), a checklist that is almost always all-ticks was the second-largest thing on screen.

So: everything that passes becomes ONE line naming the checks, and anything that does NOT pass keeps
its full row and its fix button. The strip is still always present and still in the same place, so
it is still familiar -- but it grows only when it has something to say, and the space goes to the
picture the rest of the time.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout,
                               QWidget)

from flygym_tracker.gui import theme
from flygym_tracker.readiness import BAD, OK, UNKNOWN, Readiness

#: Mark, colour and accessible word per state. The word matters: colour alone fails a colour-blind
#: operator, and this strip is the one surface where a missed cross costs days of experiment.
MARKS = {
    OK: ("OK", theme.GOOD),
    BAD: ("X", theme.ERROR),
    UNKNOWN: ("-", theme.DIM),
}


#: What to call each check on the one-line summary. Keyed by `Check.key`, which is a STABLE
#: identifier -- the same reason `fix_action` is matched on rather than the button's prose.
#: A key with no entry here falls back to itself, so a new check appears in the summary the day it
#: is added rather than silently going missing from it.
SHORT_NAMES = {
    "config": "config",
    "calibration": "vial positions",
    "output": "output folder",
    "camera": "camera",
    "unverified": "camera values",
    "unsaved": "settings saved",
}


def _short_name(check) -> str:
    return SHORT_NAMES.get(check.key, check.key)


class ReadinessStrip(QWidget):
    """One row per `Check`. Rebuilt wholesale on every update -- there are six of them."""

    fix_requested = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 4, 10, 4)
        self._layout.setSpacing(2)
        self._rows = []

    def set_readiness(self, readiness: Readiness) -> None:
        """Problems in full, with their fix buttons; everything that passes on one line.

        THE PASSING LINE COMES LAST, deliberately. Whatever is wrong is what the operator has to
        act on, so it sits at the top of the strip where the eye lands, and the reassurance sits
        under it. Ordering it the other way round would push a cross below a row of ticks.
        """
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []

        checks = list(readiness.checks)
        problems = [c for c in checks if c.state != OK]
        passing = [c for c in checks if c.state == OK]
        for check in problems:
            self._add(self._make_row(check))
        if passing:
            self._add(self._make_summary(passing, len(checks)))

    def _add(self, row: QWidget) -> None:
        self._layout.addWidget(row)
        self._rows.append(row)

    def _make_summary(self, passing, total: int) -> QWidget:
        """One line for everything that passed, NAMING what passed rather than counting it.

        "5 of 6 checks pass" tells an operator nothing they can act on; naming them is what makes
        this readable as the pre-flight list it is. The full sentences stay reachable as a tooltip,
        so nothing that was on screen before has been thrown away -- only stacked.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        glyph = QLabel(MARKS[OK][0])
        glyph.setFixedWidth(20)
        glyph.setStyleSheet("color: %s; font-weight: 700;" % MARKS[OK][1])
        layout.addWidget(glyph)

        names = ", ".join(_short_name(check) for check in passing)
        label = QLabel("%s ready: %s" % (
            "everything" if len(passing) == total else "%d of %d" % (len(passing), total), names))
        label.setProperty("role", "note")
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        label.setToolTip("\n".join(check.sentence for check in passing))
        layout.addWidget(label, 1)
        return row

    def _make_row(self, check) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        mark, colour = MARKS[check.state]
        glyph = QLabel(mark)
        glyph.setFixedWidth(20)
        glyph.setStyleSheet("color: %s; font-weight: 700;" % colour)
        layout.addWidget(glyph)

        sentence = QLabel(check.sentence)
        sentence.setWordWrap(True)
        sentence.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        if check.state != BAD:
            sentence.setProperty("role", "note")
        layout.addWidget(sentence, 1)

        if check.fix_label and check.fix_action:
            button = QPushButton(check.fix_label)
            # `fix_action` is a stable identifier, not the label: a fix button that silently stops
            # working because a sentence was reworded is worse than no button at all.
            action = check.fix_action
            button.clicked.connect(lambda _checked=False, a=action: self.fix_requested.emit(a))
            layout.addWidget(button)
        return row
