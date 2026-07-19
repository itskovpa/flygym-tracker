"""The "can I start" strip: one line per check, a plain sentence, and the button that fixes it.

Renders `flygym_tracker.readiness`, which is where the checks and their wording live. This file
owns no policy at all -- it turns `Check`s into rows and emits the `fix_action` of whichever row's
button was pressed.

NEVER HIDDEN, EVEN WHEN EVERYTHING PASSES. A strip that appears only when something is wrong is a
strip nobody has read before, so the first time it appears it is unfamiliar -- at the moment it is
most needed. All ticks is information too: it is the thing an operator looks at before walking away
from a rig for three days.
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
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows = []
        for check in readiness.checks:
            row = self._make_row(check)
            self._layout.addWidget(row)
            self._rows.append(row)

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
