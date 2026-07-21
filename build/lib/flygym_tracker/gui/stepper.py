"""The increment/decrement affordance, built as REAL WIDGETS because Qt's sub-controls lied.

=================================================================================================
THE BUG, REPRODUCED ON THE RIG WITH A REAL MOUSE AND A REAL SCREEN.

Clicking the UP arrow of a settings spinbox did nothing -- three clicks, 12.0 unchanged. Clicking
the DOWN arrow worked: 12.0 -> 11.5. Every offscreen test passed, because `stepUp()` and the Up key
both worked perfectly; only the MOUSE was broken, and only on one of the two arrows.

CAUSE, MEASURED. `theme.py` styled `QAbstractSpinBox` (padding, border, min-width) but never
defined the `::up-button` / `::down-button` / `::up-arrow` / `::down-arrow` sub-controls. The
Windows 11 style then DREW the two arrows side by side, while Qt's `subControlRect` still
hit-tested them stacked vertically -- measured up=(138, 0, 14, 15), down=(138, 15, 14, 15): same x,
split by y. So the pixel the operator aimed at was not the pixel Qt tested. Aiming at the visually
LEFT arrow landed in the geometric TOP half sometimes and the BOTTOM half other times, depending on
where in the 15px band the pointer fell.

WHY THIS IS NOT FIXED BY WRITING THE MISSING QSS. It could be. It would also be a fix that depends
on Qt's sub-control layout agreeing with Qt's sub-control hit-testing on every style, every Qt
version and every DPI -- which is exactly the agreement that just failed. Worse, overriding
`::up-button` replaces the native geometry wholesale, so a wrong `subcontrol-origin` gives ZERO-SIZE
hit rects on BOTH arrows, turning a half-broken control into a fully broken one. And no test can
tell the difference: a stylesheet sub-control has no `QWidget`, so there is nothing to assert the
existence, position or size of.

SO THE SUB-CONTROLS ARE SWITCHED OFF ENTIRELY (`setButtonSymbols(NoButtons)`) and the two steps
become ordinary `QPushButton`s with their own geometry. What is drawn IS what is clicked, because
there is only one rect and Qt owns both halves of it. A test can now assert that the increment
control EXISTS, that its rect is non-empty, and that `QTest.mouseClick` at ITS OWN CENTRE raises the
value -- which is precisely the assertion that was impossible before, and precisely the one that
would have caught this.

It also answers the operator's own request for "reactive clickable elements": a 24x24 button with
hover and pressed states is a bigger, more obviously live target than a 14x15 painted glyph.

=================================================================================================
WHAT THIS MODULE DELIBERATELY DOES NOT DO.

It does not wrap the spinbox in a facade. `row.value_widget` stays the `QAbstractSpinBox` itself,
and the buttons are its SIBLINGS inside this container. The tri-state's central assertion is
`row.findChild(QAbstractSpinBox) is None` on a default row, and the whole commit funnel, the wheel
guard and the write-back all address the spinbox directly. A wrapper exposing `value()`/`setValue()`
would have made every one of those tests pass against the wrapper while the real widget drifted --
the same class of mistake as testing a control that is off-screen.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractSpinBox, QHBoxLayout, QPushButton, QWidget

from flygym_tracker.gui import theme

#: How long a held button waits before repeating, and how fast it repeats afterwards. Slow enough
#: that a single click is unambiguously a single step -- on a live sensor, an accidental burst of
#: steps is a burst of SDK writes and a burst of `setting_change` rows.
AUTOREPEAT_DELAY_MS = 400
AUTOREPEAT_INTERVAL_MS = 120


class StepperField(QWidget):
    """A spinbox flanked by two real step buttons: ``[-] [ 12.0 fps ] [+]``.

    `spin` is the editor and stays the source of every signal the row listens to. `up` and `down`
    are ordinary buttons; they call `stepUp()` / `stepDown()`, which emit `valueChanged` exactly as
    a typed edit does, so the commit funnel is entered by the same door either way.
    """

    def __init__(self, spin: QAbstractSpinBox, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.spin = spin

        # THE LINE THAT KILLS THE BUG. With no native buttons there is no sub-control to draw in
        # one place and hit-test in another; the spinbox becomes a plain text field.
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.down = self._step_button("−", "One step down", spin.stepDown)
        layout.addWidget(self.down)
        layout.addWidget(spin, 1)
        self.up = self._step_button("+", "One step up", spin.stepUp)
        layout.addWidget(self.up)

        self.setFixedWidth(theme.VALUE_WIDTH)

    def _step_button(self, glyph: str, tip: str, slot) -> QPushButton:
        button = QPushButton(glyph)
        button.setProperty("role", "step")
        button.setFixedSize(theme.STEP_BUTTON, theme.STEP_BUTTON)
        button.setToolTip(tip)
        # NO FOCUS. Clicking a step button must not move focus off the spinbox: a focus change on
        # a spinbox emits `editingFinished`, which the row treats as an operator edit, so a single
        # click on [+] would otherwise enter the commit funnel twice.
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setAutoRepeat(True)
        button.setAutoRepeatDelay(AUTOREPEAT_DELAY_MS)
        button.setAutoRepeatInterval(AUTOREPEAT_INTERVAL_MS)
        button.clicked.connect(lambda _checked=False, s=slot: s())
        return button

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - Qt casing
        """Enable the whole cell together. A live [+] over a disabled field is a control that
        looks like it works and does not, which is the failure this module exists to end."""
        super().setEnabled(enabled)
        self.spin.setEnabled(enabled)
        self.up.setEnabled(enabled)
        self.down.setEnabled(enabled)

    def refresh_step_limits(self) -> None:
        """Grey the step button that has nowhere to go. Not cosmetic: at the top of the range the
        old native up-arrow silently did nothing, which is indistinguishable from the bug above."""
        spin = self.spin
        try:
            value = spin.value()
            self.up.setEnabled(spin.isEnabled() and value < spin.maximum())
            self.down.setEnabled(spin.isEnabled() and value > spin.minimum())
        except Exception:
            pass
