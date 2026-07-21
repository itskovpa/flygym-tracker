"""One widget per `Setting`, and the tri-state that makes "camera default" impossible to hit.

INVARIANT 2, AND WHY IT IS STRUCTURAL HERE. In the cv2 panel a row at its device default draws an
EMPTY track -- no handle, nothing to grab -- because a click anywhere on a track is a value. That
guard was geometry, and it very nearly failed: a stray click on such a row imposed 2.6 fps on a
live 88 fps recording. The Qt version does not defend the control; IT DOES NOT BUILD ONE. A row at
"camera default" holds a `QLabel`, a hint and a button. `row.findChild(QAbstractSpinBox) is None`
is the assertion -- existence, not geometry, and it survives a layout change in a way a rect test
cannot.

THE EDITOR IS CREATED ON ARM AND DESTROYED ON RETURN, which is NOT what a plain `QStackedWidget`
does, and the difference is not cosmetic. Two defects were caught by writing the test first:

  1. A `QStackedWidget` keeps every page as a live child; only visibility changes. So with both
     pages built up front, `findChild(QAbstractSpinBox)` FOUND the editor on a row showing "camera
     default" -- the central claim of the design was simply false as specified.
  2. Worse: switching the stack back to the default page made the spinbox lose focus, which emits
     `editingFinished`, which re-entered the commit funnel and RE-IMPOSED the value that had just
     been cleared. Pressing "back to camera default" would have written the number to the sensor
     again.

So the explicit page is built when the row is armed and deleted when it goes back, with the
disconnect happening before the teardown. THIS RESTYLE DOES NOT TOUCH THAT LOGIC. The row was
recomposed into a fixed grid and repainted; `_ensure_editor`, `_connect_editor`, `_disconnect_editor`
and the `_tearing_down` flag are the same code they were, deliberately, because they are the part
that is load-bearing and the part a visual pass is most likely to break by accident.

`setSpecialValueText("camera default")` IS THE OBVIOUS ANSWER AND IT IS WRONG. Measured: the
special text occupies `minimum()`, so the box reads back `('camera default', 0.1)`. That makes
"camera default" the same thing as 0.1 fps -- reachable by stepping down, indistinguishable from
the slowest legal value, and one wheel notch from either. Rejected.

THE STATE IS UNMISTAKABLE THROUGH FIVE CHANNELS, because any one of them fails on a tired operator
at 2 am or a colour-blind one at any hour:
    SHAPE   flat text with no editor at all, versus a boxed, recessed editor well
    COLOUR  green means "we are leaving this alone", amber means "we are sending this"
    WORDS   literally "camera default", with what the sensor reports beside it
    COUNT   the group header carries "3 OF 5 AT CAMERA DEFAULT" (see `settings_view`)
    SPINE   a 3px bar at the row's left edge -- see `StateSpine`

THE SPINE IS THE NEW ONE, and it carries SHAPE as well as colour on purpose. Green-vs-amber is the
worst confusion pair under deuteranopia, and a colour-only channel would have handed the fastest-to-
scan cue in the design to everyone except the operators who need it most. So an imposed row draws a
FULL-HEIGHT bar and a default row draws a SHORT CENTRED TICK: the vertical stripe down a column of
rows is readable as a pattern of long and short marks with the colour removed entirely.

THE WIDGET IS NEVER THE SOURCE OF TRUTH. Every edit goes to `SettingsController.commit`, which
clamps, snaps and casts; whatever it stores is written back with signals blocked. Blocking is
mandatory, not hygiene: measured, a programmatic `setValue` emits `valueChanged`, so without it the
write-back re-enters the funnel as a fresh operator edit -- a second SDK write and a second
`setting_change` event for one keystroke.
"""
from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from flygym_tracker.gui import theme
from flygym_tracker.gui.qt_compat import NoWheelDoubleSpinBox, NoWheelSpinBox, no_signals
from flygym_tracker.gui.stepper import StepperField
from flygym_tracker.settings_controller import NEVER_CHECKED_BADGE, arming_plan, to_default_notice
from flygym_tracker.settings_model import DEFAULT_TEXT, Setting, _decimals_for, format_hint

#: Reveal the help line on hover and focus instead of reserving a second line on every row.
#:
#: THE TRADE, STATED. Showing it on hover reflows the rows below by one line, and this app's own
#: motion rule is that nothing should move under a pointer that is about to click. Against that:
#: the user asked for everything on one window, and a permanent second line drops the pane from
#: about twelve settings to about eight. The block REASON is exempt and always visible, because a
#: greyed control that does not say why is a support call. The full help text is also always in the
#: label's tooltip, so nothing is only reachable by hovering.
#:
#: Set False to go back to a permanently visible help line. One constant, one line.
HELP_ON_HOVER = True

#: Badges are annotations, not controls. See `_badge` for the render that produced this number.
BADGE_HEIGHT = 18

#: Widest a value editor may get -- re-exported from `theme` so the geometry the layout test
#: measures and the geometry the stylesheet assumes are one number.
#:
#: THE MEASUREMENT THAT MADE ALL OF THIS NECESSARY. Before it, the settings pane's content had a
#: minimum width of 1222 px inside a 584 px viewport, with the horizontal scrollbar deliberately
#: disabled -- so EVERY value control on EVERY row sat off the right edge, unreachable, in a
#: settings window whose entire purpose is reaching them. The whole test suite was green: offscreen
#: tests drove the widgets by calling their methods, which works perfectly on a control nobody can
#: see. `tests/test_gui_layout.py` now measures the geometry instead of trusting it.
VALUE_WIDTH = theme.VALUE_WIDTH


def _badge(text: str, role: str = "badge") -> QLabel:
    """A badge is a small mark beside the label, NOT a column.

    Seen by rendering the row and looking at it: without an explicit alignment a QLabel in an
    HBox is stretched to the tallest item in the row, so "unchecked" and "start only" drew as
    full-height outlined boxes taller than the value field -- reading as a second control rather
    than as an annotation on the first.
    """
    label = QLabel(text)
    label.setProperty("role", role)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    label.setFixedHeight(BADGE_HEIGHT)
    return label


def _wrapping_label(text: str, role: str) -> QLabel:
    """A label that wraps instead of widening its row. See `SettingRow._build` for why that matters."""
    label = QLabel(text)
    label.setProperty("role", role)
    label.setWordWrap(True)
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
    return label


def _shrinkable(widget: QWidget, tooltip: str = "") -> QWidget:
    """Let `widget` be squeezed below its natural width, keeping the full text in a tooltip.

    Three things force a row wide, and all three are capped: a `QLabel` reports its full text width
    as its minimum unless it is allowed to shrink, a spinbox stretches to fill, and a button sizes
    to its label.
    """
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    widget.setMinimumWidth(0)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


class StateSpine(QWidget):
    """The 3px bar at a row's left edge. Colour AND shape, never colour alone.

    ``full`` (imposed) draws the whole height; ``tick`` (at camera default) draws a short centred
    segment; ``thin`` (blocked) draws a 1px full-height line; ``none`` draws nothing at all, which
    is what a non-nullable app setting gets -- it has no sensor to impose anything on, so a state
    mark there would be a fifth channel reporting a state that does not exist.
    """

    WIDTH = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(self.WIDTH)
        self._color = "transparent"
        self._shape = "none"

    def set_state(self, color: str, shape: str) -> None:
        if (color, shape) != (self._color, self._shape):
            self._color, self._shape = color, shape
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt casing
        if self._shape == "none" or self._color == "transparent":
            return
        painter = QPainter(self)
        color = QColor(self._color)
        height = self.height()
        if self._shape == "tick":
            # Short centred mark: "we are leaving this alone" reads as a small, quiet mark even
            # with the colour removed.
            span = max(8, int(height * 0.35))
            painter.fillRect(0, (height - span) // 2, self.WIDTH, span, color)
        elif self._shape == "thin":
            painter.fillRect(0, 0, 1, height, color)
        else:
            painter.fillRect(0, 0, self.WIDTH, height, color)
        painter.end()


def make_value_widget(setting: Setting) -> QWidget:
    """The editor for one `Setting`, configured the four ways that are not optional.

    NO SLIDERS, ANYWHERE. A `QSlider` is a pixel-to-value map over a live sensor -- every position
    on the track is a number, and a stray interaction lands on one. That is the cv2 panel's failure
    mode ported into Qt. Numbers here are typed or stepped, never dragged.

    1. `setKeyboardTracking(False)`. Measured: with tracking ON, typing "5000" into the exposure box
       emits `valueChanged` at 5, then 50, then 500, then 5000 -- four SDK writes walking the sensor
       through three exposures nobody asked for, and four `setting_change` rows in events.csv for
       one edit. With it OFF the probe recorded no intermediate emissions; the value commits on
       Enter or when the box loses focus.
    2. The no-wheel guard (see `qt_compat._NoWheelSpin`): an unfocused box under the pointer went
       50.0 -> 49.0 on one notch.
    3. `StrongFocus`, so the box takes focus by click AND by tab -- the guard in (2) is keyed on
       focus, so a box that could not be focused could not be edited at all.
    4. The native step arrows are switched off by `StepperField` and replaced with real buttons.
       See `stepper` for the measured bug: what Qt DREW and what Qt HIT-TESTED were different
       rectangles, so the up arrow was dead to the mouse and the down arrow was not.
    """
    if setting.kind == "bool":
        box = QCheckBox()
        box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        return box
    if setting.kind == "int":
        box = NoWheelSpinBox()
        box.setRange(int(setting.lo), int(setting.hi))
        box.setSingleStep(max(1, int(setting.step)))
    else:
        box = NoWheelDoubleSpinBox()
        box.setDecimals(_decimals_for(setting.step))
        box.setRange(float(setting.lo), float(setting.hi))
        box.setSingleStep(float(setting.step))
    box.setKeyboardTracking(False)
    box.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    # NO SUFFIX INSIDE THE BOX, AND THIS WAS MEASURED. Promoting the numeral to 15pt monospace and
    # fixing the value column at 168px left a 112px field, and `setSuffix(" grey levels")` then
    # rendered "12.0 grey levels" as "12.0 gre" -- the hero of the row, truncated mid-word. Seen by
    # rendering the window and looking at the picture, which is the same way the off-the-right-edge
    # defect in `tests/test_gui_layout.py` was found.
    #
    # The unit is not dropped, it MOVES to the label ("pixel threshold (grey levels)"), where the
    # column has stretch and elides into a tooltip instead of clipping. `format_hint` and the cv2
    # panel still say the unit beside the number, so the fact is stated in the same places it was.
    box.setAccessibleName(label_with_unit(setting))
    return box


def label_with_unit(setting: Setting) -> str:
    """"pixel threshold (grey levels)". The unit lives here now -- see `make_value_widget`."""
    unit = (setting.unit or "").strip()
    if not unit or unit.lower() in setting.label.lower():
        return setting.label
    return "%s (%s)" % (setting.label, unit)


class SettingRow(QWidget):
    """A non-nullable setting: label, editor, badges, and a help line that doubles as the reason.

    `edited` carries only the KEY. The view asks the controller for the value, because the
    controller is what knows what was actually stored -- a signal carrying the widget's number
    would make the widget the source of truth by the back door.
    """

    edited = Signal(str)

    def __init__(self, setting: Setting, controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setting = setting
        self.key = setting.key
        self.controller = controller
        #: True while an editor is being torn down, so its parting `editingFinished` cannot be
        #: mistaken for an operator edit. See the module docstring, defect 2.
        self._tearing_down = False
        self._hovered = False
        self._stepper: Optional[StepperField] = None
        self._flash_timer: Optional[QTimer] = None

        self.value_widget: Optional[QWidget] = self._make_editor()
        self._build()
        if self.value_widget is not None:
            self._connect_editor(self.value_widget)
        self.refresh()

    # -- construction --------------------------------------------------------------------------
    def _make_editor(self) -> Optional[QWidget]:
        return make_value_widget(self.setting)

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # THE FOUR-COLUMN GRID. Fixed widths on the value and action columns so that every numeral
        # in the pane lands in ONE vertical stripe. An HBox-with-stretch puts every row's value at
        # a different x, and a single stripe of right-aligned monospace numerals is the strongest
        # "this was engineered" signal available -- and it is free.
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.spine = StateSpine()
        top.addWidget(self.spine)

        left = QHBoxLayout()
        left.setContentsMargins(8, 0, 0, 0)
        left.setSpacing(6)
        self.label = _shrinkable(QLabel(label_with_unit(self.setting)), self.setting.help)
        self.label.setProperty("role", "label")
        left.addWidget(self.label, 1)

        # Same words the cv2 panel uses, for the same two facts, so an operator who learned them
        # there does not have to learn them again here.
        self.badge_next_run = _badge("next run")
        self.badge_next_run.setVisible(not self.setting.live)
        left.addWidget(self.badge_next_run)
        self.badge_start_only = _badge("start only")
        self.badge_start_only.setVisible(bool(self.setting.start_only))
        left.addWidget(self.badge_start_only)
        # SHORT ON THE ROW, FULL EVERYWHERE THERE IS ROOM. The badge sits between the label and
        # the editor, so the full phrase ("never checked against a camera") would push the editor
        # off the pane -- the defect `tests/test_gui_layout.py` was written for.
        self.badge_unchecked = _badge("unchecked", role="warnbadge")
        self.badge_unchecked.setToolTip(NEVER_CHECKED_BADGE)
        self.badge_unchecked.setVisible(False)
        left.addWidget(self.badge_unchecked)
        top.addLayout(left, 1)

        self.value_cell = self._value_container()
        self.value_cell.setFixedWidth(theme.VALUE_WIDTH)
        top.addWidget(self.value_cell)

        # The action column exists on every row, occupied or not, so the value column's right edge
        # is the same x on all of them.
        self.action_cell = QWidget()
        self.action_cell.setFixedWidth(theme.ACTION_WIDTH)
        action_layout = QHBoxLayout(self.action_cell)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(0)
        self._build_action(action_layout)
        top.addWidget(self.action_cell)

        holder = QWidget()
        holder.setLayout(top)
        holder.setMinimumHeight(theme.ROW_HEIGHT)
        outer.addWidget(holder)

        # Full width, under the top line: anything too long for the 168px value column goes here
        # rather than being clipped inside it. See `NullableSettingRow._value_container`.
        self._build_second_line(outer)

        # WRAPPING LABELS MUST BE ALLOWED TO SHRINK. A word-wrapped QLabel reports a minimum width
        # based on its longest unbreakable run, and inside a layout that respects it the label
        # pushes the whole row wider instead of wrapping -- which is how a help line ends up
        # clipped at the pane edge with no scrollbar to reveal it (the scroll area deliberately has
        # no horizontal bar; see `settings_view`).
        self.help = _wrapping_label(self.setting.help, "help")
        self.help.setContentsMargins(11, 0, 0, 0)
        outer.addWidget(self.help)

        self.notice = _wrapping_label("", "notice")
        self.notice.setContentsMargins(11, 0, 0, 0)
        self.notice.setVisible(False)
        outer.addWidget(self.notice)

        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def _build_action(self, layout) -> None:
        """Non-nullable rows have nothing to revert to -- the column stays empty but present."""
        return None

    def _build_second_line(self, outer) -> None:
        """A non-nullable row has no sensor hint to place; only the tri-state rows do."""
        return None

    def _value_container(self) -> QWidget:
        """Wrap the editor in its stepper, so the step buttons are real widgets with real rects."""
        if isinstance(self.value_widget, QCheckBox):
            holder = QWidget()
            layout = QHBoxLayout(holder)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addStretch(1)
            layout.addWidget(self.value_widget)
            return holder
        self._stepper = StepperField(self.value_widget)
        return self._stepper

    def _connect_editor(self, widget: QWidget) -> None:
        if isinstance(widget, QCheckBox):
            widget.toggled.connect(self._on_widget_changed)
            return
        # BOTH signals, deliberately. With keyboardTracking off, `valueChanged` fires on Enter and
        # on the step buttons; `editingFinished` covers the focus-out path, which measured as NOT
        # emitting valueChanged offscreen. The commit is idempotent (a no-op edit returns early),
        # so the overlap costs nothing and the gap would have cost an edit.
        widget.valueChanged.connect(self._on_widget_changed)
        widget.editingFinished.connect(self._on_editing_finished)

    def _disconnect_editor(self, widget: QWidget) -> None:
        """Detach before teardown, so a parting signal cannot re-enter the funnel."""
        try:
            if isinstance(widget, QCheckBox):
                widget.toggled.disconnect(self._on_widget_changed)
            else:
                widget.valueChanged.disconnect(self._on_widget_changed)
                widget.editingFinished.disconnect(self._on_editing_finished)
        except (RuntimeError, TypeError):
            pass                       # already gone, or never connected

    # -- hover, purely visual --------------------------------------------------------------------
    def enterEvent(self, event) -> None:  # noqa: N802 - Qt casing
        self._hovered = True
        self._sync_help_visibility()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt casing
        self._hovered = False
        self._sync_help_visibility()
        super().leaveEvent(event)

    def _sync_help_visibility(self) -> None:
        """The block reason is ALWAYS visible; ordinary help is revealed on hover or focus."""
        if not HELP_ON_HOVER:
            self.help.setVisible(True)
            return
        blocked = self.controller.block_reason(self.key) is not None
        focused = self.value_widget is not None and self.value_widget.hasFocus()
        self.help.setVisible(bool(blocked or self._hovered or focused))

    # -- the commit loop -----------------------------------------------------------------------
    def _on_editing_finished(self) -> None:
        if self._tearing_down or self.value_widget is None:
            return
        self._on_widget_changed(self.value_widget.value())

    def _on_widget_changed(self, value: Any) -> None:
        """Widget -> controller -> widget. The controller decides what the value IS."""
        if self._tearing_down:
            return
        result = self.controller.commit(self.key, value)
        if result.blocked:
            # Nothing was stored, so put the widget back to what the model still holds. This is the
            # path a programmatic `setValue` on a disabled box would take -- measured to succeed,
            # which is exactly why the refusal cannot live in `setEnabled`.
            self._write_back()
            self._show_notice(result.message)
            return
        if result.moved:
            self._write_back()
            self._show_notice("" if result.applied else result.message)
            self.edited.emit(self.key)

    def _write_back(self) -> None:
        """Display what was actually STORED, with signals blocked so it is not read as an edit."""
        widget = self.value_widget
        if widget is None:
            return
        value = self.controller.model.value(self.key)
        if value is None:
            return
        with no_signals(widget):
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, NoWheelSpinBox):
                widget.setValue(int(value))
            else:
                widget.setValue(float(value))
        if self._stepper is not None:
            self._stepper.refresh_step_limits()

    def _show_notice(self, text: str) -> None:
        self.notice.setText(text or "")
        self.notice.setVisible(bool(text))

    def flash_applied(self) -> None:
        """The 140ms amber border flash that says the write REACHED THE SENSOR.

        This is the only channel distinguishing "I typed it" from "the camera took it" -- the
        `setting_change` event goes to a file nobody is looking at while the run is going.
        """
        widget = self.value_widget
        if widget is None:
            return
        duration = theme.motion_ms(theme.FLASH_MS)
        widget.setProperty("applied", True)
        _restyle(widget)
        if duration <= 0:
            widget.setProperty("applied", False)
            _restyle(widget)
            return
        if self._flash_timer is None:
            self._flash_timer = QTimer(self)
            self._flash_timer.setSingleShot(True)
            self._flash_timer.timeout.connect(self._end_flash)
        self._flash_timer.start(duration)

    def _end_flash(self) -> None:
        if self.value_widget is not None:
            self.value_widget.setProperty("applied", False)
            _restyle(self.value_widget)

    # -- refresh -------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read everything that can change underneath this row: value, block state, badges."""
        self._write_back()
        reason = self.controller.block_reason(self.key)
        blocked = reason is not None
        if self._stepper is not None:
            self._stepper.setEnabled(not blocked)
            self._stepper.refresh_step_limits()
        elif self.value_widget is not None:
            self.value_widget.setEnabled(not blocked)
        # A GREYED CONTROL WITH NO REASON IS A SUPPORT CALL. The help line is replaced by the block
        # reason, which is the operator-readable sentence the pipeline already writes for exactly
        # this situation, rather than a second wording invented here.
        self.help.setText(reason if blocked else self.setting.help)
        self.help.setProperty("role", "blocked" if blocked else "help")
        _restyle(self.help)
        # WHICH KNOBS CAN I TURN RIGHT NOW, answered from across the room. A row that cannot be
        # changed at this moment drops its label to the secondary colour, on top of the badge and
        # the reason it already carries. Pure styling over the existing `block_reason` -- this adds
        # no gate, and invariant 3 is untouched.
        self.label.setProperty("live", "off" if blocked else "on")
        _restyle(self.label)
        self._sync_help_visibility()
        self.badge_unchecked.setVisible(self.key in self.controller.never_checked())
        self._refresh_spine(blocked)

    def _refresh_spine(self, blocked: bool) -> None:
        """A non-nullable app setting has no sensor to impose on, so it draws no spine at all."""
        self.spine.set_state(theme.spine_color(blocked=blocked, imposed=False, nullable=False),
                             "none")

    def rebuild_limits(self, setting: Setting) -> None:
        """Adopt new limits for the same key -- what a camera open or a start-only write produces.

        The limits change is real, not cosmetic: a new Width changes the legal ExposureTime and
        AcquisitionFrameRate ranges, and a spinbox still offering the old range would let an
        operator pick a value the SDK rejects outright.
        """
        self.setting = setting
        self._apply_limits()
        self.refresh()

    def _apply_limits(self) -> None:
        widget = self.value_widget
        if widget is None or isinstance(widget, QCheckBox):
            return
        setting = self.setting
        with no_signals(widget):
            if isinstance(widget, NoWheelSpinBox):
                widget.setRange(int(setting.lo), int(setting.hi))
                widget.setSingleStep(max(1, int(setting.step)))
            else:
                widget.setDecimals(_decimals_for(setting.step))
                widget.setRange(float(setting.lo), float(setting.hi))
                widget.setSingleStep(float(setting.step))


class NullableSettingRow(SettingRow):
    """A camera row: EITHER a green "camera default" label OR an editor. Never both, never neither.

    The editor is CONSTRUCTED when the row is armed and DELETED when it goes back to the default,
    so on a default row there is genuinely nothing in the widget tree to grab. See the module
    docstring for the two defects that made this necessary rather than merely tidy.
    """

    def _make_editor(self) -> Optional[QWidget]:
        # A nullable row starts wherever the model is, and the model starts at "camera default" on
        # every rig config that ships. The editor is built by `_ensure_editor` when armed.
        return None

    def _build_second_line(self, outer) -> None:
        self.hint_label.setContentsMargins(11, 0, 0, 0)
        outer.addWidget(self.hint_label)

    def _build_action(self, layout) -> None:
        """The revert glyph. A 16px "x" in the 28px action column, present only when imposed.

        The old version was a full "back to default" QPushButton capped at 150px, and it ate the
        row -- the width it gives back is what pays for the 15pt numeral inside the same 560px pane
        minimum. The full sentence survives as the tooltip, which is where it was already.
        """
        self.default_button = QToolButton()
        self.default_button.setText("×")
        self.default_button.setProperty("role", "revert")
        self.default_button.setFixedSize(20, 20)
        self.default_button.setToolTip(
            "Stop imposing this value. The camera KEEPS running at whatever was last sent - "
            "returning to '%s' only stops this software from sending it, now and at the next "
            "start." % DEFAULT_TEXT)
        self.default_button.clicked.connect(self._on_to_default)
        self.default_button.setVisible(False)
        layout.addStretch(1)
        layout.addWidget(self.default_button)

    def _value_container(self) -> QWidget:
        """The default page: a green label, what the sensor says, and the way in.

        `format_hint` supplies "camera: 88.5 fps" when a camera is open and NOTHING when one is
        not: a row that printed a hint with no camera attached would be claiming a reading nobody
        took.

        VISUALLY THIS CELL IS A HOLE, NOT A CONTROL -- no well, no border, no fill. Default is flat
        and open; imposed is boxed and lit. That is the shape half of the tri-state, and it is the
        half that survives a dim monitor.
        """
        self._holder = QWidget()
        self._holder_layout = QVBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(0, 0, 0, 0)
        self._holder_layout.setSpacing(0)

        self._default_page = QWidget()
        page = QHBoxLayout(self._default_page)
        page.setContentsMargins(0, 0, 0, 0)
        page.setSpacing(4)
        self.default_label = QLabel(DEFAULT_TEXT)
        self.default_label.setProperty("role", "default")
        self.default_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        page.addWidget(self.default_label, 1)
        self._holder_layout.addWidget(self._default_page)

        # THE HINT DOES NOT LIVE IN THIS CELL, and that was a rendering defect before it was a
        # decision. Inside the 168px value column, beside the words "camera default", the sensor
        # hint drew as "no camera - that i" -- clipped mid-word, on the line whose whole job is
        # saying that the number on the arm button is a GUESS rather than a reading (invariant 6).
        # It is built here, next to the state it describes, and PLACED on the row's full-width
        # second line by `_build`, where it wraps instead of clipping.
        self.hint_label = _wrapping_label("", "hint")

        # The arm affordance stays a small, deliberate, explicitly-clicked button. Making the whole
        # ~168px cell click-to-arm was considered and REJECTED: arming imposes a value, so a large
        # forgiving hit target is precisely the wrong shape for it. Small target for the action
        # that writes to the sensor; large area for the action that does not.
        self.arm_button = QPushButton("Set a value...")
        self.arm_button.setProperty("role", "ghost")
        self.arm_button.clicked.connect(self._on_arm)
        self.arm_button.setMaximumWidth(theme.VALUE_WIDTH)
        self._holder_layout.addWidget(self.arm_button)
        return self._holder

    # -- the two states ------------------------------------------------------------------------
    def _ensure_editor(self, wanted: bool) -> None:
        """Build or destroy the editor so the widget tree matches the state. Idempotent."""
        if wanted and self.value_widget is None:
            widget = make_value_widget(self.setting)
            self.value_widget = widget
            self._apply_limits()
            with no_signals(widget):
                value = self.controller.model.value(self.key)
                if value is not None:
                    if isinstance(widget, NoWheelSpinBox):
                        widget.setValue(int(value))
                    else:
                        widget.setValue(float(value))
            self._connect_editor(widget)
            self._stepper = StepperField(widget)
            self._holder_layout.insertWidget(0, self._stepper)
        elif not wanted and self.value_widget is not None:
            widget, self.value_widget = self.value_widget, None
            stepper, self._stepper = self._stepper, None
            # ORDER MATTERS: disconnect, then flag, then remove. Removing a focused spinbox emits
            # `editingFinished`, and before this was handled that signal re-imposed the value the
            # row had just been cleared of.
            self._disconnect_editor(widget)
            self._tearing_down = True
            try:
                if stepper is not None:
                    self._holder_layout.removeWidget(stepper)
                    stepper.setParent(None)
                    stepper.deleteLater()
                else:
                    self._holder_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
            finally:
                self._tearing_down = False
        armed = self.value_widget is not None
        self._default_page.setVisible(not armed)
        self.default_label.setVisible(not armed)
        self.hint_label.setVisible(not armed)
        self.arm_button.setVisible(not armed)
        self.default_button.setVisible(armed)

    def _on_arm(self) -> None:
        """Leave the default state. NOT confirmed, because with a camera open nothing physical
        changes -- the first value written equals what the sensor was already doing. With no camera
        open it IS a guess, and `arming_plan` has put that in the button's own label."""
        result = self.controller.arm(self.key)
        if result.blocked:
            self._show_notice(result.message)
            return
        self.refresh()
        self._show_notice("" if result.applied else result.message)
        self.edited.emit(self.key)

    def _on_to_default(self) -> None:
        """Return to "impose nothing". NOT a modal -- see `settings_controller.to_default_notice`.

        The camera keeps running at whatever was last written and cannot be told otherwise, so the
        sentence saying so goes ON THE ROW, in front of the operator, where a tuning loop that
        toggles this row repeatedly cannot train them to click it away.
        """
        # Computed BEFORE the value is cleared: it has to name the number the camera will keep
        # running at, and a moment later the row no longer holds it.
        notice = to_default_notice(self.setting, camera_open=self._camera_open(), recording=False)
        result = self.controller.to_default(self.key)
        if result.blocked:
            self._show_notice(result.message)
            return
        self.refresh()
        self._show_notice(notice)
        self.edited.emit(self.key)

    def _camera_open(self) -> bool:
        """Is a camera actually open right now? Asked of the controller, which was given a callable
        rather than a boolean -- the answer changes while this window is up."""
        probe = getattr(self.controller, "camera_open", None)
        try:
            return bool(probe() if callable(probe) else probe)
        except Exception:
            return False

    # -- refresh -------------------------------------------------------------------------------
    def refresh(self) -> None:
        self._ensure_editor(self.controller.model.value(self.key) is not None)
        plan = arming_plan(self.setting)
        # What the sensor says, when a camera is open; what the button would GUESS, when one is
        # not. Never both, and never nothing -- a bare "camera default" leaves the operator with no
        # idea what they are leaving alone.
        self.hint_label.setText(format_hint(self.setting) or plan.warning)
        self.hint_label.setProperty("role", "notice" if plan.blind and plan.warning else "hint")
        _restyle(self.hint_label)
        self.arm_button.setText(plan.button_text)
        self.arm_button.setToolTip(plan.tooltip)
        super().refresh()
        reason = self.controller.block_reason(self.key)
        self.arm_button.setEnabled(reason is None)
        self.default_button.setEnabled(reason is None)

    def _refresh_spine(self, blocked: bool) -> None:
        imposed = self.controller.model.value(self.key) is not None
        shape = "thin" if blocked else ("full" if imposed else "tick")
        self.spine.set_state(
            theme.spine_color(blocked=blocked, imposed=imposed, nullable=True), shape)

    def rebuild_limits(self, setting: Setting) -> None:
        super().rebuild_limits(setting)
        self.hint_label.setText(format_hint(setting))


def _restyle(widget: QWidget) -> None:
    """Make a changed `role` property take effect -- Qt does not re-evaluate the stylesheet itself."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
