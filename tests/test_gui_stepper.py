"""The step buttons, tested THE WAY THE BUG ARRIVED: by clicking the control's own pixels.

THE DEFECT THIS FILE EXISTS FOR. On the rig, with a real mouse, clicking the UP arrow of a settings
spinbox did nothing -- three clicks, 12.0 unchanged. Clicking DOWN worked: 12.0 -> 11.5.

EVERY OFFSCREEN TEST PASSED, and would have gone on passing forever. `tests/test_gui_setting_row.py`
proves the step arrows work with `QTest.keyClick(widget, Qt.Key.Key_Up)`, and the keyboard path was
never broken. The mouse path was, because Qt DREW the two arrows side by side (Windows 11 style)
while `subControlRect` still hit-tested them stacked vertically -- measured up=(138, 0, 14, 15),
down=(138, 15, 14, 15). The pixel the operator aimed at was not the pixel Qt tested.

So the lesson is the same one `tests/test_gui_layout.py` records, one level down. Driving a control
by its METHODS proves the method works. It says nothing about whether a mouse can reach it. The
assertions here are therefore:

    1. the increment control EXISTS as a real widget (a stylesheet sub-control has no QWidget, so
       there was previously nothing to assert this about at all),
    2. its rect is non-empty and big enough to hit,
    3. `QTest.mouseClick` AT ITS OWN CENTRE raises the value -- and the same for decrement.

(3) is the assertion that was impossible before this change and would have caught the bug on the
day it was introduced.

STATED PLAINLY: offscreen, `QTest.mouseClick` posts a synthetic event at a widget-local point, so
this proves the button's own geometry is coherent and wired. It does NOT reproduce the platform's
real hit-test walk, which is the thing that actually failed. What makes the bug impossible is
structural -- there is now exactly ONE rectangle per step, owned by a QWidget, so "drawn here,
tested there" has nowhere to happen. The rig click test is still the last word.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox

from flygym_tracker.config import load_config
from flygym_tracker.gui.setting_row import NullableSettingRow, SettingRow
from flygym_tracker.gui.stepper import StepperField
from flygym_tracker.settings_controller import SettingsController
from flygym_tracker.settings_model import build_app_settings


@pytest.fixture
def controller():
    return SettingsController(build_app_settings(load_config(path="config/flygym_rig.yaml")))


def make_row(controller, key):
    setting = controller.model.get(key)
    cls = NullableSettingRow if setting.nullable else SettingRow
    row = cls(setting, controller)
    row.show()
    return row


def armed_row(controller, key):
    """A camera row with its editor built, which is the only state that has step buttons."""
    row = make_row(controller, key)
    row.arm_button.click()
    return row


def click_centre(widget) -> None:
    """Click the widget at the centre of ITS OWN rect -- the pixel an operator aims at."""
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                     widget.rect().center())


# =============================================================================================
# THE REGRESSION: clicking the increment control must raise the value
# =============================================================================================
def test_the_increment_control_is_a_real_widget_with_a_real_rect(qapp, controller):
    """A stylesheet ::up-button has no QWidget, so this could not previously be asserted at all."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    assert stepper is not None, "the value cell has no stepper"
    assert stepper.up is not None and stepper.down is not None
    for button in (stepper.up, stepper.down):
        assert button.rect().isValid() and not button.rect().isEmpty()
        assert button.width() >= 20 and button.height() >= 20, \
            "%dx%d is too small to aim at" % (button.width(), button.height())


def test_clicking_the_up_button_at_its_own_centre_raises_the_value(qapp, controller):
    """THE BUG, ASSERTED. On the rig this went 12.0 -> 12.0 -> 12.0 for three clicks."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    before = row.value_widget.value()
    click_centre(stepper.up)
    qapp.processEvents()
    assert row.value_widget.value() > before, \
        "up button click did nothing: %r -> %r" % (before, row.value_widget.value())


def test_clicking_the_down_button_at_its_own_centre_lowers_the_value(qapp, controller):
    """The half that WORKED on the rig. It must keep working -- a fix that breaks the good arrow
    to fix the dead one is not a fix, and overriding Qt's sub-control geometry wholesale is exactly
    how both arrows end up with zero-size hit rects."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    row.value_widget.setValue(20.0)
    qapp.processEvents()
    before = row.value_widget.value()
    click_centre(stepper.down)
    qapp.processEvents()
    assert row.value_widget.value() < before


def test_both_step_buttons_move_the_value_by_the_settings_own_step(qapp, controller):
    """Not an arbitrary amount: the step is the one the `Setting` declares, so stepping is the same
    size whether it came from the keyboard, the wheel or the mouse."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    row.value_widget.setValue(20.0)
    qapp.processEvents()
    step = row.value_widget.singleStep()
    click_centre(stepper.up)
    qapp.processEvents()
    assert row.value_widget.value() == pytest.approx(20.0 + step)


def test_a_step_click_reaches_the_controller_and_is_stored(qapp, controller):
    """The button is not a display trick: it goes through the same commit funnel a typed edit does,
    so a mid-run step is applied AND logged exactly like every other change."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    row.value_widget.setValue(20.0)
    qapp.processEvents()
    click_centre(stepper.up)
    qapp.processEvents()
    assert controller.model.value("activity.pixel_threshold") == row.value_widget.value()


# =============================================================================================
# The native sub-controls are OFF, which is what makes the bug structurally impossible
# =============================================================================================
def test_the_spinbox_draws_no_native_step_arrows_at_all(qapp, controller):
    """With `NoButtons` there is no sub-control for Qt to draw in one place and hit-test in
    another. That mismatch was the whole bug, and this is the line that removes it."""
    row = make_row(controller, "activity.pixel_threshold")
    assert row.value_widget.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons


def test_step_buttons_never_steal_focus_from_the_field(qapp, controller):
    """A focus change on a spinbox emits `editingFinished`, which the row treats as an operator
    edit -- so a focus-taking [+] would enter the commit funnel twice for one click."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    assert stepper.up.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert stepper.down.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_one_click_on_a_step_button_is_exactly_one_commit(qapp, controller):
    """On a camera row a double commit is two SDK writes and two `setting_change` rows for one
    click. The measured cause of that class of defect is the write-back re-entering the funnel."""
    seen = []
    controller.on_change = lambda key, value: seen.append(value)
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    row.value_widget.setValue(20.0)
    qapp.processEvents()
    seen.clear()
    click_centre(stepper.up)
    qapp.processEvents()
    assert len(seen) == 1, "one click produced %d commits: %r" % (len(seen), seen)


# =============================================================================================
# INVARIANT 2 -- the steppers must not exist on a row at the camera default
# =============================================================================================
def test_a_camera_row_at_its_default_has_no_step_buttons_to_click(qapp, controller):
    """The steppers are built WITH the editor and destroyed with it. If they outlived it, a click
    on [+] would impose a value on a row whose whole point is that nothing can be grabbed -- which
    is invariant 2's original near-miss rebuilt out of the fix for a different bug."""
    row = make_row(controller, "source.camera.frame_rate")
    assert row.findChild(StepperField) is None
    assert row.findChild(QAbstractSpinBox) is None


def test_arming_builds_the_steppers_and_reverting_removes_them_again(qapp, controller):
    row = armed_row(controller, "source.camera.gain_db")
    qapp.processEvents()
    assert row.findChild(StepperField) is not None
    row.default_button.click()
    qapp.processEvents()
    assert row.findChild(StepperField) is None, "a step button survived the return to default"
    assert row.findChild(QAbstractSpinBox) is None


def test_stepping_an_armed_camera_row_moves_the_sensor_value(qapp, controller):
    """End to end on the surface the operator actually tunes: arm, click [+], value moves."""
    row = armed_row(controller, "source.camera.gain_db")
    qapp.processEvents()
    stepper = row.findChild(StepperField)
    before = row.value_widget.value()
    click_centre(stepper.up)
    qapp.processEvents()
    assert row.value_widget.value() > before
    assert controller.model.value("source.camera.gain_db") == row.value_widget.value()


# =============================================================================================
# Blocked rows
# =============================================================================================
def test_a_blocked_row_disables_the_step_buttons_too(qapp):
    """A live [+] over a disabled field is a control that looks like it works and does not, which
    is the same operator experience as the bug this module was written to end."""
    from flygym_tracker.settings_controller import camera_block_reason

    class Acquiring:
        is_acquiring = True

        def ranges(self):
            from flygym_tracker.frame_source import fallback_camera_ranges

            return fallback_camera_ranges()

    config = load_config(path="config/flygym_rig.yaml")
    controller = SettingsController(build_app_settings(config),
                                    block_reason=camera_block_reason(lambda: Acquiring()))
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    assert stepper.up.isEnabled() is True        # this key is not blocked

    row.controller.block_reason = lambda key: "the stream is running"
    row.refresh()
    assert stepper.up.isEnabled() is False
    assert stepper.down.isEnabled() is False


def test_the_up_button_greys_out_at_the_top_of_the_range(qapp, controller):
    """At the maximum the old native arrow silently did nothing, which is indistinguishable from
    the dead-arrow bug. Saying so is the difference between "broken" and "at the limit"."""
    row = make_row(controller, "activity.pixel_threshold")
    stepper = row.findChild(StepperField)
    row.value_widget.setValue(row.value_widget.maximum())
    qapp.processEvents()
    row.refresh()
    assert stepper.up.isEnabled() is False
    assert stepper.down.isEnabled() is True
