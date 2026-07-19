"""The tri-state widget, the wheel guard, and invariant 2 re-proved as an EXISTENCE claim.

The cv2 panel proved invariant 2 with geometry: a default row draws an empty track, so a click
lands on nothing. That test asserts on rectangles, and it will not protect a Qt widget. The Qt
version is stronger and simpler -- a row at "camera default" HAS NO EDITOR IN IT. There is nothing
to click, wheel, tab into, or drive programmatically, and the assertion is
`findChild(QAbstractSpinBox) is None`, which survives any layout change.

Offscreen throughout (`QT_QPA_PLATFORM=offscreen`, set in conftest before PySide6 is imported).
There is no camera on this machine, so every camera here is a fake and what is asserted is the
CALL PATTERN -- what would be sent, and when nothing would be.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QAbstractSpinBox, QSlider

from flygym_tracker.config import load_config
from flygym_tracker.gui.setting_row import NullableSettingRow, SettingRow
from flygym_tracker.settings_controller import (NEVER_CHECKED_BADGE, SettingsController,
                                                camera_block_reason)
from flygym_tracker.settings_model import DEFAULT_TEXT, build_app_settings, build_camera_settings

CAMERA_KEYS = ["source.camera.frame_rate", "source.camera.exposure_us", "source.camera.gain_db",
               "source.camera.width", "source.camera.height"]


class FakeCamera:
    def __init__(self, *, acquiring=False, values=None):
        self.is_acquiring = acquiring
        self._values = dict(values or {})

    def current_values(self):
        return dict(self._values)

    def ranges(self):
        from flygym_tracker.frame_source import fallback_camera_ranges

        return fallback_camera_ranges()

    # `build_camera_settings` duck-types "is this a live camera" on `set_frame_rate` being
    # callable, so a fake without these is treated as no camera at all -- which silently turned a
    # hints test into a no-hints test.
    def set_frame_rate(self, value):
        pass

    def set_exposure_us(self, value):
        pass

    def set_gain_db(self, value):
        pass


@pytest.fixture
def config():
    return load_config(path="config/flygym_rig.yaml")


@pytest.fixture
def controller(config):
    return SettingsController(build_app_settings(config))


def make_row(controller, key, parent=None):
    setting = controller.model.get(key)
    cls = NullableSettingRow if setting.nullable else SettingRow
    row = cls(setting, controller)
    row.show()
    return row


def wheel(widget, notches=-1):
    """Send one wheel notch at a widget. Returns whether the event was accepted."""
    pos = QPointF(widget.rect().center())
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, notches * 120),
                        QPoint(0, notches * 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    from PySide6.QtWidgets import QApplication

    QApplication.instance().sendEvent(widget, event)
    return event.isAccepted()


# =============================================================================================
# INVARIANT 2 -- a row at the camera default has NOTHING TO GRAB
# =============================================================================================
@pytest.mark.parametrize("key", CAMERA_KEYS)
def test_a_row_at_the_camera_default_contains_no_editor_at_all(qapp, controller, key):
    """Not "the editor is disabled" and not "the editor is empty" -- there is no editor in the
    widget tree. That is what makes this structural rather than defended."""
    row = make_row(controller, key)
    assert controller.model.value(key) is None
    assert row.findChild(QAbstractSpinBox) is None
    assert row.findChild(QSlider) is None, "a slider is a pixel-to-value map over a live sensor"


@pytest.mark.parametrize("key", CAMERA_KEYS)
def test_wheeling_over_every_widget_of_a_default_row_imposes_nothing(qapp, controller, key):
    """THE ORIGINAL NEAR-MISS, in its Qt form. A stray interaction on a default row once imposed
    2.6 fps on a live 88 fps recording."""
    row = make_row(controller, key)
    for widget in [row] + row.findChildren(object):
        if hasattr(widget, "rect") and hasattr(widget, "mapToGlobal"):
            wheel(widget)
    qapp.processEvents()
    assert controller.model.value(key) is None


@pytest.mark.parametrize("key", CAMERA_KEYS)
def test_clicking_and_typing_at_a_default_row_imposes_nothing(qapp, controller, key):
    """Clicks anywhere on the row, and arrow keys at it, must not arm it. Only the button does."""
    row = make_row(controller, key)
    for point in (row.rect().topLeft(), row.rect().center(), row.rect().bottomRight()):
        QTest.mouseClick(row, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    for key_code in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Right, Qt.Key.Key_Left):
        QTest.keyClick(row, key_code)
    qapp.processEvents()
    assert controller.model.value(key) is None


def test_tabbing_through_a_form_of_default_rows_imposes_nothing(qapp, controller):
    """Keyboard traversal must not arm anything either -- an operator finding their way around the
    window with Tab is not making a decision about the sensor."""
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    host = QWidget()
    layout = QVBoxLayout(host)
    for key in CAMERA_KEYS:
        layout.addWidget(make_row(controller, key, host))
    host.show()
    for _ in range(24):
        QTest.keyClick(host, Qt.Key.Key_Tab)
    qapp.processEvents()
    assert [controller.model.value(k) for k in CAMERA_KEYS] == [None] * 5


# =============================================================================================
# The tri-state's four channels
# =============================================================================================
def test_a_default_row_says_camera_default_in_words(qapp, controller):
    row = make_row(controller, "source.camera.frame_rate")
    assert row.default_label.text() == DEFAULT_TEXT
    assert row.value_widget is None


def test_a_default_row_never_claims_a_camera_reading_when_none_was_taken(qapp, controller):
    """INVARIANT 6 at widget level: printing "camera: 88.5 fps" with nothing attached would be the
    display inventing a measurement. With no camera the line says the opposite -- that the number
    on the arm button is the lowest legal value rather than something a sensor reported."""
    row = make_row(controller, "source.camera.frame_rate")
    assert "camera:" not in row.hint_label.text()
    assert "not a reading" in row.hint_label.text()


def test_a_default_row_shows_what_the_sensor_reports_when_one_IS_open(qapp, config):
    settings, notes = build_camera_settings(config, FakeCamera(values={"frame_rate": 88.5}))
    controller = SettingsController(
        __import__("flygym_tracker.settings_model", fromlist=["SettingsModel"]).SettingsModel(
            settings, group_notes=notes))
    row = make_row(controller, "source.camera.frame_rate")
    assert "88.5" in row.hint_label.text()
    assert row.hint_label.text().startswith("camera:")


def test_arming_a_row_swaps_the_label_for_an_editor(qapp, controller):
    row = make_row(controller, "source.camera.gain_db")
    assert row.findChild(QAbstractSpinBox) is None
    row.arm_button.click()
    qapp.processEvents()
    assert row.value_widget is not None
    assert row.findChild(QAbstractSpinBox) is not None


def test_going_back_to_default_removes_the_editor_again(qapp, controller):
    row = make_row(controller, "source.camera.gain_db")
    row.arm_button.click()
    qapp.processEvents()
    row.default_button.click()
    qapp.processEvents()
    assert controller.model.value("source.camera.gain_db") is None
    assert row.value_widget is None
    assert row.findChild(QAbstractSpinBox) is None


def test_going_back_to_default_with_a_camera_open_puts_the_warning_on_the_row(qapp, controller):
    """NOT a modal. This is the action a tuning loop repeats, and a dialog on a repeated action is
    a dialog that gets dismissed unread -- which then trains the same reflex on the two that
    matter. The sentence goes in front of the operator instead."""
    controller.camera_open = lambda: True
    row = make_row(controller, "source.camera.frame_rate")
    row.arm_button.click()
    row.value_widget.setValue(30.0)
    qapp.processEvents()
    row.default_button.click()
    qapp.processEvents()
    assert row.notice.isVisible()
    assert "KEEP running at 30.0 fps" in row.notice.text()


def test_with_the_camera_closed_that_transition_says_nothing_at_all(qapp, controller):
    """Nothing was ever sent, so there is nothing to un-send. Keeping the notice rare is what keeps
    it readable."""
    controller.camera_open = lambda: False
    row = make_row(controller, "source.camera.frame_rate")
    row.arm_button.click()
    qapp.processEvents()
    row.default_button.click()
    qapp.processEvents()
    assert row.notice.text() == ""


# =============================================================================================
# The arming hazard, at the widget
# =============================================================================================
def test_the_arm_button_names_its_landing_value_before_it_is_pressed(qapp, controller):
    """With no camera the landing value is the lowest legal one, and saving 0.1 fps into the rig
    config would ruin the next experiment. It is not hidden behind a tooltip -- it is the label."""
    row = make_row(controller, "source.camera.frame_rate")
    assert "0.1 fps" in row.arm_button.text()
    # ...and the reason that number is a guess is on the same row, beside the button, not hidden
    # in the tooltip. (It moved off the button itself because the long label forced a horizontal
    # scrollbar that pushed the value controls off the right edge -- see `arming_plan`.)
    assert "no camera" in row.hint_label.text()
    assert NEVER_CHECKED_BADGE in row.arm_button.toolTip()


def test_a_row_armed_with_no_camera_wears_the_never_checked_badge(qapp, controller):
    row = make_row(controller, "source.camera.frame_rate")
    assert row.badge_unchecked.isVisible() is False
    row.arm_button.click()
    qapp.processEvents()
    assert row.badge_unchecked.isVisible() is True
    # SHORT ON THE ROW, FULL ON HOVER. The whole phrase between the label and the editor pushed the
    # editor off the pane (see tests/test_gui_layout.py), so the row carries the short form and the
    # full wording lives in the tooltip -- and, at full length, in the readiness strip and in the
    # sentence the save confirmation shows, which both have a line each to spend.
    assert row.badge_unchecked.text() == "unchecked"
    assert row.badge_unchecked.toolTip() == NEVER_CHECKED_BADGE


def test_the_arm_button_with_a_camera_open_names_what_the_camera_is_doing(qapp, config):
    from flygym_tracker.settings_model import SettingsModel

    settings, notes = build_camera_settings(config, FakeCamera(values={"exposure_us": 5000.0}))
    controller = SettingsController(SettingsModel(settings, group_notes=notes))
    row = make_row(controller, "source.camera.exposure_us")
    assert "5000" in row.arm_button.text()
    assert row.hint_label.text().startswith("camera:")


# =============================================================================================
# The wheel guard -- two-sided, because a guard that kills real editing is a different bug
# =============================================================================================
def test_the_wheel_does_not_edit_an_unfocused_spinbox(qapp, controller):
    """MEASURED HAZARD: a plain QDoubleSpinBox at 50.0, unfocused, went to 49.0 on one notch."""
    row = make_row(controller, "activity.pixel_threshold")
    row.value_widget.clearFocus()
    before = row.value_widget.value()
    accepted = wheel(row.value_widget)
    qapp.processEvents()
    assert row.value_widget.value() == before
    assert accepted is False, "the event must PROPAGATE so the scroll area can still scroll"


def test_the_wheel_still_edits_a_FOCUSED_spinbox(qapp, controller):
    """The other side of the same guard. An unconditional event filter measured 50.0 -> 50.0 even
    when focused, i.e. the wheel never edits at all -- which is a different bug, not a fix."""
    row = make_row(controller, "activity.pixel_threshold")
    row.value_widget.setFocus()
    qapp.processEvents()
    before = row.value_widget.value()
    wheel(row.value_widget, notches=-1)
    qapp.processEvents()
    assert row.value_widget.value() != before


def test_arrow_keys_and_the_step_buttons_still_work(qapp, controller):
    """The guard is keyed on the wheel only: typing, arrows and the step arrows must be untouched."""
    row = make_row(controller, "activity.pixel_threshold")
    row.value_widget.setFocus()
    before = row.value_widget.value()
    QTest.keyClick(row.value_widget, Qt.Key.Key_Up)
    qapp.processEvents()
    assert row.value_widget.value() > before


# =============================================================================================
# The commit loop
# =============================================================================================
def test_typing_a_value_stores_the_COERCED_value_and_displays_that(qapp, controller):
    """The widget is never the source of truth. `pixel_threshold` caps at 60."""
    row = make_row(controller, "activity.pixel_threshold")
    row.value_widget.setValue(999.0)
    qapp.processEvents()
    assert controller.model.value("activity.pixel_threshold") == 60.0
    assert row.value_widget.value() == 60.0


def test_the_write_back_does_not_re_enter_the_funnel_as_a_fresh_edit(qapp, controller):
    """MEASURED: a programmatic `setValue` emits `valueChanged`. Without `no_signals` around the
    write-back, one operator edit becomes two commits -- and on a camera row, two SDK writes and
    two `setting_change` events."""
    seen = []
    controller.on_change = lambda key, value: seen.append(value)
    row = make_row(controller, "activity.pixel_threshold")
    row.value_widget.setValue(999.0)          # coerces to 60.0, which is then written back
    qapp.processEvents()
    assert seen == [60.0]


def test_keyboard_tracking_is_off_on_every_spinbox(qapp, controller):
    """With tracking ON, typing "5000" into exposure emits at 5, 50, 500, 5000: four SDK writes
    walking the sensor through three exposures nobody asked for, and four `setting_change` rows in
    events.csv for one edit."""
    for key in ("activity.pixel_threshold", "rotation.debounce_frames"):
        row = make_row(controller, key)
        assert row.value_widget.keyboardTracking() is False


def test_a_row_emits_edited_only_when_the_value_actually_moved(qapp, controller):
    row = make_row(controller, "activity.pixel_threshold")
    spy = QSignalSpy(row.edited)
    row.value_widget.setValue(20.0)
    qapp.processEvents()
    assert spy.count() == 1
    row.value_widget.setValue(20.0)          # the same value again
    qapp.processEvents()
    assert spy.count() == 1


# =============================================================================================
# Blocked rows
# =============================================================================================
def test_a_blocked_row_is_disabled_AND_explains_why_in_place_of_its_help_line(qapp, config):
    """A greyed control with no reason is a support call."""
    camera = FakeCamera(acquiring=True)
    controller = SettingsController(build_app_settings(config),
                                    block_reason=camera_block_reason(lambda: camera))
    row = make_row(controller, "source.camera.width")
    assert row.arm_button.isEnabled() is False
    assert "close it" in row.help.text()

    # And it re-asserts itself: something that enables the button behind the row's back is undone
    # at the next refresh, rather than leaving a live control over a blocked setting.
    row.arm_button.setEnabled(True)
    row.refresh()
    assert row.arm_button.isEnabled() is False


def test_a_forced_commit_on_a_blocked_row_is_still_refused(qapp, config):
    """DISABLING IS NOT ENFORCEMENT -- measured: a programmatic `setValue` on a disabled QSpinBox
    succeeds and emits. So the guard cannot live in `setEnabled`, and this drives the row the way a
    stray programmatic write would."""
    camera = FakeCamera(acquiring=True)
    calls = []
    controller = SettingsController(build_app_settings(config),
                                    block_reason=camera_block_reason(lambda: camera),
                                    on_change=lambda k, v: calls.append((k, v)))
    controller.commit("source.camera.width", 640)      # blocked: stores nothing
    assert controller.model.value("source.camera.width") is None
    assert calls == []


def test_the_start_only_badge_is_on_width_and_height_and_nowhere_else(qapp, controller):
    for key in ("source.camera.width", "source.camera.height"):
        assert make_row(controller, key).badge_start_only.isVisible() is True
    for key in ("source.camera.frame_rate", "activity.pixel_threshold"):
        assert make_row(controller, key).badge_start_only.isVisible() is False


def test_a_next_run_row_says_so_on_its_own_face(qapp, controller):
    """`bin_seconds` cannot be applied to a run in progress, and the row says it rather than
    looking like it did something."""
    assert make_row(controller, "binning.bin_seconds").badge_next_run.isVisible() is True
