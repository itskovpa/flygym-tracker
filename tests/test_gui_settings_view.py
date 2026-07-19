"""The whole settings pane: groups, notes, the default counter, the disable matrix -- and
INVARIANT 1 re-proved through the app's own code path against the real fake SDK.

The existing control test (`test_frame_source.py::test_a_config_with_every_camera_setting_unset_
sends_nothing_to_the_camera`) proves that `_configure` writes nothing when all five settings are
null. That is the SDK layer. What this file proves is that the GUI does not undo it: a whole
settings window, built and abused, still sends nothing -- asserted on the same `FakeSdk` the
frame_source tests use, by importing it rather than writing a second one that could be wrong in a
convenient direction.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QSlider

from flygym_tracker.config import load_config
from flygym_tracker.frame_source import HikCameraSource
from flygym_tracker.gui.settings_view import SettingsView
from flygym_tracker.settings_controller import SettingsController, camera_block_reason
from flygym_tracker.settings_model import DEFAULT_TEXT, build_app_settings

# The REAL fake SDK, imported from the frame_source tests. A second copy here could be wrong in a
# way that made this test pass -- e.g. by not recording a setter call at all.
from test_frame_source import SERIAL, FakeSdk, _SETTING_WRITE_CALLS

CAMERA_KEYS = ["source.camera.frame_rate", "source.camera.exposure_us", "source.camera.gain_db",
               "source.camera.width", "source.camera.height"]


@pytest.fixture
def config():
    return load_config(path="config/flygym_rig.yaml")


@pytest.fixture
def sdk(monkeypatch):
    fake = FakeSdk()
    monkeypatch.setattr(HikCameraSource, "_import_sdk", staticmethod(lambda: fake))
    return fake


def wheel(widget, notches=-1):
    pos = QPointF(widget.rect().center())
    event = QWheelEvent(pos, widget.mapToGlobal(pos), QPoint(0, notches * 120),
                        QPoint(0, notches * 120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.instance().sendEvent(widget, event)


def make_view(config, camera=None):
    controller = SettingsController(build_app_settings(config),
                                    block_reason=camera_block_reason(lambda: camera))
    view = SettingsView(controller)
    view.show()
    return view, controller


# =============================================================================================
# INVARIANT 1 -- a whole window of default rows, abused, still sends nothing
# =============================================================================================
def test_building_and_abusing_a_settings_window_writes_nothing_to_the_camera(qapp, config, sdk):
    """THE HEADLINE, at GUI level. A camera is opened with every setting null -- so `_configure`
    must issue exactly two calls, PixelFormat and TriggerMode -- and then the settings window is
    clicked, keyed, wheeled and tabbed through. Nothing may reach a setter."""
    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    try:
        view, controller = make_view(config, camera)
        for key in CAMERA_KEYS:
            row = view.rows[key]
            for widget in [row] + row.findChildren(object):
                if hasattr(widget, "rect") and hasattr(widget, "mapToGlobal"):
                    wheel(widget)
                    QTest.mouseClick(widget, Qt.MouseButton.LeftButton,
                                     Qt.KeyboardModifier.NoModifier, widget.rect().center())
            for code in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right):
                QTest.keyClick(row, code)
        for _ in range(40):
            QTest.keyClick(view, Qt.Key.Key_Tab)
        qapp.processEvents()

        for call in sdk.calls:
            assert call[0] not in _SETTING_WRITE_CALLS, \
                "the settings window wrote to the camera: %r" % (call,)
        for node in ("Width", "Height", "ExposureTime", "Gain", "AcquisitionFrameRate",
                     "AcquisitionFrameRateEnable", "ExposureAuto", "GainAuto"):
            assert sdk.sets_for(node) == [], "%s was set by the GUI" % node
        assert [controller.model.value(k) for k in CAMERA_KEYS] == [None] * 5
    finally:
        camera.close()


def test_no_camera_row_in_a_default_window_contains_anything_to_grab(qapp, config):
    view, _controller = make_view(config)
    for key in CAMERA_KEYS:
        row = view.rows[key]
        assert row.findChild(QAbstractSpinBox) is None, "%s has an editor" % key
        assert row.findChild(QSlider) is None


def test_no_slider_exists_anywhere_in_the_settings_pane(qapp, config):
    """The cv2 panel's original sin, kept out by assertion rather than by intention."""
    view, _ = make_view(config)
    assert view.findChildren(QSlider) == []


# =============================================================================================
# Groups, notes and the counter
# =============================================================================================
def test_there_is_one_group_box_per_model_group_in_model_order(qapp, config):
    view, controller = make_view(config)
    titles = [box.title() for box in view._group_boxes.values()]
    assert [t.split(" - ")[0] for t in titles] == controller.model.groups()


def test_the_camera_group_note_says_the_limits_are_not_live_when_no_camera_is_open(qapp, config):
    """INVARIANT 6: a documented range must not be presented with the authority of a measured one."""
    view, _ = make_view(config)
    assert "not live" in view._group_notes["Camera"].text()


def test_the_group_title_counts_the_rows_left_at_the_camera_default(qapp, config):
    view, controller = make_view(config)
    assert view._group_boxes["Camera"].title() == "Camera - 5 of 5 left at %s" % DEFAULT_TEXT
    view.rows["source.camera.gain_db"].arm_button.click()
    qapp.processEvents()
    assert view._group_boxes["Camera"].title() == "Camera - 4 of 5 left at %s" % DEFAULT_TEXT


# =============================================================================================
# The change banner
# =============================================================================================
def test_save_and_reset_are_disabled_until_something_changes(qapp, config):
    view, _ = make_view(config)
    assert view.save_button.isEnabled() is False
    assert view.reset_button.isEnabled() is False
    assert view.change_label.text() == "no changes"

    view.rows["activity.pixel_threshold"].value_widget.setValue(20.0)
    qapp.processEvents()
    assert view.save_button.isEnabled() is True
    assert view.change_label.text() == "1 unsaved change"


def test_reset_puts_every_row_back_and_the_banner_with_it(qapp, config):
    view, controller = make_view(config)
    view.rows["activity.pixel_threshold"].value_widget.setValue(20.0)
    qapp.processEvents()
    view.reset_button.click()
    qapp.processEvents()
    assert controller.changed() == []
    assert view.change_label.text() == "no changes"


def test_the_save_result_is_shown_because_the_file_is_not_on_screen(qapp, config, tmp_path):
    """A save that produced no visible output would leave an operator unable to tell a write from a
    silently skipped one, and the next run becomes a mystery."""
    view, controller = make_view(config)
    path = tmp_path / "rig.yaml"
    path.write_text("activity:\n  pixel_threshold: 12.0  # measured\n", encoding="utf-8")
    controller.config_path = str(path)
    view.rows["activity.pixel_threshold"].value_widget.setValue(20.0)
    qapp.processEvents()
    result = controller.save(confirm=lambda text: True)
    view.set_status(result.message)
    assert "wrote 1 change" in view.change_label.text()
    assert str(path) in view.change_label.text()
    assert "# measured" in path.read_text(encoding="utf-8")


# =============================================================================================
# The disable matrix
# =============================================================================================
def test_width_and_height_are_disabled_while_acquiring_and_say_why(qapp, config, sdk):
    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    try:
        view, _ = make_view(config, camera)
        for key in ("source.camera.width", "source.camera.height"):
            row = view.rows[key]
            assert row.arm_button.isEnabled() is False
            assert "close it" in row.help.text()
    finally:
        camera.close()


def test_the_live_rows_stay_enabled_while_acquiring(qapp, config, sdk):
    """Blocking exposure mid-stream would defeat the point of having a preview beside it."""
    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    try:
        view, _ = make_view(config, camera)
        for key in ("source.camera.frame_rate", "source.camera.exposure_us",
                    "source.camera.gain_db"):
            assert view.rows[key].arm_button.isEnabled() is True
    finally:
        camera.close()


def test_nothing_in_the_settings_path_ever_stops_or_restarts_the_stream(qapp, config, sdk):
    """INVARIANT 3's other half. Restarting the stream to apply a geometry change would cost a gap
    in a days-long recording AND a frame-diff baseline reset -- two incomparable regimes in one
    file with nothing marking the seam."""
    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    sdk.calls.clear()
    try:
        view, controller = make_view(config, camera)
        controller.commit("source.camera.width", 640)          # refused
        view.rows["source.camera.frame_rate"].arm_button.click()
        view.reset_button.click()
        qapp.processEvents()
        assert [c for c in sdk.calls if c[0] in ("StopGrabbing", "StartGrabbing")] == []
    finally:
        camera.close()


# =============================================================================================
# Rebuilding from live limits
# =============================================================================================
def test_opening_a_camera_replaces_the_documented_limits_with_the_sensors_own(qapp, config, sdk):
    """`ranges()` caches at open, and a spinbox built from the fallbacks can offer a value the SDK
    REJECTS -- on a start-only node that is a failed run, not a slightly wrong picture.

    Asserted on the sensor's OWN numbers rather than on "something changed": the fake SDK could
    legitimately report the same limits as the datasheet, and a test that only checked for a
    difference would pass whether or not the live values were read at all.
    """
    view, controller = make_view(config)
    assert "not live" in view._group_notes["Camera"].text()

    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    try:
        live = camera.ranges()
        view.rebuild_camera_rows(config, camera)
        width = controller.model.get("source.camera.width")
        assert width.hi == pytest.approx(live["Width"].hi)
        assert width.step == pytest.approx(live["Width"].inc)
        assert "limits read from the camera" in view._group_notes["Camera"].text()
    finally:
        camera.close()


def test_rebuilding_limits_never_overwrites_a_value_the_operator_chose(qapp, config, sdk):
    """Adopting the sensor's current values here would be the "impose whatever it happens to be
    doing" behaviour the whole tri-state exists to prevent."""
    view, controller = make_view(config)
    view.rows["source.camera.gain_db"].arm_button.click()
    controller.commit("source.camera.gain_db", 3.0)
    qapp.processEvents()

    camera = HikCameraSource(serial=SERIAL)
    camera.open()
    try:
        view.rebuild_camera_rows(config, camera)
        assert controller.model.value("source.camera.gain_db") == pytest.approx(3.0)
    finally:
        camera.close()


def test_a_value_outside_the_fresh_limits_stays_reachable(qapp, config):
    """A row the config set to 88 fps must not become uneditable because a fresh range stops
    lower -- the panel must always be able to show the number the run is actually using."""
    view, controller = make_view(config)
    view.rows["source.camera.frame_rate"].arm_button.click()
    controller.commit("source.camera.frame_rate", 88.0)
    qapp.processEvents()
    view.rebuild_camera_rows(config, None)
    setting = controller.model.get("source.camera.frame_rate")
    assert setting.lo <= 88.0 <= setting.hi
