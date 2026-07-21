"""Choosing which physical camera to use, on a machine that is not the one it was configured on.

THE FAILURE THESE TESTS PIN DOWN. The shipped config carried `serial: "DA4282883"` -- the serial of
the development rig's own camera. On a second PC the app said "camera could not be opened" while
that machine's camera worked fine in HikRobot's MVS Viewer minutes earlier, and there was no way to
see what WAS attached, and no way to pick it. The cure required knowing that a YAML file somewhere
pinned a serial belonging to hardware on another desk.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from flygym_tracker.frame_source import CameraInfo  # noqa: E402
from flygym_tracker.gui.camera_picker import ANY_CAMERA, CameraPicker  # noqa: E402


def _cameras(*serials):
    return [CameraInfo(index=i, serial=s, model="MV-CA013-A0UM", interface="USB3")
            for i, s in enumerate(serials)]


# =============================================================================================
# The shipped config must not name one machine's hardware
# =============================================================================================
def test_no_shipped_template_pins_a_serial():
    """THE ACTUAL BUG, guarded at its source. A serial in a template is one bench's hardware
    imposed on every machine that installs the software -- and the symptom is the least helpful
    message the app can produce: "camera could not be opened"."""
    import yaml

    from flygym_tracker.config import DEFAULT_CONFIG_PATH, RIG_CONFIG_PATH

    for path in (DEFAULT_CONFIG_PATH, RIG_CONFIG_PATH):
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        serial = ((loaded.get("source") or {}).get("camera") or {}).get("serial")
        assert serial is None, "%s pins serial %r - it will fail on every other machine" % (
            path.name, serial)


# =============================================================================================
# The picker
# =============================================================================================
def test_it_offers_the_attached_cameras(qapp):
    picker = CameraPicker(lister=lambda: _cameras("AAA111", "BBB222"))
    picker.refresh()
    labels = [picker.combo.itemText(i) for i in range(picker.combo.count())]
    assert labels[0] == ANY_CAMERA
    assert any("AAA111" in text for text in labels)
    assert any("BBB222" in text for text in labels)


def test_any_camera_is_the_default_and_means_no_serial(qapp):
    """`serial: null` is the right answer on a single-camera bench, which is nearly every bench,
    and it is what a fresh install now does."""
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.refresh()
    assert picker.serial() is None
    assert picker.combo.currentText() == ANY_CAMERA


def test_choosing_a_camera_reports_its_serial(qapp):
    chosen = []
    picker = CameraPicker(lister=lambda: _cameras("AAA111", "BBB222"))
    picker.serial_chosen.connect(chosen.append)
    picker.refresh()
    picker.combo.setCurrentIndex(picker.combo.findData("BBB222"))
    picker._on_activated(picker.combo.currentIndex())
    assert chosen == ["BBB222"]
    assert picker.serial() == "BBB222"


def test_going_back_to_any_camera_reports_None(qapp):
    """Not the empty string: `None` is what `serial: null` means, and what the config writer and
    `_find_device` both read as "use whatever is attached"."""
    chosen = []
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.set_serial("AAA111")
    picker.refresh()
    picker.serial_chosen.connect(chosen.append)
    picker.combo.setCurrentIndex(0)
    picker._on_activated(0)
    assert chosen == [None]


def test_re_choosing_the_same_camera_says_nothing(qapp):
    """Otherwise every stray activation rewrites the config file."""
    chosen = []
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.set_serial("AAA111")
    picker.refresh()
    picker.serial_chosen.connect(chosen.append)
    picker._on_activated(picker.combo.findData("AAA111"))
    assert chosen == []


# =============================================================================================
# The situation that started this
# =============================================================================================
def test_a_pinned_camera_that_is_not_attached_is_shown_as_exactly_that(qapp):
    """THE WHOLE POINT. Silently falling back to "any camera" would hide the mismatch that is
    stopping the app opening the camera -- the operator would see a sane-looking dropdown and no
    hint that the config names hardware on another machine."""
    picker = CameraPicker(lister=lambda: _cameras("BBB222"))
    picker.set_serial("AAA111")
    picker.refresh()
    assert "AAA111" in picker.combo.currentText()
    assert "NOT ATTACHED" in picker.combo.currentText()
    assert "NOT attached" in picker.note.text()


def test_no_cameras_at_all_names_the_two_usual_causes(qapp):
    """A cable, or another program holding the device -- USB3 Vision allows one at a time, so an
    open MVS Viewer is enough to make the camera invisible."""
    picker = CameraPicker(lister=lambda: [])
    picker.refresh()
    text = picker.note.text()
    assert "no cameras detected" in text
    assert "MVS" in text and "cable" in text


def test_a_missing_sdk_is_reported_as_a_missing_sdk(qapp):
    """DIFFERENT PROBLEM, DIFFERENT FIX. "MVS is not installed" is a download; "no cameras found"
    is a cable. Reporting one as the other sends the operator on the wrong hunt."""
    def explode():
        raise RuntimeError("HikRobot MvImport SDK is not available (looked in ...)")

    picker = CameraPicker(lister=explode)
    picker.refresh()
    assert "MVS software is not installed" in picker.note.text()


def test_enumeration_failing_never_raises_into_the_window(qapp):
    def explode():
        raise OSError("something odd")

    picker = CameraPicker(lister=explode)
    picker.refresh()                    # must not raise
    assert picker.note.text()


# =============================================================================================
# In the window
# =============================================================================================
def test_the_window_writes_the_choice_to_the_machines_own_config(qapp, tmp_path):
    """NEVER TO THE SHIPPED TEMPLATE. A serial is a fact about one physical bench; writing it into
    the template is the bug this whole change exists to undo."""
    import yaml

    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    local = tmp_path / "rig.local.yaml"
    local.write_text("source:\n  camera:\n    frame_rate: 20.0\n", encoding="utf-8")
    state = gui_state.default_state()
    win = MainWindow(config=load_config(), config_path=str(local), state=state,
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda t: True)
    try:
        win._on_camera_serial_changed("ZZZ999")
        written = yaml.safe_load(local.read_text(encoding="utf-8"))
        assert written["source"]["camera"]["serial"] == "ZZZ999"
        assert written["source"]["camera"]["frame_rate"] == 20.0, "the write clobbered a neighbour"
    finally:
        win.run.shutdown()
        win.session.shutdown()


def test_the_new_camera_is_adopted_without_restarting_the_app(qapp, tmp_path):
    """A picker that wrote the file and then kept opening the OLD camera for the rest of the
    session would be a control that appears to work and does not -- on the one screen an operator
    reaches precisely because nothing is working."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    local = tmp_path / "rig.local.yaml"
    local.write_text("source:\n  camera:\n    frame_rate: 20.0\n", encoding="utf-8")
    original = lambda: None                                          # noqa: E731
    win = MainWindow(config=load_config(), config_path=str(local), state=gui_state.default_state(),
                     root=str(tmp_path), camera_factory=original, confirm=lambda t: True)
    try:
        assert win.session._worker._factory is original
        win._on_camera_serial_changed("ZZZ999")
        assert win.session._worker._factory is not original, "the session kept the old camera"
        assert "Open the camera" in win.settings_view._status
    finally:
        win.run.shutdown()
        win.session.shutdown()


def test_the_factory_is_not_swapped_under_a_live_camera(qapp, tmp_path):
    """`_factory` is read on the worker thread inside `open`, and swapping it under a live stream
    would leave the window describing a camera that is not the one delivering frames."""
    from flygym_tracker.gui import camera_session as cs
    from flygym_tracker.gui.camera_session import CameraSession

    session = CameraSession(lambda: None)
    try:
        session._state = cs.STREAMING
        assert session.set_factory(lambda: "new") is False
        session._state = cs.CLOSED
        assert session.set_factory(lambda: "new") is True
    finally:
        session.shutdown()
