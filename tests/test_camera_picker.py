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

from flygym_tracker.frame_source import KIND_UVC, CameraInfo  # noqa: E402
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
    picker.refresh(blocking=True)
    labels = [picker.combo.itemText(i) for i in range(picker.combo.count())]
    assert labels[0] == ANY_CAMERA
    assert any("AAA111" in text for text in labels)
    assert any("BBB222" in text for text in labels)


def test_any_camera_is_the_default_and_means_no_serial(qapp):
    """`serial: null` is the right answer on a single-camera bench, which is nearly every bench,
    and it is what a fresh install now does."""
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.refresh(blocking=True)
    assert picker.serial() is None
    assert picker.combo.currentText() == ANY_CAMERA


def test_choosing_a_camera_reports_its_serial(qapp):
    chosen = []
    picker = CameraPicker(lister=lambda: _cameras("AAA111", "BBB222"))
    picker.serial_chosen.connect(chosen.append)
    picker.refresh(blocking=True)
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
    picker.refresh(blocking=True)
    picker.serial_chosen.connect(chosen.append)
    picker.combo.setCurrentIndex(0)
    picker._on_activated(0)
    assert chosen == [None]


def test_re_choosing_the_same_camera_says_nothing(qapp):
    """Otherwise every stray activation rewrites the config file."""
    chosen = []
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.set_serial("AAA111")
    picker.refresh(blocking=True)
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
    picker.refresh(blocking=True)
    assert "AAA111" in picker.combo.currentText()
    assert "NOT ATTACHED" in picker.combo.currentText()
    assert "NOT attached" in picker.note.text()


def test_no_cameras_at_all_names_the_two_usual_causes(qapp):
    """A cable, or another program holding the device -- USB3 Vision allows one at a time, so an
    open MVS Viewer is enough to make the camera invisible."""
    picker = CameraPicker(lister=lambda: [])
    picker.refresh(blocking=True)
    text = picker.note.text()
    assert "no cameras detected" in text
    assert "MVS" in text and "cable" in text


def test_a_missing_sdk_is_reported_as_the_rig_camera_being_unavailable(qapp):
    """DIFFERENT PROBLEM, DIFFERENT FIX. "MVS is not installed" is a download; "no cameras found"
    is a cable. The line leads with RIG CAMERA NOT AVAILABLE so the operator knows which of the two
    listed cameras is the one that is missing."""
    def explode():
        raise RuntimeError("the HikRobot MVS software was not found on this computer.")

    picker = CameraPicker(lister=explode)
    picker.refresh(blocking=True)
    assert "RIG CAMERA NOT AVAILABLE" in picker.note.text()
    assert "MVS software was not found" in picker.note.text()


def test_a_webcam_being_found_does_not_hide_the_rig_camera_failure(qapp):
    """THE BUG REPORTED FROM A SECOND PC. The rig lookup failed, a webcam was found, and the error
    was swallowed entirely -- so the picker listed the laptop camera and said NOTHING about why the
    rig camera was absent. That is the least useful possible state for the one screen somebody
    opens precisely BECAUSE the camera will not work."""
    from flygym_tracker.gui.camera_picker import _PartialResult

    def partial():
        raise _PartialResult([_webcam()], RuntimeError("MVS was not found on this computer."))

    picker = CameraPicker(lister=partial)
    picker.refresh(blocking=True)
    labels = [picker.combo.itemText(i) for i in range(picker.combo.count())]
    assert any("webcam" in text for text in labels), "the webcam that WAS found got lost"
    assert "RIG CAMERA NOT AVAILABLE" in picker.note.text(),         "a webcam in the list hid the reason the rig camera is missing"


def test_the_full_diagnosis_is_kept_in_the_tooltip(qapp):
    """The message names every folder searched and what to set MVS_PYTHON_SDK to. That is several
    lines and this is one line of a settings row, so the detail lives in the tooltip rather than
    being truncated away."""
    def explode():
        raise RuntimeError("MVS was not found.\n\nLooked in:\n    C:/Program Files/MVS\n\n"
                           "Set MVS_PYTHON_SDK to its MvImport folder.")

    picker = CameraPicker(lister=explode)
    picker.refresh(blocking=True)
    assert "MVS_PYTHON_SDK" in picker.note.toolTip()


def test_enumeration_failing_never_raises_into_the_window(qapp):
    def explode():
        raise OSError("something odd")

    picker = CameraPicker(lister=explode)
    picker.refresh(blocking=True)                    # must not raise
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


# =============================================================================================
# Webcams are listed too -- and marked as what they are
# =============================================================================================
def _webcam(index=0, model="Integrated Camera"):
    return CameraInfo(index=index, serial=None, model=model, interface="UVC", kind=KIND_UVC)


def test_a_built_in_webcam_is_listed(qapp):
    """A picker that showed nothing on a laptop with a working built-in camera looks broken -- and
    the question being asked is usually "is ANY camera visible to this program?"."""
    picker = CameraPicker(lister=lambda: [_webcam()])
    picker.refresh(blocking=True)
    labels = [picker.combo.itemText(i) for i in range(picker.combo.count())]
    assert any("Integrated Camera" in text for text in labels)


def test_a_webcam_is_labelled_as_unfit_for_an_experiment(qapp):
    """NOT COSMETIC. A webcam auto-exposes, and on a drum turning past an IR backlight that
    re-levels the whole image between frames -- which is exactly the signal the activity
    measurement reads. It produces numbers that measure the camera's gain control."""
    picker = CameraPicker(lister=lambda: [_webcam()])
    picker.refresh(blocking=True)
    assert "NOT for experiments" in picker.combo.itemText(1)


def test_choosing_a_webcam_says_so_every_time_it_is_shown(qapp):
    picker = CameraPicker(lister=lambda: [_webcam()])
    picker.set_serial("uvc:0")
    picker.refresh(blocking=True)
    assert "NOT valid for an experiment" in picker.note.text()


def test_a_webcam_is_identified_by_index_because_it_has_no_serial(qapp):
    chosen = []
    picker = CameraPicker(lister=lambda: [_webcam(index=1)])
    picker.serial_chosen.connect(chosen.append)
    picker.refresh(blocking=True)
    picker._on_activated(picker.combo.findData("uvc:1"))
    assert chosen == ["uvc:1"]


def test_webcams_alone_are_reported_as_no_rig_camera(qapp):
    """The honest answer on a machine with a working laptop camera and no rig attached: cameras
    were found, and none of them can run an experiment."""
    picker = CameraPicker(lister=lambda: [_webcam()])
    picker.refresh(blocking=True)
    assert "no rig camera" in picker.note.text()


def test_a_chosen_webcam_builds_a_webcam_source():
    """The selection has to actually open something, or the picker is a control that does not
    control anything."""
    from flygym_tracker.cli import _camera_source_from_config
    from flygym_tracker.config import load_config
    from flygym_tracker.frame_source import UvcCameraSource

    config = load_config(overrides={"source": {"camera": {"serial": "uvc:2"}}})
    source = _camera_source_from_config(config)
    assert isinstance(source, UvcCameraSource)
    assert source.index == 2


def test_a_real_serial_still_builds_the_rig_camera():
    from flygym_tracker.cli import _camera_source_from_config
    from flygym_tracker.config import load_config
    from flygym_tracker.frame_source import HikCameraSource

    config = load_config(overrides={"source": {"camera": {"serial": "DA4282883"}}})
    source = _camera_source_from_config(config)
    assert isinstance(source, HikCameraSource)
    assert source.serial == "DA4282883"


def test_a_webcam_with_no_trustworthy_name_still_shows_its_index(qapp):
    """MEASURED, AND THE REASON NAMES ARE CONDITIONAL. Windows reported two cameras on the dev
    machine ('Integrated IR Camera', 'Integrated Camera') where probing found one, so pairing by
    position would have put a name on a device that may not own it. The index is what is opened,
    so the index is what must always be visible."""
    picker = CameraPicker(lister=lambda: [_webcam(index=1, model="")])
    picker.refresh(blocking=True)
    assert "webcam 1" in picker.combo.itemText(1)


def test_names_are_dropped_when_they_cannot_be_paired_with_the_probe():
    from flygym_tracker import frame_source as fs

    class FakeCv2:
        CAP_DSHOW = 700

        class VideoCapture:
            def __init__(self, index, backend=0):
                self._ok = index == 0        # one camera, at index 0

            def isOpened(self):
                return self._ok

            def release(self):
                pass

    import sys
    import types
    module = types.ModuleType("cv2")
    module.CAP_DSHOW = FakeCv2.CAP_DSHOW
    module.VideoCapture = FakeCv2.VideoCapture
    saved = sys.modules.get("cv2")
    sys.modules["cv2"] = module
    saved_names = fs._windows_camera_names
    fs._windows_camera_names = lambda: ["Camera A", "Camera B"]   # TWO names, ONE camera probed
    try:
        cameras = fs._list_uvc_cameras(max_index=3)
        assert len(cameras) == 1
        assert cameras[0].model == "", "a name was paired with a camera it may not belong to"
    finally:
        fs._windows_camera_names = saved_names
        if saved is not None:
            sys.modules["cv2"] = saved
        else:
            del sys.modules["cv2"]


def test_closing_the_window_while_it_is_still_looking_does_not_crash(qapp):
    """A REAL CRASH, caught in the suite and reproduced here. The first version emitted a Qt signal
    from the enumeration thread straight into this widget; if the window closed while a probe was
    still running -- and a probe takes over a second, so closing the app shortly after launch is
    enough -- the emit landed on a destroyed C++ object and took the whole process down, with no
    Python traceback, just a stack dump during teardown.

    The timer belongs to the widget and dies with it, so nothing on the GUI side is called again.
    """
    import threading
    import time

    release = threading.Event()

    def slow():
        release.wait(5.0)
        return [_webcam()]

    picker = CameraPicker(lister=slow)
    picker.refresh()                          # threaded; still probing
    qapp.processEvents()
    picker.deleteLater()
    del picker
    qapp.processEvents()
    release.set()                             # the worker finishes AFTER the widget is gone
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    # Reaching here at all is the assertion: the process is still alive.


def test_the_not_found_error_counts_what_it_lists_and_opens_nothing(qapp, monkeypatch):
    """TWO BUGS IN ONE LINE, both found by running it against the real camera.

    It read "Found 1 camera(s)" over a list of TWO, because the count came from the MVS device list
    while the list came from `list_cameras()`, which also enumerates webcams. And enumerating
    webcams OPENS them -- so reporting that a camera failed to open would have switched on the
    operator's webcam, indicator light and all, as a side effect.
    """
    from flygym_tracker import frame_source as fs

    probed = []
    monkeypatch.setattr(fs, "_list_uvc_cameras",
                        lambda *a, **k: probed.append(True) or [_webcam()])
    monkeypatch.setattr(fs, "_list_hikrobot_cameras",
                        lambda: [fs.CameraInfo(index=0, serial="AAA111", model="MV-CA013-A0UM")])

    source = fs.HikCameraSource(serial="NOPE")
    message = source._found(1)
    assert "Found 1 camera(s)" in message
    assert message.count("    ") == 1, "the count disagrees with the list:\n%s" % message
    assert not probed, "reporting a camera failure probed (and lit up) the webcam"


def test_the_status_line_speaks_rather_than_spelling_the_identifier(qapp, tmp_path):
    """`uvc:0` is how the choice is STORED; "webcam 0" is what it IS. An operator-facing line
    should never make somebody decode an internal identifier -- and this one carries a warning they
    have to actually read."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    local = tmp_path / "rig.local.yaml"
    local.write_text("source:\n  camera:\n    frame_rate: 20.0\n", encoding="utf-8")
    win = MainWindow(config=load_config(), config_path=str(local), state=gui_state.default_state(),
                     root=str(tmp_path), camera_factory=lambda: None, confirm=lambda t: True)
    try:
        win._on_camera_serial_changed("uvc:0")
        status = win.settings_view._status
        assert "uvc:0" not in status, "the raw identifier reached the operator: %s" % status
        assert "webcam 0" in status
        assert "not valid for an experiment" in status.lower()
    finally:
        win.run.shutdown()
        win.session.shutdown()



# =============================================================================================
# Finding the MVS SDK wherever it is installed
# =============================================================================================
def test_the_sdk_is_looked_for_in_more_than_one_place():
    r"""A REAL BUG FROM A SECOND PC. The code looked in exactly one hard-coded path --
    `C:\Program Files (x86)\MVS\...`, the development machine's layout -- while the INSTALLER
    shipped beside it already checked both Program Files trees. A PC with 64-bit MVS in
    `C:\Program Files\MVS\...` therefore found no rig camera at all."""
    from flygym_tracker.frame_source import mvs_sdk_candidates

    candidates = [c.lower() for c in mvs_sdk_candidates()]
    assert len(candidates) >= 2
    assert any("program files (x86)" in c for c in candidates)
    assert any(c.startswith(r"c:\program files\mvs") for c in candidates)


def test_an_explicit_override_is_searched_first(monkeypatch, tmp_path):
    """Somebody who sets MVS_PYTHON_SDK has said exactly where it is; nothing should outrank it."""
    from flygym_tracker.frame_source import mvs_sdk_candidates

    monkeypatch.setenv("MVS_PYTHON_SDK", str(tmp_path / "mine"))
    assert mvs_sdk_candidates()[0] == str(tmp_path / "mine")


def test_the_candidate_list_has_no_duplicates(monkeypatch):
    from flygym_tracker.frame_source import MVS_SDK_SEARCH_PATHS, mvs_sdk_candidates

    monkeypatch.setenv("MVS_PYTHON_SDK", MVS_SDK_SEARCH_PATHS[0])
    candidates = mvs_sdk_candidates()
    assert len(candidates) == len(set(candidates))


def test_a_missing_sdk_names_every_folder_it_tried(monkeypatch, tmp_path):
    """So the operator can see at a glance that their install is somewhere else -- and is told the
    environment variable that fixes it."""
    from flygym_tracker.frame_source import HikCameraSource

    monkeypatch.setenv("MVS_PYTHON_SDK", str(tmp_path / "nowhere"))
    monkeypatch.setattr("flygym_tracker.frame_source.MVS_SDK_SEARCH_PATHS",
                        (str(tmp_path / "also-nowhere"),))
    with pytest.raises(RuntimeError) as excinfo:
        HikCameraSource._import_sdk()
    message = str(excinfo.value)
    assert "was not found" in message
    assert "nowhere" in message and "also-nowhere" in message
    assert "MVS_PYTHON_SDK" in message


def test_the_startup_scan_offers_every_camera_not_just_the_rig_one(qapp, tmp_path):
    """THE RIG OWNER'S CALL: "let the user choose a camera from all the enumerated cameras". A
    picker offering a choice from a list missing half the machine's cameras is not offering a
    choice -- and the operator who reaches this screen is usually there because a camera is
    already missing."""
    from flygym_tracker.config import load_config
    from flygym_tracker.gui import gui_state
    from flygym_tracker.gui.main_window import MainWindow

    asked = {}
    win = MainWindow(config=load_config(), config_path=str(tmp_path / "c.yaml"),
                     state=gui_state.default_state(), root=str(tmp_path),
                     camera_factory=lambda: None, confirm=lambda t: True)
    try:
        win.session_bar.camera_picker.refresh = (
            lambda blocking=False, include_uvc=True: asked.update(include_uvc=include_uvc))
        win.take_initial_focus()
        assert asked.get("include_uvc") is True, "startup did not look for every camera"
    finally:
        win.run.shutdown()
        win.session.shutdown()


def test_a_test_run_never_opens_a_real_camera(qapp, monkeypatch):
    """FLYGYM_NO_CAMERA_SCAN, set by conftest. Discovering a webcam means OPENING it, and several
    hundred window constructions each probing real devices would be slow, dependent on whatever is
    plugged in, and would flick the machine's camera light on throughout the run."""
    import os

    monkeypatch.setenv("FLYGYM_NO_CAMERA_SCAN", "1")
    touched = []
    picker = CameraPicker()                      # no injected lister -> would enumerate for real
    monkeypatch.setattr("flygym_tracker.frame_source.list_cameras_with_error",
                        lambda **k: touched.append(True) or ([], None))
    picker.refresh(blocking=True)
    assert not touched, "a test run reached the real camera enumeration"
    assert "switched off" in picker.note.text()


def test_the_guard_does_not_disable_an_injected_fake(qapp, monkeypatch):
    """Scoped to REAL enumeration only, which is what lets the suite forbid device access globally
    while every behaviour of this widget stays tested."""
    monkeypatch.setenv("FLYGYM_NO_CAMERA_SCAN", "1")
    picker = CameraPicker(lister=lambda: _cameras("AAA111"))
    picker.refresh(blocking=True)
    assert any("AAA111" in picker.combo.itemText(i) for i in range(picker.combo.count()))


# =============================================================================================
# What a FROZEN build needs, which no ordinary test run exercises
# =============================================================================================
def test_the_spec_bundles_every_module_the_MVS_SDK_IMPORTS():
    """THE BUG THAT MADE THE RIG CAMERA IMPOSSIBLE TO USE IN ANY INSTALLED BUILD.

    The MVS SDK is loaded at RUNTIME from the operator's MVS installation, so PyInstaller never
    analyses it and never bundles what it imports. `MvCameraControl_class.py` opens with
    `import platform`; nothing in this application imports `platform` itself; so the module was
    absent from the build and the SDK import died with `ModuleNotFoundError: No module named
    'platform'` on EVERY machine.

    The symptom was perfect camouflage -- the app listed the built-in webcam (OpenCV is bundled)
    and no rig camera, which reads as a hardware or driver problem.

    This reads the SDK's own source where it is installed and asserts the spec covers it, so a new
    HikRobot SDK that imports something new fails here rather than in a customer's lab.
    """
    import os
    import re

    from flygym_tracker.frame_source import mvs_sdk_candidates

    sdk_dir = next((p for p in mvs_sdk_candidates()
                    if os.path.isfile(os.path.join(p, "MvCameraControl_class.py"))), None)
    if sdk_dir is None:
        pytest.skip("the MVS SDK is not installed on this machine")

    imported = set()
    for name in os.listdir(sdk_dir):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(sdk_dir, name), encoding="utf-8", errors="replace") as f:
            for line in f:
                match = re.match(r"\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)", line)
                if match:
                    imported.add(match.group(1).split(".")[0])

    # The SDK's own sibling modules come with it on sys.path; only third-party/stdlib matter.
    siblings = {n[:-3] for n in os.listdir(sdk_dir) if n.endswith(".py")}
    needed = {m for m in imported if m not in siblings}

    spec = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "packaging", "flygym_tracker.spec")
    with open(spec, encoding="utf-8") as f:
        spec_text = f.read()
    missing = [m for m in sorted(needed) if ('"%s"' % m) not in spec_text]
    assert not missing, (
        "the MVS SDK imports %s, which the PyInstaller spec does not bundle. An installed build "
        "will fail to load the rig camera on every machine." % missing)


def test_the_native_runtime_is_loaded_by_absolute_path_for_this_architecture():
    """The SDK does `WinDLL("MvCameraControl.dll")` -- a bare name, resolved off the PATH. That
    survives running from a checkout and does NOT survive being frozen.

    It is also a latent hazard even unfrozen: a normal MVS install puts BOTH the 32- and 64-bit
    runtime folders on the PATH, with the 32-bit one FIRST, so a 64-bit process is relying on the
    loader to skip a wrong-architecture match. Loading by full path removes both problems, and
    Windows then serves the SDK's later bare-name call from the already-loaded module.
    """
    import sys as _sys

    from flygym_tracker.frame_source import MVS_RUNTIME_DIRS

    bits = 64 if _sys.maxsize > 2 ** 32 else 32
    assert bits in MVS_RUNTIME_DIRS
    for directory in MVS_RUNTIME_DIRS[64]:
        assert "Win64" in directory
    for directory in MVS_RUNTIME_DIRS[32]:
        assert "Win32" in directory


def test_preloading_the_runtime_never_raises(monkeypatch):
    """It runs before every SDK import. A failure here must cost the preload, not the import --
    the SDK's own error is more informative than anything this could invent."""
    from flygym_tracker import frame_source as fs

    monkeypatch.setattr(fs, "MVS_RUNTIME_DIRS", {64: (r"C:\nope",), 32: (r"C:\nope",)})
    assert fs._preload_mvs_runtime() is None
