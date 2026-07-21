"""Which physical camera to use -- chosen from the ones actually attached, not typed from memory.

THE FAILURE THIS EXISTS TO END. The shipped config carried `serial: "DA4282883"`, the serial of the
development rig's own camera. On any other machine the app searched for a camera that was not
there, failed, and said "camera could not be opened" -- while the machine's own camera worked
perfectly in HikRobot's MVS Viewer two minutes earlier. There was no way to see what WAS attached
and no way to choose it, so the only route out was to know that a YAML file somewhere pinned a
serial belonging to a camera on a different continent.

Three things fix that, and this file is the third:

  1. the shipped templates no longer pin a serial (that was the actual bug);
  2. the "not found" error now lists every camera it DID find;
  3. this picker: enumerate, show them, let the operator choose, write it down.

IT NEVER OPENS A CAMERA. Enumeration is a separate SDK call that takes no handle -- which matters,
because USB3 Vision access is exclusive and the moment somebody most wants to ask "what cameras are
there?" is while one is already streaming. Listing must never be able to interrupt an experiment.

THE CHOICE IS WRITTEN TO THE MACHINE'S OWN CONFIG LAYER, never to the shipped template. A serial is
a fact about one physical bench; putting it in the template is precisely the bug above.
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

#: What `serial: null` means, in the operator's words rather than YAML's.
ANY_CAMERA = "use whatever camera is attached"


class _PartialResult(Exception):
    """Cameras were found AND the rig-camera lookup failed. Carries both, so neither is lost."""

    def __init__(self, cameras, error) -> None:
        super().__init__(str(error))
        self.cameras = cameras
        self.error = error


class CameraPicker(QWidget):
    """A dropdown of the attached cameras, a Refresh, and a line saying what is pinned."""

    #: The operator chose a camera. `None` means "no serial -- use whatever is attached".
    serial_chosen = Signal(object)
    #: How often the GUI thread checks whether the enumeration worker has finished. See `refresh`
    #: for why this is a poll rather than a signal from the worker.
    POLL_MS = 120

    def __init__(self, lister=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        #: Injected so this is testable with no SDK, no camera and no MVS install -- which is every
        #: machine the tests run on, and most machines the app is developed on.
        self._lister = lister
        self._include_uvc = True
        self._cameras: List = []
        self._serial: Optional[str] = None
        self._loading = False

        line = QHBoxLayout(self)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)

        self.combo = QComboBox()
        self.combo.setMinimumWidth(240)
        self.combo.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.combo.setToolTip(
            "Which physical camera to use.\n\n"
            "\"%s\" is the right answer on a bench with one camera, and is what a fresh install "
            "does. Pin a specific serial only where more than one camera is attached -- pinning "
            "one means the app will refuse to run on any other camera, which is the point of "
            "pinning it." % ANY_CAMERA)
        self.combo.activated.connect(self._on_activated)
        line.addWidget(self.combo, 1)

        self.refresh_button = QToolButton()
        self.refresh_button.setText("Refresh")
        self.refresh_button.setToolTip(
            "Look again for cameras, including ordinary webcams.\n\n"
            "Finding the rig camera opens nothing and is safe during a run. Finding WEBCAMS means "
            "briefly opening them, which lights their indicator -- which is why it happens when "
            "you press this, and not on its own.")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        line.addWidget(self.refresh_button)

        self.note = QLabel("")
        self.note.setProperty("role", "note")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        line.addWidget(self.note, 1)

        # OWNED BY THIS WIDGET, so it stops the moment the widget is destroyed. That ownership is
        # the whole mechanism -- see `refresh`.
        self._poll = QTimer(self)
        self._poll.setInterval(self.POLL_MS)
        self._poll.timeout.connect(self._check)
        #: Written by the worker THREAD, read by the timer on the GUI thread. A plain attribute,
        #: assigned atomically under the GIL; no lock, because there is exactly one writer and the
        #: value is replaced whole.
        self._result = None
        self._rebuild([])

    # -- state ---------------------------------------------------------------------------------
    def set_serial(self, serial: Optional[str]) -> None:
        """Show what the config currently pins, without re-emitting it as a fresh choice."""
        self._serial = (str(serial).strip() or None) if serial else None
        self._rebuild(self._cameras)

    def serial(self) -> Optional[str]:
        return self._serial

    def refresh(self, blocking: bool = False, include_uvc: bool = True) -> None:
        """Re-enumerate, OFF THE GUI THREAD. A failure is reported, never raised.

        THREADED BECAUSE WEBCAMS CANNOT BE ENUMERATED WITHOUT OPENING THEM. GenICam enumeration is
        a cheap handle-free call, but UVC has no equivalent: the only way to learn that webcam 1
        exists is to open it, and each attempt costs a few hundred milliseconds. Probing four
        indices on the GUI thread would freeze the window for a second or more at launch, which on
        a scientific instrument reads as a crash.

        EVERY CAMERA, INCLUDING WEBCAMS, BY DEFAULT -- at startup as well. This was rig-only for a
        while, on the reasoning that discovering a webcam means briefly opening it and lighting its
        indicator. The rig owner overruled that, and they are right: a picker that offers a choice
        from a list missing half the machine's cameras is not offering a choice, and the operator
        who reaches this screen is usually there because a camera is already missing. The brief
        open is disclosed on the Refresh button instead of being avoided.

        `include_uvc=False` still looks only for the rig camera, which is handle-free and safe to
        call during a run.

        `FLYGYM_NO_CAMERA_SCAN=1` switches enumeration off entirely; the test suite sets it.
        `blocking=True` is for tests, which want the answer without an event loop.
        """
        if self._lister is None and os.environ.get("FLYGYM_NO_CAMERA_SCAN"):
            # SET BY THE TEST SUITE, and it guards REAL enumeration only -- a test that injected
            # its own `lister` is asking for that fake to run, and is touching no hardware. Scoping
            # the guard this way is what lets the suite forbid device access globally while the
            # picker's own behaviour stays fully tested.
            self._say("camera scanning is switched off (FLYGYM_NO_CAMERA_SCAN)")
            return
        if self._loading:
            return
        self._loading = True
        self._include_uvc = include_uvc
        self._say("looking for cameras...")
        if blocking:
            self._finish(*self._enumerate())
            return
        self._include_uvc = include_uvc
        self._result = None
        self._poll.start()
        thread = threading.Thread(target=self._work, name="camera-enumerate", daemon=True)
        thread.start()

    def _on_refresh_clicked(self, _checked: bool = False) -> None:
        """The operator asked. THIS is what may open a webcam -- see `refresh`."""
        self.refresh(include_uvc=True)

    def _work(self) -> None:
        """On the worker thread. TOUCHES NOTHING BUT A PYTHON ATTRIBUTE.

        THE CRASH THIS SHAPE AVOIDS, and it was a real one caught in the test suite. The first
        version emitted a Qt signal from here straight into this widget. If the window closed while
        a probe was still running -- and a probe takes over a second, so closing the app shortly
        after launch is enough -- the emit landed on a C++ object that had already been destroyed
        and took the process down. No Python traceback, just a stack dump during teardown.

        A poll inverts the ownership: the timer belongs to the widget and dies with it, so once the
        window is gone nothing on the GUI side is ever called again. All this thread does is assign
        a tuple to an attribute, which is safe whatever happened to the C++ object.
        """
        self._result = self._enumerate()

    def _enumerate(self):
        try:
            return list(self._list()), ""
        except _PartialResult as partial:
            # Cameras WERE found and the rig lookup still failed. Both facts are kept.
            return list(partial.cameras), str(partial.error)
        except Exception as exc:
            return [], str(exc)

    def _check(self) -> None:
        """On the GUI thread: has the worker finished? Started and stopped by `refresh`."""
        result = self._result
        if result is None:
            return
        self._poll.stop()
        self._result = None
        self._finish(*result)

    def _finish(self, cameras, error: str) -> None:
        self._loading = False
        self._cameras = list(cameras)
        self._rebuild(self._cameras)
        if not error:
            return
        # SHOWN EVEN WHEN CAMERAS WERE FOUND. A webcam in the list does not answer "why is the rig
        # camera missing", and that is the only question being asked on this screen.
        first = error.strip().splitlines()[0]
        self._say("RIG CAMERA NOT AVAILABLE: %s" % first[:160])
        # The full text -- which names every folder searched, and what to set MVS_PYTHON_SDK to --
        # goes in the tooltip, because it is several lines and this is one line of a settings row.
        self.note.setToolTip(error)

    def _list(self):
        if self._lister is not None:
            return self._lister()
        from flygym_tracker.frame_source import list_cameras

        # WEBCAMS ONLY WHEN ASKED. A picker that showed nothing on a laptop with a working built-in
        # camera looks broken -- but finding them means OPENING them, which lights the camera up.
        # So Refresh includes them and startup does not. See `refresh`.
        from flygym_tracker.frame_source import list_cameras_with_error  # noqa: F811

        cameras, error = list_cameras_with_error(include_uvc=self._include_uvc)
        # THE ERROR TRAVELS WITH THE RESULT. Finding a webcam is not a reason to stop saying that
        # the rig camera could not be looked for -- that combination is precisely what left a
        # second PC showing its laptop camera and no explanation.
        if error is not None:
            raise _PartialResult(cameras, error)
        return cameras

    # -- the dropdown --------------------------------------------------------------------------
    def _rebuild(self, cameras) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(ANY_CAMERA, None)
        for camera in cameras:
            # `config_id`, NOT `serial`: a webcam has no serial, and its identity is its
            # index. One field carries both kinds, which is what lets the config, the
            # error messages and this dropdown all route the choice the same way.
            self.combo.addItem(camera.label, camera.config_id)

        pinned = self._serial
        if pinned:
            index = self.combo.findData(pinned)
            if index < 0:
                # PINNED BUT NOT PRESENT -- the exact situation that started all this. It is shown
                # as its own entry rather than silently falling back to "any camera", because
                # quietly selecting something else would hide the very mismatch that is stopping
                # the app from opening the camera.
                self.combo.addItem("%s  -  PINNED BUT NOT ATTACHED" % pinned, pinned)
                index = self.combo.count() - 1
            self.combo.setCurrentIndex(index)
        else:
            self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self._refresh_note(cameras)

    def _refresh_note(self, cameras) -> None:
        if not cameras:
            self._say("no cameras detected - check the USB cable, and close the MVS Viewer if it "
                      "is open (only one program can hold the camera)")
            return
        pinned = self._serial
        if pinned and not any(c.config_id == pinned for c in cameras):
            self._say("the pinned camera is NOT attached - the app cannot open a camera until this "
                      "is changed")
            return
        chosen = next((c for c in cameras if c.config_id == pinned), None) if pinned else None
        if chosen is not None and not chosen.suitable:
            # SAID EVERY TIME A WEBCAM IS SELECTED, not once when it is picked. A webcam
            # auto-exposes, and on a drum turning past an IR backlight that re-levels the whole
            # image between frames -- which is exactly the signal the activity measurement reads.
            # It will produce numbers. They measure the camera's gain control, not the flies.
            self._say("a webcam - fine for trying the software out, NOT valid for an experiment "
                      "(auto-exposure alone would be measured as activity)")
            return
        rig = [c for c in cameras if c.suitable]
        found = "%d camera%s found" % (len(cameras), "" if len(cameras) == 1 else "s")
        if not rig:
            self._say("%s, but no rig camera among them - only the HikRobot camera can run an "
                      "experiment" % found)
            return
        self._say(found if pinned else "%s - using the first rig camera" % found)

    def _say(self, text: str) -> None:
        self.note.setText(text)
        self.note.setToolTip(text)

    def _on_activated(self, index: int) -> None:
        serial = self.combo.itemData(index)
        serial = str(serial) if serial else None
        if serial == self._serial:
            return
        self._serial = serial
        self._refresh_note(self._cameras)
        self.serial_chosen.emit(serial)
