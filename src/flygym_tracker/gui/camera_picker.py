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

from typing import List, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

#: What `serial: null` means, in the operator's words rather than YAML's.
ANY_CAMERA = "use whatever camera is attached"


class CameraPicker(QWidget):
    """A dropdown of the attached cameras, a Refresh, and a line saying what is pinned."""

    #: The operator chose a camera. `None` means "no serial -- use whatever is attached".
    serial_chosen = Signal(object)

    def __init__(self, lister=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        #: Injected so this is testable with no SDK, no camera and no MVS install -- which is every
        #: machine the tests run on, and most machines the app is developed on.
        self._lister = lister
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
        self.refresh_button.setToolTip("Look again for attached cameras. Does not open any of "
                                       "them, so this is safe during a run.")
        self.refresh_button.clicked.connect(self.refresh)
        line.addWidget(self.refresh_button)

        self.note = QLabel("")
        self.note.setProperty("role", "note")
        self.note.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        line.addWidget(self.note, 1)

        self._rebuild([])

    # -- state ---------------------------------------------------------------------------------
    def set_serial(self, serial: Optional[str]) -> None:
        """Show what the config currently pins, without re-emitting it as a fresh choice."""
        self._serial = (str(serial).strip() or None) if serial else None
        self._rebuild(self._cameras)

    def serial(self) -> Optional[str]:
        return self._serial

    def refresh(self) -> None:
        """Re-enumerate. A failure is REPORTED, never raised: this is a convenience on a window
        that must still open on a machine with no camera software installed at all."""
        if self._loading:
            return
        self._loading = True
        try:
            self._cameras = list(self._list())
            self._rebuild(self._cameras)
        except Exception as exc:
            self._cameras = []
            self._rebuild([])
            # THE TWO FAILURES ARE DIFFERENT AND HAVE DIFFERENT FIXES: "no MVS installed" is a
            # download, "no cameras found" is a cable. Saying which one it is saves the wrong hunt.
            text = str(exc)
            if "MvImport" in text or "MVS" in text:
                self._say("the HikRobot MVS software is not installed - the camera needs it")
            else:
                self._say("could not look for cameras: %s" % text.splitlines()[0][:120])
        finally:
            self._loading = False

    def _list(self):
        if self._lister is not None:
            return self._lister()
        from flygym_tracker.frame_source import list_cameras

        return list_cameras()

    # -- the dropdown --------------------------------------------------------------------------
    def _rebuild(self, cameras) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(ANY_CAMERA, None)
        for camera in cameras:
            self.combo.addItem(camera.label, camera.serial)

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
        if pinned and not any(c.serial == pinned for c in cameras):
            self._say("the pinned camera is NOT attached - the app cannot open a camera until this "
                      "is changed")
            return
        found = "%d camera%s found" % (len(cameras), "" if len(cameras) == 1 else "s")
        self._say(found if pinned else "%s - using the first one" % found)

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
