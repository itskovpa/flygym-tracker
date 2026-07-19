"""The one-line answer to "is the camera mine right now", always on screen.

WHY IT IS A FIXED BAND AND NOT A QStatusBar. A status bar reads as chrome and is ignored; this is
the single most consequential fact in the window. Every camera failure on this rig arrives as the
same symptom -- the SDK's ``0x80000203``, which names no culprit -- and the usual culprit is a
headless Bonsai with no window and no taskbar entry, so the rig LOOKS idle while the camera is
held. The bar exists so an operator never has to deduce that.

FOUR STATES, EACH ONE SENTENCE, EACH DISTINGUISHABLE AT A GLANCE:

    grey    Camera not open - nothing is being sent            [Open camera] [Free the camera...]
    amber   Opening camera DA4282883...                        (both disabled)
    green   Camera DA4282883 is yours - streaming, 88.5 fps delivered (measured)   [Close camera]
    red     Camera is busy - held by: a Bonsai workflow (PID 1234)   [Show what's holding it]

[Free the camera...] IS DISABLED UNLESS THE STATE IS CLOSED, and that is a safety rule rather than
a nicety: with the camera open, the process holding it is THIS one, and an app that offers to stop
whatever holds the camera while it is the holder is an app offering to kill itself.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from flygym_tracker.gui import theme
from flygym_tracker.gui.camera_session import (CLOSED, CLOSING, ERROR_BUSY, ERROR_OTHER, OPENING,
                                               STREAMING)


class CameraStatusBar(QWidget):
    """Coloured dot, one sentence, and the actions that are legal in this state."""

    open_requested = Signal()
    close_requested = Signal()
    free_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(44)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        self.dot = QLabel("*")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)

        self.sentence = QLabel("")
        self.sentence.setWordWrap(False)
        layout.addWidget(self.sentence, 1)

        self.open_button = QPushButton("Open camera")
        self.open_button.clicked.connect(self.open_requested)
        layout.addWidget(self.open_button)

        self.close_button = QPushButton("Close camera")
        self.close_button.clicked.connect(self.close_requested)
        layout.addWidget(self.close_button)

        self.free_button = QPushButton("Free the camera...")
        self.free_button.clicked.connect(self.free_requested)
        layout.addWidget(self.free_button)

        self.set_state(CLOSED, "")

    def set_state(self, state: str, detail: str = "", *, measured_fps: float = 0.0) -> None:
        """Render one state. `measured_fps` is shown ONLY when it was actually counted.

        A zero means no frames have been timed yet, and the sentence then says "starting" rather
        than "0.0 fps" -- a rate of zero next to the word "measured" is a claim about the sensor
        that nobody made.
        """
        self.dot.setStyleSheet("color: %s; font-size: 20px;"
                               % theme.STATE_COLORS.get(state, theme.DIM))
        self.sentence.setText(self._sentence(state, detail, measured_fps))
        self.open_button.setEnabled(state == CLOSED)
        self.close_button.setEnabled(state == STREAMING)
        # See the module docstring: never offer to stop the holder while we are the holder.
        self.free_button.setEnabled(state == CLOSED)
        self.free_button.setVisible(state in (CLOSED, ERROR_BUSY, ERROR_OTHER))
        if state in (ERROR_BUSY, ERROR_OTHER):
            self.free_button.setText("Show what's holding it")
            self.free_button.setEnabled(True)
        else:
            self.free_button.setText("Free the camera...")

    def _sentence(self, state: str, detail: str, measured_fps: float) -> str:
        if state == CLOSED:
            return "Camera not open - nothing is being sent"
        if state == OPENING:
            return "Opening %s..." % (detail or "the camera")
        if state == CLOSING:
            return "Closing the camera..."
        if state == STREAMING:
            base = "Camera %s" % (detail or "is yours")
            if measured_fps > 0:
                # "delivered (measured)" -- counted from frames that arrived, deliberately worded
                # apart from the frame-rate SETTING and from the camera's ResultingFrameRate
                # read-back, because this camera's rate limiter is documented to disengage
                # mid-stream while both of those still read back correct.
                return "%s - streaming, %.1f fps delivered (measured)" % (base, measured_fps)
            return "%s - streaming, waiting for the first frames to time" % base
        if state == ERROR_BUSY:
            return "Camera is busy - something else has it open"
        return "Camera could not be opened - %s" % (detail or "reason unknown")
