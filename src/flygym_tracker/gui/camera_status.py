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
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget

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
        # A FLOOR, NOT A CEILING. This top band carries real buttons and a 20 px status dot; a hard
        # setFixedHeight clips them at the very top edge of the window under Windows "make text
        # bigger" (>100% UI font, independent of display scaling). A minimum keeps the compact look
        # while letting the row grow the few pixels a larger font needs.
        self.setMinimumHeight(44)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        # THE WAY BACK TO A CLOSED SETTINGS DOCK. A dock that can be closed needs a visible switch
        # to reopen it, or closing it once removes the settings from the app for good as far as the
        # operator can tell. It is CHECKABLE and driven by the dock's own `toggleViewAction`, so the
        # button and the dock cannot disagree about whether the settings are showing.
        self.settings_button = QPushButton("Settings")
        self.settings_button.setCheckable(True)
        self.settings_button.setProperty("role", "ghost")
        self.settings_button.setToolTip(
            "Show or hide the settings. Drag its title bar out to float it as its own window - "
            "on a second monitor, say - and drop it back on an edge to re-dock it.")
        layout.addWidget(self.settings_button)

        self.dot = QLabel("*")
        self.dot.setFixedWidth(14)
        layout.addWidget(self.dot)

        self.sentence = QLabel("")
        self.sentence.setWordWrap(False)
        # A STATUS LINE MUST NOT DECIDE HOW WIDE THE WINDOW IS. With the default policy this
        # label's sizeHint is the full width of its longest sentence ("Camera DA4282883 is yours -
        # 1280x1024 - streaming, 20.0 fps delivered (measured)"), and measured that alone demanded
        # 1186 px of the window's minimum. It stayed under the 1400 px limit only while it was the
        # widest thing in the window; the moment the settings moved into a dock beside it, the
        # window's minimum became 1752 px on a 1440 px desktop. Same rule the run band's state
        # label already follows, and the same rule as `flow_layout` -- fifth occurrence.
        self.sentence.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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
