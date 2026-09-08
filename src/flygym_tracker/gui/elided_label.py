"""A one-line label that ELIDES instead of either widening the window or disappearing.

THE TWO FAILURES THIS REPLACES, both measured on the rig laptop (1440 x 852 of work area):

  * A plain `QLabel` has a minimum width of its whole sentence. Notes in this window grow while a
    run is going ("Run in progress - camera and algorithm settings are live - ...", the camera
    scanner's advice, the measurement's header), and each one that grew pushed the window's
    minimum past the screen. The window then could not be shrunk to fit and its bottom row sat
    under the taskbar.
  * The fix applied to that, one label at a time, was `QSizePolicy.Ignored` for the width. It
    stops the widening -- and lets the layout squeeze the label to NOTHING when the row is tight.
    Rendered at 1440 px the run's status read "Run in progress - camera and algorith"; at 1280 px
    the same label was 6 px wide. A note that is cut mid-word looks like a bug in the sentence,
    and a note that is 6 px wide looks like there is no note.

This label reports a small minimum (a dozen characters), asks for its full sentence when there is
room, and when there is not it draws "..." at the cut and puts the whole sentence in the tooltip.
`text()` still returns the full sentence, so nothing that reads a note has to know it is elided.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    """A single-line `QLabel` that elides to its width; the full text lives in `text()` and the tip."""

    #: The width the label insists on, in average characters. Enough to read that something is
    #: there and hover for the rest; small enough never to push a window past its screen.
    MIN_CHARS = 12

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__("", parent)
        self._full = ""
        self._own_tip: Optional[str] = None
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setText(text)

    # -- the text -------------------------------------------------------------------------------
    def setText(self, text: str) -> None:            # noqa: N802 - Qt name
        self._full = "" if text is None else str(text)
        self._sync()

    def text(self) -> str:
        return self._full

    def setToolTip(self, tip: str) -> None:          # noqa: N802 - Qt name
        """A tooltip set by the owner wins over the automatic full-text one."""
        self._own_tip = tip or None
        self._sync()

    # -- sizing ---------------------------------------------------------------------------------
    def minimumSizeHint(self) -> QSize:              # noqa: N802 - Qt name
        fm = self.fontMetrics()
        return QSize(fm.averageCharWidth() * self.MIN_CHARS, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:                     # noqa: N802 - Qt name
        fm = self.fontMetrics()
        margins = self.contentsMargins()
        width = fm.horizontalAdvance(self._full) + margins.left() + margins.right() + 4
        return QSize(width, super().sizeHint().height())

    def resizeEvent(self, event) -> None:            # noqa: N802 - Qt name
        super().resizeEvent(event)
        self._sync()

    def _sync(self) -> None:
        margins = self.contentsMargins()
        room = max(0, self.width() - margins.left() - margins.right() - 2)
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, room)
        super().setText(shown)
        if self._own_tip is not None:
            super().setToolTip(self._own_tip)
        else:
            super().setToolTip(self._full if shown != self._full else "")

    @property
    def is_elided(self) -> bool:
        """Whether the sentence on screen is shorter than the sentence held."""
        return super().text() != self._full
