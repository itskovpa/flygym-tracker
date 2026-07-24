"""A row of controls that WRAPS instead of forcing the window wider than the screen.

THE REGRESSION THIS EXISTS FOR, and it is the FOURTH of its kind on this project. The rig laptop is
a 2880x1800 panel at 200% scaling -- a 1440x900 desktop. Three times already something has been
laid out wider or taller than that and simply gone off the edge: the cv2 selector drew the lower
vial row 130 px below the screen bottom (unclickable), the cv2 settings panel was placed 66 px past
the right edge, and the Qt window came to 873 px on an 852 px work area.

A QHBoxLayout OF BUTTONS IS THE SAME BUG WAITING TO HAPPEN, because its minimum width is the SUM of
its children and it has no other option: the layout's only answer to "you have less room than that"
is to make the window bigger. Measured here: the vial-drawing strip (7 buttons) reported a 1082 px
minimum and the run band's tool strip 1532 px, which together forced the window's minimum width to
1646 px -- 246 px wider than the whole desktop. Nothing warns; the window just cannot be fully
seen, and the controls that fall off the edge are unreachable.

WHAT THIS DOES INSTEAD. It fills a row until the next item would not fit, then starts another row.
Its minimum width is therefore the WIDEST SINGLE ITEM, not the sum -- so a strip of controls can
never again be the reason a window does not fit a screen. It grows DOWNWARD, which the layout
above it can absorb, rather than sideways, which it cannot.

This is Qt's own documented flow-layout pattern (`heightForWidth` + manual `setGeometry`); it is
written out here rather than depended on because the example ships as C++ documentation, not as an
importable class.
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QMargins, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """Left-to-right, wrapping to a new line when the next item would not fit."""

    def __init__(self, parent: Optional[QWidget] = None, margin: int = 0,
                 spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: List = []
        self._spacing = int(spacing)
        self.setContentsMargins(QMargins(margin, margin, margin, margin))

    # -- QLayout plumbing --------------------------------------------------------------------
    def addItem(self, item) -> None:                    # noqa: N802 - Qt's name
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):                       # noqa: N802 - Qt's name
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):                       # noqa: N802 - Qt's name
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):                      # noqa: N802 - Qt's name
        """Nothing. A control strip should take the room it needs and leave the rest to the
        picture, which is the thing on this screen worth every pixel it can get."""
        return Qt.Orientation(0)

    # -- the wrapping ------------------------------------------------------------------------
    def hasHeightForWidth(self) -> bool:                # noqa: N802 - Qt's name
        return True

    def heightForWidth(self, width: int) -> int:        # noqa: N802 - Qt's name
        return self._lay_out(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:         # noqa: N802 - Qt's name
        super().setGeometry(rect)
        self._lay_out(rect, apply=True)

    def sizeHint(self) -> QSize:                        # noqa: N802 - Qt's name
        """One row, everything on it -- the size this WANTS when there is room for it."""
        margins = self.contentsMargins()
        width = margins.left() + margins.right()
        height = 0
        for index, item in enumerate(self._items):
            hint = item.sizeHint()
            width += hint.width() + (self._spacing if index else 0)
            height = max(height, hint.height())
        return QSize(width, height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:                     # noqa: N802 - Qt's name
        """THE WIDEST SINGLE ITEM, NOT THE SUM. This one line is the whole point of the file: it
        is what makes a control strip unable to force a window wider than the screen."""
        margins = self.contentsMargins()
        size = QSize(0, 0)
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return QSize(size.width() + margins.left() + margins.right(),
                     size.height() + margins.top() + margins.bottom())

    def _lay_out(self, rect: QRect, apply: bool) -> int:
        """Place the items (or just measure them) and return the total height used."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if line_height > 0 and next_x - self._spacing > area.right() + 1:
                x = area.x()
                y = y + line_height + self._spacing
                next_x = x + hint.width() + self._spacing
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


def flow_strip(parent: Optional[QWidget] = None, spacing: int = 6):
    """A widget carrying a `FlowLayout`, sized so it never fights the picture for room."""
    widget = QWidget(parent)
    layout = FlowLayout(widget, margin=0, spacing=spacing)
    widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return widget, layout


class FlowContainer(QWidget):
    """A widget whose height TRACKS its `FlowLayout`'s wrap, in ANY parent layout.

    THE BUG THIS CLOSES. `FlowLayout` reports `heightForWidth` correctly, and a `QVBoxLayout` honours
    it -- so the run band's tool strip wraps to a second line and grows. But a `QGridLayout` cell
    NEVER queries `heightForWidth`: it gives the cell the one-line `sizeHint` height and CLIPS the
    wrapped rows. The recording and fly-tracking rows live in grid cells, so their second line (the
    "size x" spin, the notes) vanished whenever the row wrapped -- exactly the "controls not visible
    at the edge" the operator reported, appearing only when the central column was narrow enough to
    force a wrap.

    THE FIX, FROM THE OTHER SIDE. Rather than hope the parent asks, this widget TELLS it: on every
    resize it sets its own `minimumHeight` to the flow's height at the new width, and a grid cell
    DOES respect a child's minimum height. So the row is given the room its wrapped lines need
    whatever layout holds it. Converges in one step because changing the height never changes the
    width that drove it.
    """

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._flow = FlowLayout(self, margin=0, spacing=spacing)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

    def flow(self) -> "FlowLayout":
        """The `FlowLayout` to add controls to."""
        return self._flow

    def resizeEvent(self, event) -> None:                # noqa: N802 - Qt's name
        super().resizeEvent(event)
        wanted = self._flow.heightForWidth(self.width())
        if wanted != self.minimumHeight():
            self.setMinimumHeight(wanted)
