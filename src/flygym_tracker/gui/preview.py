"""The live picture, and a caption that does not overstate what it is.

WHY THE PREVIEW SITS NEXT TO THE SETTINGS AND NOT ON ITS OWN TAB. Exposure and gain are tuned by
looking at the image. A design that makes you switch views to see the effect of the knob you are
turning is the cv2 panel's problem in a new toolkit.

WHY THE CAPTION NAMES THREE NUMBERS. The sensor runs at up to 88 fps and the eye needs about 15, so
the worker reads every frame (the SDK buffer fills otherwise) and offers roughly one in six to the
preview. A preview that silently showed 15 fps of an 88 fps stream would let someone tune exposure
against motion blur they cannot see -- so it says "preview 15 fps - camera delivering 88.5 fps
(measured) - 73 of 88 frames not shown". Same rule as everywhere else in this program: never
present a number as more than it is.

"DELIVERED" IS THE LOAD-BEARING WORD in that caption. It is counted from frames that actually
arrived, and it is NOT the AcquisitionFrameRate setting and NOT the camera's ResultingFrameRate
read-back. On this rig's camera the frame-rate limiter is documented to disengage mid-stream while
its registers still read back correct -- i.e. the two numbers the camera reports can both be right
and both be wrong about what is happening. The counted one cannot.

THE ndarray IS KEPT ALIVE ON PURPOSE. `QImage(arr.data, ...)` BORROWS the buffer; it does not copy
it. Dropping the array while the QImage is still on screen leaves the image pointing at freed
memory the moment the garbage collector runs. `HikCameraSource.read()` already copies out of the
SDK-owned buffer, so the array is ours to hold; `_image` and `_array` are therefore replaced
together and never separately.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

#: How often the GUI pulls the latest frame. Faster than the worker offers them, so the box is
#: usually empty and the pull is a cheap lock-and-return; the worker's throttle sets the real rate.
PULL_INTERVAL_MS = 50

#: Qt key -> the key NAMES `live_vial_selector.handle_key` already understands. That module's keymap
#: is the one an operator has learnt at this rig, and it is tested; this table exists so the Qt
#: surface drives THAT keymap rather than growing a second one that can drift out of step with it.
_QT_KEY_NAMES = {
    Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter",
    Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Delete: "backspace",
    Qt.Key.Key_Space: "space", Qt.Key.Key_Escape: "esc",
}


def key_name(event) -> Optional[str]:
    """The `handle_key` name for a `QKeyEvent`, or None for a key that means nothing here.

    Letters come through as themselves, lower-cased, so ``u`` and ``c`` work with or without Shift
    and with CapsLock on -- an operator clicking 32 polygons at 2 am should not lose a vial to a
    stuck CapsLock.
    """
    named = _QT_KEY_NAMES.get(event.key())
    if named is not None:
        return named
    text = (event.text() or "").strip().lower()
    return text if len(text) == 1 and text.isprintable() else None


def fit_rect(src_w: int, src_h: int, dst_w: int, dst_h: int) -> Tuple[int, int, int, int]:
    """`src` scaled to fit inside `dst` with aspect preserved and centred: ``(x, y, w, h)``.

    A PURE FUNCTION WITH ITS OWN TESTS, rather than four lines buried in `paintEvent`, because
    Stage 2 needs exactly this transform inverted: drawing vial polygons on the preview means
    turning a click at a widget coordinate into an image coordinate, and hit-testing a drawn ROI
    means the same in reverse. Having it extracted and numerically tested is the difference between
    extending this widget and rewriting it.

    Degenerate inputs return an empty rect rather than raising: a camera that has not delivered a
    frame yet reports a size of (0, 0), and a paint that threw would take the window with it.
    """
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return (0, 0, 0, 0)
    scale = min(float(dst_w) / float(src_w), float(dst_h) / float(src_h))
    w = max(1, int(round(src_w * scale)))
    h = max(1, int(round(src_h * scale)))
    return ((dst_w - w) // 2, (dst_h - h) // 2, w, h)


class PreviewWidget(QWidget):
    """Paints the most recent frame, letterboxed, and holds its buffer alive while it does.

    IT IS ALSO THE SURFACE EVERY VIDEO JOB IS DONE ON. Drawing vial positions, replaying a
    recording and watching a run all happen HERE, in the window, rather than in a cv2 window in a
    child process. Two things make that possible and both are on this widget:

      * `overlay` -- an object with ``paint(painter, view)``, called after the frame is drawn and
        given this widget so it can map image coordinates to widget ones. The frame underneath is
        never modified, so what the operator draws on is the picture the camera sent.
      * `to_image` / `to_widget` -- the inverse and forward of `fit_rect`. A click lands at a widget
        pixel and has to become an IMAGE pixel before it can be a polygon vertex, or the saved
        calibration is in screen coordinates of whatever size the window happened to be.

    THE cv2 WINDOW HAD TO KNOW HOW BIG THE DESKTOP WAS; this one does not. `live_vial_selector.
    screen_view_limit` exists because an AUTOSIZE cv2 window draws at the frame's own pixel size and
    silently runs off the screen edge -- the regression that put the lower vial row where it could
    not be clicked. A letterboxed Qt widget inside a layout cannot do that: it is given a rectangle
    and it fits the frame into it, at any window size, with the mapping staying exact.
    """

    #: Where the operator clicked, in IMAGE pixels (floats -- the caller rounds if it wants whole
    #: pixels). Clicks on the letterbox margin are not emitted at all: they are not on the picture.
    clicked = Signal(float, float)
    #: A `handle_key` key name. Emitted only while `interactive` is on, so ordinary use of the
    #: window cannot type into a drawing session that is not happening.
    key_pressed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._array = None
        self._image: Optional[QImage] = None
        self._interactive = False
        self.overlay = None
        self.placeholder = "No picture - the camera is not open"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

    # -- interaction ---------------------------------------------------------------------------
    def set_overlay(self, overlay) -> None:
        """Draw `overlay` over every frame from now on. None removes it."""
        self.overlay = overlay
        self.update()

    def set_interactive(self, on: bool) -> None:
        """Take clicks and keys, or stop taking them.

        FOCUS IS ONLY TAKEN WHILE A JOB WANTS IT. A frame view that always grabbed the keyboard
        would swallow the keystrokes the settings pane is entitled to -- and the window already
        goes out of its way to keep initial focus off anything that edits a camera setting.
        """
        self._interactive = bool(on)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus if on else Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        if on:
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    @property
    def interactive(self) -> bool:
        return self._interactive

    def image_rect(self) -> QRect:
        """Where the frame is actually painted inside this widget. Empty if there is no frame."""
        if self._image is None:
            return QRect(0, 0, 0, 0)
        x, y, w, h = fit_rect(self._image.width(), self._image.height(),
                              self.width(), self.height())
        return QRect(x, y, w, h)

    def to_image(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Widget pixel -> image pixel, or None if that point is not ON the picture.

        Returning None rather than a clamped edge coordinate is deliberate: a click on the black
        letterbox margin is not a vertex the operator meant to place, and silently snapping it to
        the frame edge would put a polygon corner somewhere nobody clicked.
        """
        rect = self.image_rect()
        if rect.width() <= 0 or rect.height() <= 0:
            return None
        if not (rect.x() <= px <= rect.x() + rect.width()
                and rect.y() <= py <= rect.y() + rect.height()):
            return None
        sx = self._image.width() / float(rect.width())
        sy = self._image.height() / float(rect.height())
        return ((px - rect.x()) * sx, (py - rect.y()) * sy)

    def to_widget(self, x: float, y: float) -> QPointF:
        """Image pixel -> widget pixel. The exact inverse of `to_image` inside the frame."""
        rect = self.image_rect()
        if self._image is None or rect.width() <= 0 or rect.height() <= 0:
            return QPointF(0.0, 0.0)
        sx = rect.width() / float(self._image.width())
        sy = rect.height() / float(self._image.height())
        return QPointF(rect.x() + x * sx, rect.y() + y * sy)

    def mousePressEvent(self, event) -> None:
        if not self._interactive or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        position = event.position()
        point = self.to_image(position.x(), position.y())
        if point is not None:
            self.clicked.emit(point[0], point[1])
        event.accept()

    def keyPressEvent(self, event) -> None:
        if not self._interactive:
            return super().keyPressEvent(event)
        name = key_name(event)
        if name is None:
            return super().keyPressEvent(event)
        self.key_pressed.emit(name)
        event.accept()

    def set_frame(self, array) -> None:
        """Adopt a mono8 HxW ndarray as the frame to paint. Ignores anything else.

        Shape-checked rather than trusted: a colour frame or a non-contiguous view would produce a
        QImage with the wrong stride, which renders as a diagonal smear -- and a smeared preview is
        a preview an operator will tune exposure against.
        """
        if array is None or getattr(array, "ndim", 0) != 2:
            return
        height, width = array.shape
        if width <= 0 or height <= 0:
            return
        # Replaced TOGETHER: the QImage borrows this exact buffer (see the module docstring).
        self._array = array
        self._image = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)
        self.update()

    @property
    def frame_size(self) -> Tuple[int, int]:
        return (self._image.width(), self._image.height()) if self._image else (0, 0)

    def clear(self) -> None:
        self._array = None
        self._image = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            return
        rect = self.image_rect()
        if rect.width() and rect.height():
            painter.drawImage(rect, self._image)
        if self.overlay is None:
            return
        # THE OVERLAY MAY NEVER TAKE THE WINDOW DOWN. It is drawn on every frame of a run that is
        # watched for days; a paint that raised would raise again on the next frame and every one
        # after it. A broken overlay costs its own drawing and nothing else.
        try:
            painter.save()
            self.overlay.paint(painter, self)
        except Exception:
            pass
        finally:
            painter.restore()


# `PreviewPane` USED TO LIVE HERE and has been deleted rather than left unused. It was the picture
# plus its caption plus the pull timer -- and `video_stage.VideoStage` is now that, with the video
# JOBS as well. Keeping both would leave two widgets that show a camera, one of which is wired to
# nothing: the next person to add something to "the preview" would have a 50% chance of adding it
# to the one that is not on screen. The caption logic moved with it, to `video_stage._camera_caption`.
