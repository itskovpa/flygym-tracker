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

from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

#: How often the GUI pulls the latest frame. Faster than the worker offers them, so the box is
#: usually empty and the pull is a cheap lock-and-return; the worker's throttle sets the real rate.
PULL_INTERVAL_MS = 50


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
    """Paints the most recent frame, letterboxed, and holds its buffer alive while it does."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._array = None
        self._image: Optional[QImage] = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAutoFillBackground(False)

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
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No picture - the camera is not open")
            return
        x, y, w, h = fit_rect(self._image.width(), self._image.height(),
                              self.width(), self.height())
        if w and h:
            painter.drawImage(QRect(x, y, w, h), self._image)


class PreviewPane(QWidget):
    """The picture, its honest caption, and the timer that pulls frames off the one-slot box.

    THE PULL IS A TIMER, NOT A SIGNAL PER FRAME. Qt does not coalesce queued signals: measured, 300
    frame-sized payloads emitted at a stalled GUI thread were all still queued after half a second,
    holding every one of them in memory. A run is watched for days, and a stall of a minute -- a
    dragged window, a screen lock, a virus scan -- would queue thousands. Pulling from a one-slot
    box means a stalled GUI shows a STALE frame and drops the rest, which is the only failure mode
    here that cannot end an experiment.
    """

    def __init__(self, session, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.session = session
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.view = PreviewWidget()
        layout.addWidget(self.view, 1)

        self.caption = QLabel("Camera not open - nothing is being read")
        self.caption.setProperty("role", "note")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)

        self._timer = QTimer(self)
        self._timer.setInterval(PULL_INTERVAL_MS)
        # A bound method of an object built on the GUI thread -- the only connection style measured
        # to actually deliver on the GUI thread. See `camera_session`.
        self._timer.timeout.connect(self._pull)
        self._timer.start()

    def _pull(self) -> None:
        frame = self.session.latest.take()
        if frame is not None:
            self.view.set_frame(frame)
        self._update_caption()

    def _update_caption(self) -> None:
        shown, dropped = self.session.latest.stats
        if not self.session.is_open:
            self.caption.setText("Camera not open - nothing is being read")
            return
        delivered = self.session.measured_fps
        width, height = self.view.frame_size
        bits = []
        if width and height:
            bits.append("%dx%d" % (width, height))
        # "measured" is stated, and it means COUNTED FROM FRAMES THAT ARRIVED -- not the frame-rate
        # setting, and not the camera's own read-back. See the module docstring.
        bits.append("camera delivering %.1f fps (measured)" % delivered if delivered > 0
                    else "waiting for the first frame")
        total = shown + dropped
        if total:
            bits.append("showing %d of %d frames - %d not shown" % (shown, total, dropped))
        self.caption.setText("  -  ".join(bits))
