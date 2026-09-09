"""Cumulative pixel-motion overlay on a fixed video frame for each drum face."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QComboBox, QDockWidget, QHBoxLayout, QLabel, QVBoxLayout, QWidget
from flygym_tracker.gui.preview import fit_rect
from flygym_tracker.gui import theme

HEATMAP_KEY = 'activity_heatmap'
HEATMAP_LABEL = 'spatial activity heatmap'


def overlay_image(background, counts, maximum):
    """RGB rendering only; never smooth, relocate or change accumulated pixel counts."""
    rgb = np.repeat(background[:, :, None], 3, axis=2)
    if maximum <= 0:
        return rgb
    level = np.clip(counts.astype(np.float32) / maximum, 0, 1)
    heat = np.empty(rgb.shape, dtype=np.float32)
    for channel, values in enumerate(((30, 255, 255), (100, 220, 30), (255, 20, 0))):
        heat[:, :, channel] = np.interp(level, (0, .5, 1), values)
    alpha = np.where(counts > 0, .15 + .65 * np.sqrt(level), 0)[:, :, None]
    return np.rint(rgb * (1-alpha) + heat * alpha).astype(np.uint8)


class ActivityHeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.setMinimumSize(240, 200)

    def set_image(self, rgb):
        self.image = None if rgb is None else QImage(
            rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
            QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme.INK_1))
        if self.image is not None:
            target = QRect(*fit_rect(self.image.width(), self.image.height(), self.width(), self.height()))
            painter.drawImage(target, self.image)
        else:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             'Waiting for stationary video frames')


class ActivityHeatmapPanel(QWidget):
    def __init__(self, snapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self._rendered = None
        layout = QVBoxLayout(self)
        self.face_box = QComboBox()
        self.face_box.addItems(['Face A', 'Face B'])
        self.face_box.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.face_box)
        self.range_label = QLabel()
        self.range_label.setWordWrap(True)
        layout.addWidget(self.range_label)
        self.heatmap = ActivityHeatmapWidget()
        layout.addWidget(self.heatmap, 1)
        self.legend = QLabel('')
        self.legend.setFixedHeight(16)
        self.legend.setStyleSheet('color: white; padding: 4px; background: '
                                 'qlineargradient(x1:0,y1:0,x2:1,y2:0, '
                                 'stop:0 rgb(30,100,255), stop:0.5 rgb(255,220,20), stop:1 rgb(255,30,0));')
        layout.addWidget(self.legend)
        legend_labels = QHBoxLayout()
        legend_labels.addWidget(QLabel('low activity'))
        legend_labels.addStretch()
        legend_labels.addWidget(QLabel('high activity'))
        layout.addLayout(legend_labels)
        note = QLabel('Accumulated motion detections per pixel since the run began. '
                      'Static background per face; image and shared scale refresh every 10 s '
                      'of recording time and at run end. Rotation and settling are excluded. '
                      'Uncolored pixels have no accumulated motion. Replay video to populate; '
                      'per-vial CSV totals cannot reconstruct this map.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.refresh()

    def refresh(self):
        face = 'AB'[self.face_box.currentIndex()]
        data = self.snapshot.get('faces', {}).get(face)
        key = (face, id(data), self.snapshot.get('elapsed_s'))
        if key == self._rendered:
            return
        self._rendered = key
        if data is None:
            self.heatmap.set_image(None)
            self.range_label.setText('No spatial activity frames for face %s yet.' % face)
            return
        maximum = self.snapshot['maximum']
        self.heatmap.set_image(overlay_image(data['background'], data['counts'], maximum))
        self.range_label.setText('Face %s | %d measured frame pairs | shared scale 0–%d detections/pixel '
                                 '| updated at %.1f s' % (face, data['frames'], maximum,
                                                         self.snapshot['elapsed_s']))


class ActivityHeatmapDock(QDockWidget):
    def __init__(self, snapshot, parent=None):
        super().__init__('Spatial activity heatmap', parent)
        self.field = HEATMAP_KEY
        self.setObjectName('plot-' + HEATMAP_KEY)
        self.panel = ActivityHeatmapPanel(snapshot)
        self.setWidget(self.panel)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                         QDockWidget.DockWidgetFeature.DockWidgetClosable)

    def refresh(self):
        self.panel.refresh()
