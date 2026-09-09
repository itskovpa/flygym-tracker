"""Cumulative pixel-motion overlay on a fixed video frame for each drum face."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QComboBox, QDockWidget, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from flygym_tracker.gui.preview import fit_rect
from flygym_tracker.gui.stepper import StepperField
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
    alpha = (.8 * np.sqrt(level))[:, :, None]
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
    threshold_requested = Signal(float)
    background_window_requested = Signal(float)

    def __init__(self, snapshot, parent=None):
        super().__init__(parent)
        self.snapshot = snapshot
        self._rendered = None
        layout = QVBoxLayout(self)
        self.face_box = QComboBox()
        self.face_box.addItems(['Face A', 'Face B'])
        self.face_box.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.face_box)
        self.mode_box = QComboBox()
        self.mode_box.addItems(['Accumulated activity', 'Live detection', 'Mean image (frame sum)',
                                'Relative occupancy (lighting corrected)'])
        self.mode_box.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.mode_box)
        self.background_controls = QWidget()
        background_layout = QHBoxLayout(self.background_controls)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.addWidget(QLabel('Background window (s)'))
        self.background_window = QDoubleSpinBox()
        self.background_window.setRange(10, 600)
        self.background_window.setDecimals(0)
        self.background_window.setSingleStep(10)
        self.background_window.setValue(120)
        self.background_window.setKeyboardTracking(False)
        stepper = StepperField(self.background_window)
        self.background_window.valueChanged.connect(stepper.refresh_step_limits)
        self.background_window.valueChanged.connect(self.background_window_requested.emit)
        background_layout.addWidget(stepper)
        layout.addWidget(self.background_controls)
        self.threshold_controls = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_controls)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(QLabel('Pixel threshold'))
        self.threshold_box = QDoubleSpinBox()
        self.threshold_box.setRange(0, 255)
        self.threshold_box.setDecimals(1)
        self.threshold_box.setKeyboardTracking(False)
        self.threshold_stepper = StepperField(self.threshold_box)
        self.threshold_box.valueChanged.connect(self.threshold_stepper.refresh_step_limits)
        self.threshold_stepper.refresh_step_limits()
        threshold_layout.addWidget(self.threshold_stepper)
        self.apply_button = QPushButton('Apply to detector')
        self.apply_button.clicked.connect(lambda: self.threshold_requested.emit(self.threshold_box.value()))
        threshold_layout.addWidget(self.apply_button)
        self._threshold_initialized = False
        layout.addWidget(self.threshold_controls)
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
        self.legend_low = QLabel('low activity')
        legend_labels.addWidget(self.legend_low)
        legend_labels.addStretch()
        self.legend_high = QLabel('high activity')
        legend_labels.addWidget(self.legend_high)
        layout.addLayout(legend_labels)
        self.note = QLabel('Motion detections divided by measured frame pairs at each pixel. '
                      'Registered to a static frame per face; image and shared scale refresh every 10 s '
                      'of recording time and at run end. Rotation and settling are excluded. '
                      'Uncolored pixels have no accumulated motion. Replay video to populate; '
                      'per-vial CSV totals cannot reconstruct this map.')
        self.note.setWordWrap(True)
        self._cumulative_note = self.note.text()
        layout.addWidget(self.note)
        self.refresh()

    def refresh(self):
        if self.snapshot.get('error') and self.mode_box.currentIndex() != 1:
            self.heatmap.set_image(None)
            self.range_label.setText(self.snapshot['error'])
            return
        is_live = self.mode_box.currentIndex() == 1
        self.threshold_controls.setVisible(is_live)
        self.background_controls.setVisible(self.mode_box.currentIndex() == 3)
        self.face_box.setVisible(not is_live)
        self.legend.setVisible(not is_live)
        self.legend_low.setVisible(not is_live)
        self.legend_high.setVisible(not is_live)
        if is_live:
            self._rendered = None
            self.note.setText('Red pixels are the exact motion detections in this video frame. '
                              'Preview refreshes up to 5 times/s; the detector processes every frame. '
                              'Use Measure noise floor on a still scene without animal motion for a '
                              'noise-based starting threshold (mean + k standard deviations; default k=5), then '
                              'inspect detections here. Changes affect subsequent measurements and '
                              'are logged; use Save settings to retain them. Cumulative history is not recomputed.')
            data = self.snapshot.get('live')
            if data is None:
                self.heatmap.set_image(None)
                self.range_label.setText('Start a run or replay to inspect live activity.')
                self.apply_button.setEnabled(False)
                return
            self.apply_button.setEnabled(True)
            threshold = data.get('threshold')
            if not self._threshold_initialized and threshold is not None:
                self.threshold_box.setValue(float(threshold))
                self._threshold_initialized = True
            rgb = np.repeat(data['frame'][:, :, None], 3, axis=2)
            motion = data.get('motion')
            if motion is not None:
                rgb[motion] = np.rint(rgb[motion] * .25 + np.array([255, 30, 0]) * .75).astype(np.uint8)
                detail = '%d pixels detected' % np.count_nonzero(motion)
            else:
                detail = 'not measured (rotation, settling, unknown face or first reference frame)'
            self.heatmap.set_image(rgb)
            self.range_label.setText('%s | face %s | %.2f s | actual threshold %s | %s' %
                                     ('RUN ENDED - last frame' if data.get('ended') else 'LIVE', data.get('face'), data['elapsed_s'], threshold, detail))
            return
        self.note.setText(self._cumulative_note)
        face = 'AB'[self.face_box.currentIndex()]
        data = self.snapshot.get('faces', {}).get(face)
        mode = self.mode_box.currentIndex()
        key = (mode, face, id(data), self.snapshot.get('elapsed_s'))
        if key == self._rendered:
            return
        self._rendered = key
        if data is None:
            self.heatmap.set_image(None)
            self.range_label.setText('No spatial activity frames for face %s yet.' % face)
            return
        if mode == 2:
            valid = data.get('valid')
            mean = data.get('mean')
            if mean is None or valid is None or not np.any(valid):
                self.heatmap.set_image(None)
                self.range_label.setText('Replay video to accumulate the mean image.')
                return
            lo, hi = float(mean[valid].min()), float(mean[valid].max())
            shown = data['background'].copy()
            shown[valid] = (np.rint((mean[valid]-lo)*255/(hi-lo)).astype(np.uint8)
                            if hi > lo else np.rint(mean[valid]).astype(np.uint8))
            self.heatmap.set_image(np.repeat(shown[:, :, None], 3, axis=2))
            self.legend.setVisible(False)
            self.legend_low.setText('dark'); self.legend_high.setText('bright')
            self.range_label.setText('Face %s | %d stationary frames | mean intensity %.2f-%.2f '
                                     '| updated at %.1f s' % (face, data['image_frames'], lo, hi,
                                                             self.snapshot['elapsed_s']))
            self.note.setText('Exact 64-bit pixel sum divided by each pixel sample count. '
                              'Display contrast is rescaled every 10 s; accumulated data is never rescaled. '
                              'Only registered vial pixels are averaged. Dark walls and stationary flies '
                              'both remain dark: this is an average image, not an occupancy probability.')
        elif mode == 3:
            maps = {name: entry['occupancy'] for name, entry in self.snapshot.get('faces', {}).items()
                    if 'occupancy' in entry}
            if face not in maps:
                self.heatmap.set_image(None)
                self.range_label.setText('Replay video to accumulate relative occupancy.')
                return
            maximum = max(float(values.max()) for values in maps.values())
            self.heatmap.set_image(overlay_image(data['background'], maps[face], maximum))
            self.legend_low.setText('0'); self.legend_high.setText('%.1f%% relative darkness' % (100*maximum))
            self.range_label.setText('Face %s | %d corrected frames | %d background samples | scale 0-%.1f%% '
                                     '| window %.0f s | updated at %.1f s' % (face, data['occupancy_frames'], data['background_samples'],
                                                             100*maximum, self.snapshot.get('background_window_s', 120),
                                                             self.snapshot['elapsed_s']))
            self.note.setText(('Learning background. ' if data['background_learning'] else '') +
                              'Threshold-free relative darkness after per-vial brightness correction. '
                              'Background is the temporal 90th percentile of up to 32 frames in the selected '
                              'moving window (recording seconds), separately for each face. Corrected darkness '
                              'accumulates over the run after background warm-up. Changing the window relearns '
                              'the background and keeps earlier corrected history. Resting flies and changes within '
                              'a vial and registration errors remain limitations. Colors are not time occupied.')
        else:
            values = data.get('rate', data['counts'])
            maximum = self.snapshot.get('rate_maximum', self.snapshot['maximum'])
            self.heatmap.set_image(overlay_image(data['background'], values, maximum))
            self.legend_low.setText('0'); self.legend_high.setText('%.1f%% of frame pairs' % (100*maximum))
            self.range_label.setText('Face %s | %d measured frame pairs | shared scale 0-%.1f%% detections/pixel '
                                     '| updated at %.1f s' % (face, data['frames'], 100*maximum,
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
