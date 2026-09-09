"""Tracking controls and inspection, with draft previews separate from recorded results."""
import numpy as np
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget)
from .activity_heatmap import ActivityHeatmapWidget
from .stepper import StepperField


FIELDS = [
    ('spatial_background_window_s', 'Background window (s)', 10, 600, 120,
     'Learn the bright background separately for each face over this moving window. Longer follows drift more slowly. Preview uses the background already learned by the current run.'),
    ('fast_tracking_threshold', 'Darkness threshold (%)', .1, 99.9, 15.,
     '15 means at least 15% darker than the learned background. Lower includes faint flies and more noise. This is different from the frame-difference activity threshold.'),
    ('fast_tracking_min_area', 'Minimum blob area (pixels)', 1, 100000, 8,
     'Discard connected regions smaller than this area. Inspect the purple regions while adjusting.'),
    ('fast_tracking_max_area', 'Maximum single-fly area (pixels)', 1, 100000, 300,
     'Larger regions remain unresolved groups (orange); they are not discarded or counted as one fly.'),
    ('fast_tracking_max_speed', 'Maximum link speed (pixels/s)', 1., 10000., 150.,
     'Maximum centroid travel used when matching between frames. Too high can connect different flies.'),
    ('fast_tracking_max_gap_s', 'Disappearance tolerance (s)', 0., 10., .25,
     'Keep a missing trajectory available for this long. Straight-line displacement across the gap is reported separately.'),
    ('fast_tracking_max_group_s', 'Group retention (s)', .01, 60., 1.,
     'Retain previously observed flies in a merged region for this long after their last individual measurement. No individual movement is inferred inside a group.'),
]


def detection_preview(data, threshold, minimum, maximum):
    """Re-threshold one snapshot per vial; do not invent temporal tracks for draft settings."""
    import cv2
    from flygym_tracker.cv_setup import CV_LOCK
    dark = data['contrast']
    binary = np.zeros(dark.shape, np.uint8)
    rgb = np.repeat(data['frame'][:, :, None], 3, axis=2)
    counts = [0, 0, 0]
    for bbox, mask in data['geometry'].values():
        x, y, w, h = bbox
        selected = ((dark[y:y+h, x:x+w] > threshold) & mask).astype(np.uint8)
        binary[y:y+h, x:x+w] |= selected * 255
        with CV_LOCK:
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(selected, 8)
        crop = rgb[y:y+h, x:x+w]
        colors = np.zeros((n, 3), np.uint8)
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            kind = 0 if area < minimum else 2 if area > maximum else 1
            counts[kind] += 1
            color = [(170, 80, 210), (60, 230, 100), (255, 160, 35)][kind]
            colors[i] = color
        foreground = labels > 0
        crop[foreground] = colors[labels[foreground]]
    return binary, rgb, counts


class TrackingSetup(QDialog):
    applied = Signal(dict)

    def __init__(self, state, snapshot, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Tracking setup and inspection')
        self.resize(1120, 760)
        self.snapshot = snapshot
        self._last_key = None
        self._frozen = None
        layout = QVBoxLayout(self)
        intro = QLabel('1. Enable fast tracking and apply settings.  2. Start a run or replay.  '
                       '3. Inspect the stages and tune detection.  4. Apply and restart to record with the new settings.')
        intro.setWordWrap(True)
        layout.addWidget(intro)
        columns = QHBoxLayout()
        layout.addLayout(columns, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)
        body = QWidget()
        form = QFormLayout(body)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        scroll.setWidget(body)
        columns.addWidget(scroll)
        self.enabled = QCheckBox('Enable fly trajectories')
        self.enabled.setChecked(state.get('track_flies', True))
        form.addRow(self.enabled)
        self.backend = QComboBox()
        for label, value in [('Fast centroids — separate process (recommended)', 'process'),
                             ('Fast centroids — thread', 'thread'),
                             ('Tracker from configuration file', 'configured')]:
            self.backend.addItem(label, value)
        self.backend.setCurrentIndex(max(0, self.backend.findData(state.get('fast_tracking_backend', 'process'))))
        form.addRow('Processing', self.backend)
        self.controls = {}
        self.steppers = {}
        self.fast_widgets = []
        for key, label, low, high, default, help_text in FIELDS:
            spin = QDoubleSpinBox() if isinstance(default, float) else QSpinBox()
            spin.setRange(low, high)
            if isinstance(default, float):
                spin.setDecimals(2)
                spin.setSingleStep(.05 if key.endswith('_s') else 1.)
            spin.setValue(float(state.get(key, default)) if isinstance(default, float) else int(state.get(key, default)))
            spin.setKeyboardTracking(False)
            spin.setToolTip(help_text)
            note = QLabel(help_text)
            note.setWordWrap(True)
            stepper = StepperField(spin)
            spin.valueChanged.connect(stepper.refresh_step_limits)
            stepper.refresh_step_limits()
            self.steppers[key] = stepper
            form.addRow(label, stepper)
            form.addRow(note)
            self.controls[key] = spin
            if key != 'spatial_background_window_s':
                self.fast_widgets.extend([stepper, note])
            spin.valueChanged.connect(self.refresh)
        self.backend.currentIndexChanged.connect(self._backend_changed)
        note = QLabel('Links reset on drum rotation. Tracks have temporary labels; these are not persistent animal identities.')
        note.setWordWrap(True)
        form.addRow(note)
        right = QVBoxLayout()
        columns.addLayout(right, 1)
        inspection_row = QHBoxLayout()
        self.vial = QComboBox()
        self.vial.addItem('Whole frame', None)
        self.vial.currentIndexChanged.connect(self.refresh)
        inspection_row.addWidget(self.vial, 1)
        self.freeze = QCheckBox('Freeze frame for inspection')
        self.freeze.toggled.connect(self._freeze_changed)
        inspection_row.addWidget(self.freeze)
        right.addLayout(inspection_row)
        self.stage = QComboBox()
        self.stage.addItems(['1. Raw video', '2. Brightness normalized', '3. Learned background',
                            '4. Background subtraction', '5. Binary threshold (draft)',
                            '6. Blob classification (draft)', '7. Recorded trajectories'])
        self.stage.currentIndexChanged.connect(self.refresh)
        right.addWidget(self.stage)
        self.image = ActivityHeatmapWidget()
        right.addWidget(self.image, 1)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        right.addWidget(self.caption)
        self.performance = QLabel('No tracking run yet.')
        self.performance.setWordWrap(True)
        right.addWidget(self.performance)
        self.message = QLabel('Draft detection changes affect this preview only. All saved tracking settings apply to the next run.')
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QHBoxLayout()
        self.apply_button = QPushButton('Apply to next run')
        self.apply_button.clicked.connect(self._apply)
        buttons.addWidget(self.apply_button)
        close = QPushButton('Close')
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self._backend_changed()

    def _backend_changed(self):
        for widget in self.fast_widgets:
            widget.setEnabled(self.backend.currentData() != 'configured')
        for stepper in self.steppers.values():
            stepper.refresh_step_limits()
        self.refresh()

    def values(self):
        return dict({key: spin.value() for key, spin in self.controls.items()},
                    track_flies=self.enabled.isChecked(), fast_tracking_backend=self.backend.currentData())

    def _freeze_changed(self, checked):
        self._frozen = (self.snapshot().get('fast_tracking') or {}) if checked else None
        self._last_key = None
        self.refresh()

    def _apply(self):
        if self.controls['fast_tracking_min_area'].value() > self.controls['fast_tracking_max_area'].value():
            self.message.setText('Minimum blob area must not exceed maximum single-fly area. Nothing saved.')
            return
        self.applied.emit(self.values())
        self.message.setText('Saved for the next run. The current run and its learned background remain unchanged. Restart the run to use these values.')

    def refresh(self, *_):
        if not self.isVisible():
            return
        live = self.snapshot().get('fast_tracking') or {}
        data = self._frozen if self._frozen is not None else live
        ids = sorted(data.get('geometry', {}))
        if ids != [self.vial.itemData(i) for i in range(1, self.vial.count())]:
            selected = self.vial.currentData()
            self.vial.blockSignals(True)
            self.vial.clear()
            self.vial.addItem('Whole frame', None)
            for vid in ids:
                self.vial.addItem('Vial %s' % vid, vid)
            self.vial.setCurrentIndex(max(0, self.vial.findData(selected)))
            self.vial.blockSignals(False)
        stats = data.get('stats') or {}
        self.performance.setText('Worker: %.1f ms/frame · queue delay %.1f ms · completed %s/%s · pending %s · dropped %s · warm-up %s · failures %s%s' % (
            stats.get('processing_mean_ms', 0), stats.get('queue_delay_mean_ms', 0),
            stats.get('frames_completed', 0), stats.get('frames_submitted', 0),
            stats.get('pending_frames', 0), stats.get('frames_dropped', 0),
            stats.get('warmup_frames', 0), stats.get('failures', 0),
            (' · ' + stats['last_error']) if stats.get('last_error') else ''))
        stage = self.stage.currentIndex()
        key = (id(data.get('frame')), data.get('elapsed_s'), data.get('dwell'), stage,
               self.vial.currentData(), tuple(spin.value() for spin in self.controls.values()))
        if key == self._last_key:
            return
        self._last_key = key
        if data.get('frame') is None:
            self.image.set_image(None)
            self.caption.setText('Enable fast tracking, apply, then start a run or replay. Only stationary frames are inspected; rotation and settling are excluded.')
            return
        source = ['frame', 'normalized', 'background', 'contrast'][min(stage, 3)]
        array = data.get(source)
        if array is None:
            self.image.set_image(None)
            self.caption.setText('Background is warming up. At least four samples of this face are needed.')
            return
        scale = [255., 1.4, 1.4, .5][min(stage, 3)]
        rgb = np.repeat(np.rint(np.clip(array / scale, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        explanation = ['Raw camera intensities, 0–255.',
            'Frame divided by P90 brightness inside the vial masks. Display range 0–1.4.',
            'Temporal P90 of normalized frames for this face. Display range 0–1.4.',
            'Relative darkness: max(0, (background − frame) / background). White = 50% or more.',
            'Draft threshold inside each vial; no area filtering yet.',
            'Draft: green = single-size blobs, purple = too small, orange = unresolved large regions.',
            'Recorded results from the current run: green = measured, orange = unresolved groups. Draft controls do not change these trajectories.'][stage]
        if stage in (5, 4):
            if not data.get('geometry'):
                self.caption.setText('Restart the application and run to receive inspection geometry.')
                self.image.set_image(None)
                return
            selected = self.vial.currentData()
            preview_data = data if selected is None else dict(data, geometry={selected: data['geometry'][selected]})
            binary, blobs, counts = detection_preview(preview_data,
                self.controls['fast_tracking_threshold'].value()/100,
                self.controls['fast_tracking_min_area'].value(),
                self.controls['fast_tracking_max_area'].value())
            rgb = blobs if stage == 5 else np.repeat(binary[:, :, None], 3, axis=2)
            explanation += ' Small: %d · single-size: %d · large: %d. Blob counts are not verified fly counts.' % tuple(counts)
        if stage == 6:
            rgb = np.repeat(data['frame'][:, :, None], 3, axis=2)
            import cv2
            from flygym_tracker.cv_setup import CV_LOCK
            for result in data.get('vials', {}).values():
                x, y, _, _ = result['bbox']
                for path in result.get('paths', []):
                    if len(path) > 1:
                        with CV_LOCK:
                            cv2.polylines(rgb, [np.rint([(x+px, y+py) for px, py in path]).astype(np.int32)], False, (60, 230, 100), 1)
                for point in result['measured'] + result['groups']:
                    px, py = x+point['x'], y+point['y']
                    color = (255, 160, 35) if 'count' in point else (60, 230, 100)
                    with CV_LOCK:
                        cv2.circle(rgb, (round(px), round(py)), 3, color, 2)
        for result in data.get('vials', {}).values():
            x, y, _, _ = result['bbox']
            outline = result.get('outline')
            if outline is not None:
                rgb[outline[:, 0]+y, outline[:, 1]+x] = (70, 140, 255)
        if self.vial.currentData() is not None:
            (x, y, w, h), _ = data['geometry'][self.vial.currentData()]
            rgb = np.ascontiguousarray(rgb[y:y+h, x:x+w])
        self.image.set_image(rgb)
        current = data.get('parameters') or {}
        if current:
            explanation += '\nCurrent run: threshold %.1f%%, area %s–%s px, background window %s s.' % (100*current['threshold'], current['min_area'], current['max_single_area'], data.get('background_window_s', '?'))
        self.caption.setText('Face %s · %.2f s · frame %s\n%s' % (
            data.get('face'), data.get('elapsed_s', 0), data.get('frame_index', '?'), explanation))
        if self._frozen is not None:
            self.caption.setText('Frozen snapshot — processing continues.\n' + self.caption.text())
