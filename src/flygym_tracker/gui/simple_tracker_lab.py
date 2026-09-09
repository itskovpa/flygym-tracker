"""Interactive diagnostic lab for the simple tracker on an annotated recording."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QSlider, QVBoxLayout, QWidget)
from flygym_tracker.calibration import load_calibration, vial_shape, bbox_from_quad, quad_polygon_mask
from flygym_tracker.cv_setup import configure_opencv, CV_LOCK
from flygym_tracker.gui import theme
from flygym_tracker.gui.activity_heatmap import ActivityHeatmapWidget
from flygym_tracker.gui.flow_layout import FlowLayout
from flygym_tracker.gui.qt_compat import NoWheelDoubleSpinBox
from flygym_tracker.gui.stepper import StepperField
from flygym_tracker.simple_tracker import SimpleTrackerParams, CentroidTracker, detect_centroids


def lab_rois(directory):
    calibration=load_calibration(directory)
    rois={}
    for face,fc in calibration.faces.items():
        with CV_LOCK:
            image=cv2.imread(fc.illum_mask_path,cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError('Missing illumination mask: '+fc.illum_mask_path)
        rois[face]={}
        for vial in fc.vials:
            if not vial.present: continue
            shape=vial_shape(vial)
            x,y,w,h=bbox_from_quad(shape) if shape is not None else (vial.x,vial.y,vial.w,vial.h)
            x0,y0=max(0,x),max(0,y)
            w,h=min(image.shape[1],x+w)-x0,min(image.shape[0],y+h)-y0
            if w<=0 or h<=0: continue
            bbox=(x0,y0,w,h)
            mask=image[y0:y0+h,x0:x0+w]==255
            if shape is not None: mask &= quad_polygon_mask(shape,bbox)
            rois[face][vial.id]=(bbox,mask)
    return rois


def selected_vials(rois,face,primary,count):
    ids=sorted(rois.get(face,{}))
    return ([primary]+[v for v in ids if v!=primary])[:count] if primary in ids else ids[:count]


class SimpleTrackerLab(QWidget):
    def __init__(self,annotations,calib):
        super().__init__()
        self.dataset=json.loads(Path(annotations).read_text())
        self.rows=self.dataset['frames']
        self.rois=lab_rois(calib)
        with CV_LOCK: self.cap=cv2.VideoCapture(self.dataset['video'])
        if not self.cap.isOpened(): raise ValueError('Cannot open test video')
        self.index=0
        self.gray=None
        self.trackers={}
        self.last_face=None
        self.timer=QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.advance)
        self.started=None
        self.presented=0
        self.setWindowTitle('FlyGym - simple centroid tracker lab')
        layout=QVBoxLayout(self)
        controls=FlowLayout(margin=0,spacing=8)
        self.inputs={}
        defaults=SimpleTrackerParams()
        for key,label,lo,hi,decimals,step in (
            ('threshold','Dark threshold',0,255,0,1),('min_area','Min area (px)',1,10000,0,1),
            ('max_area','Max area (px)',1,100000,0,10),('max_speed','Max speed (px/s)',1,5000,0,10),
            ('max_gap_s','Max gap (s)',0,5,2,.05)):
            group=QWidget()
            group_layout=QHBoxLayout(group)
            group_layout.setContentsMargins(0,0,0,0)
            group_layout.addWidget(QLabel(label))
            box=NoWheelDoubleSpinBox()
            box.setRange(lo,hi); box.setDecimals(decimals); box.setSingleStep(step)
            box.setValue(getattr(defaults,key)); box.setKeyboardTracking(False)
            field=StepperField(box)
            box.valueChanged.connect(field.refresh_step_limits)
            box.valueChanged.connect(self.parameters_changed)
            field.refresh_step_limits()
            self.inputs[key]=box
            group_layout.addWidget(field)
            controls.addWidget(group)
        layout.addLayout(controls)
        row=FlowLayout(margin=0,spacing=8)
        self.play=QPushButton('Play')
        self.play.clicked.connect(self.toggle)
        row.addWidget(self.play)
        step=QPushButton('Next frame');step.clicked.connect(self.advance);row.addWidget(step)
        rewind=QPushButton('Restart');rewind.clicked.connect(lambda:self.seek(0));row.addWidget(rewind)
        self.primary={}
        for face,default in [('A',2),('B',10)]:
            row.addWidget(QLabel('Inspect '+face))
            box=QComboBox()
            for vial in sorted(self.rois.get(face,{})): box.addItem(str(vial),vial)
            box.setCurrentIndex(max(0,box.findData(default)))
            box.currentIndexChanged.connect(self.parameters_changed)
            self.primary[face]=box;row.addWidget(box)
        row.addWidget(QLabel('Vials / visible face'))
        self.count=QComboBox()
        for n in (1,2,4,8,16):self.count.addItem(str(n),n)
        self.count.currentIndexChanged.connect(self.parameters_changed);row.addWidget(self.count)
        row.addWidget(QLabel('Playback target'))
        self.fps=QComboBox()
        for fps in (20,50):self.fps.addItem(str(fps)+' FPS',fps)
        self.fps.currentIndexChanged.connect(self.update_rate);row.addWidget(self.fps)
        layout.addLayout(row)
        views=QHBoxLayout()
        self.original=ActivityHeatmapWidget();self.binary=ActivityHeatmapWidget()
        for view,label in ((self.original,'Image + trajectories (orange = gap link)'),
                           (self.binary,'Binary blobs: green accepted; red rejected by area')):
            column=QVBoxLayout();column.addWidget(QLabel(label));column.addWidget(view,1);views.addLayout(column,1)
        layout.addLayout(views,1)
        self.scrub=QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0,len(self.rows)-1)
        self.scrub.sliderReleased.connect(lambda:self.seek(self.scrub.value()))
        layout.addWidget(self.scrub)
        self.status=QLabel('');self.status.setWordWrap(True);layout.addWidget(self.status)
        self.stats=QLabel('');self.stats.setWordWrap(True);layout.addWidget(self.stats)
        note=QLabel('Tracking uses absolute dark intensity, not frame differences. Parameters apply immediately; '
                    'changing them or seeking resets lab statistics. Rotation/settling end all active tracks. '
                    'Track numbers label fragments, not animal identities. Merges/crossings can produce wrong links. '
                    'Drum states were precomputed with the existing detector. Playing 20-FPS footage faster does not validate 50-FPS acquisition.')
        note.setWordWrap(True);layout.addWidget(note)
        self.resize(1250,850)
        first=next((r['index'] for r in self.rows if r['state']=='stationary'),0)
        self.seek(first)

    def params(self):
        return SimpleTrackerParams(**{k:int(v.value()) if k in ('threshold','min_area','max_area') else v.value() for k,v in self.inputs.items()})

    def parameters_changed(self,*args):
        self.trackers={};self.last_face=None
        if self.gray is not None: self.process(self.gray,self.rows[max(0,self.index-1)])

    def update_rate(self):
        self.timer.setInterval(round(1000/self.fps.currentData()))
        self.started=time.perf_counter();self.presented=0

    def toggle(self):
        if self.timer.isActive():self.timer.stop();self.play.setText('Play')
        else:self.update_rate();self.timer.start();self.play.setText('Pause')

    def seek(self,index):
        self.trackers={};self.last_face=None
        self.index=index
        with CV_LOCK:self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        self.started=time.perf_counter();self.presented=0
        self.advance()

    def advance(self):
        if self.index>=len(self.rows):self.timer.stop();self.play.setText('Play');return
        t=time.perf_counter()
        with CV_LOCK:
            ok,image=self.cap.read()
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if ok else None
        if gray is None:self.timer.stop();self.status.setText('Video read failed');return
        row=self.rows[self.index];self.index+=1;self.gray=gray
        self.process(gray,row)
        self.presented+=1
        wall_fps=self.presented/(time.perf_counter()-self.started) if self.started and self.timer.isActive() else 0
        self.status.setText(self.status.text()+' | decode + tracking + display preparation %.2f ms' %
                            ((time.perf_counter()-t)*1000)+(' | delivered %.1f FPS' % wall_fps if self.timer.isActive() else ' | paused'))
        if not self.scrub.isSliderDown():self.scrub.setValue(self.index-1)

    def process(self,gray,row):
        face=row['face'];state=row['state']
        if state!='stationary' or face not in self.rois:
            for tracker in self.trackers.values():tracker.reset()
            self.last_face=None
            self.original.set_image(np.repeat(gray[:,:,None],3,axis=2));self.binary.set_image(None)
            self.status.setText('%.2f s | %s | tracks reset; detection excluded' % (row['elapsed_s'],state))
            self.stats.setText('');return
        if self.last_face!=face:
            for tracker in self.trackers.values():tracker.reset()
        self.last_face=face
        try:params=self.params()
        except ValueError as e:self.timer.stop();self.play.setText('Play');self.status.setText(str(e));return
        primary=self.primary[face].currentData()
        count=self.count.currentData()
        total=0
        for vial in selected_vials(self.rois,face,primary,count):
            (x,y,w,h),mask=self.rois[face][vial]
            crop=gray[y:y+h,x:x+w]
            blobs,binary,labels,accepted=detect_centroids(crop,mask,params)
            tracker=self.trackers.setdefault((face,vial),CentroidTracker(params))
            linked=tracker.update(blobs,row['elapsed_s'],row['index'])
            total+=len(blobs)
            if vial==primary:
                image=np.repeat(crop[:,:,None],3,axis=2)
                segmentation=np.zeros(image.shape,np.uint8)
                segmentation[binary.astype(bool)]=[210,50,50]
                segmentation[np.isin(labels,accepted)]=[40,220,100]
                with CV_LOCK:
                    for result in linked:
                        track=tracker.active[result['fragment']]
                        points=list(track.points)
                        for a,b in zip(points,points[1:]):
                            cv2.line(image,(round(a[0]),round(a[1])),(round(b[0]),round(b[1])),
                                     (255,170,30) if b[2] else (40,255,100),1)
                        cv2.circle(image,(round(track.x),round(track.y)),2,(255,255,0),1)
                self.original.set_image(image);self.binary.set_image(segmentation)
                stat=tracker.statistics()
                self.stats.setText('%s%d: %d blobs | %d fragments | observed path %.1f px | gap displacement %.1f px '
                                   '(%d links) | mean linked-step speed %.1f px/s' %
                                   (face,vial,len(blobs),stat['fragments'],stat['observed_distance_px'],stat['gap_displacement_px'],
                                    stat['gap_links'],stat['mean_link_speed_px_s']))
        self.status.setText('%.2f s | face %s | %d vials processed | %d blobs (not known fly count)' %
                            (row['elapsed_s'],face,count,total))

    def closeEvent(self,event):
        self.timer.stop()
        with CV_LOCK:self.cap.release()
        super().closeEvent(event)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--calib',required=True)
    args=parser.parse_args()
    configure_opencv()
    app=QApplication([]);theme.apply(app)
    window=SimpleTrackerLab(args.annotations,args.calib);window.show()
    return app.exec()

if __name__=='__main__':raise SystemExit(main())
