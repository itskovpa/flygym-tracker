"""Bounded thread/process runner for experimental full-frame background centroid tracking."""
import csv
import json
import queue
import threading
import time
from pathlib import Path
import numpy as np
from flygym_tracker.fast_tracking import FastParams, GroupTracker, brightness_reference


def _run(inbox, outbox, params, window, csv_path):
    from flygym_tracker.cv_setup import configure_opencv
    from flygym_tracker.rolling_background import RollingBackground
    configure_opencv()
    faces = {}
    metrics = dict(frames_completed=0, failures=0, warmup_frames=0, queue_delay_ms_total=0.,
                   processing_ms_total=0., processing_ms_max=0., queue_delay_ms_max=0.)
    last_publish = -1e9
    latest = {}
    output = open(csv_path, 'w', newline='', encoding='utf-8')
    columns = ['elapsed_s','frame_index','face','dwell','vial_id','detections','known_group_count','unknown_groups',
               'observed_distance_px','gap_displacement_px','points_json','groups_json']
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    def publish(final=False):
        payload = dict(metrics=metrics.copy(), latest=latest, final=final)
        if final:
            outbox.put(payload)
        else:
            try: outbox.put_nowait(payload)
            except queue.Full: pass
    try:
        while True:
            item = inbox.get()
            if item is None: break
            gray,t,face,dwell,geometry,frame_index,queued = item
            start = time.perf_counter()
            delay = (start-queued)*1000
            metrics['queue_delay_ms_total'] += delay
            metrics['queue_delay_ms_max'] = max(metrics['queue_delay_ms_max'],delay)
            try:
                state = faces.get(face)
                if state is None:
                    state = dict(background=RollingBackground(gray.size, window, accumulate=False),
                                 geometry=None, trackers={}, dwell=None,
                                 normal=np.empty(gray.shape,np.float32), contrast=np.empty(gray.shape,np.float32),
                                 background_ref=None, cached_background=None)
                    faces[face] = state
                if geometry is not None:
                    state['geometry'] = geometry
                    state['outlines'] = {}
                    mask = np.zeros(gray.shape,np.uint8)
                    for vid,((x,y,w,h),sub) in geometry.items():
                        mask[y:y+h,x:x+w][sub]=255
                        padded=np.pad(sub,1)
                        interior=padded[:-2,1:-1]&padded[2:,1:-1]&padded[1:-1,:-2]&padded[1:-1,2:]
                        state['outlines'][vid]=np.argwhere(sub&~interior)
                    state['mask'] = mask
                if state['dwell'] != dwell:
                    state['trackers'] = {}
                    state['dwell'] = dwell
                light = brightness_reference(gray,state['mask'],metrics)
                normal = state['normal']
                np.divide(gray,max(light,1),out=normal,casting='unsafe')
                rolling = state['background']
                rolling.add(normal.ravel(),t)
                data = dict(face=face,elapsed_s=t,frame_index=frame_index,dwell=dwell,vials={})
                if rolling.background is None:
                    metrics['warmup_frames'] += 1
                else:
                    if state['background_ref'] is not rolling.background:
                        state['background_ref'] = rolling.background
                        state['cached_background'] = rolling.background.reshape(gray.shape).astype(np.float32)
                    background = state['cached_background']
                    contrast = state['contrast']
                    np.subtract(background,normal,out=contrast)
                    valid = background>0
                    np.divide(contrast,background,out=contrast,where=valid)
                    contrast[~valid]=0
                    np.clip(contrast,0,1,out=contrast)
                    for vid,((x,y,w,h),sub) in state['geometry'].items():
                        if w<=0 or h<=0 or not sub.any(): continue
                        if vid not in state['trackers']: state['trackers'][vid]=GroupTracker(params)
                        tracker = state['trackers'][vid]
                        result = tracker.update(contrast[y:y+h,x:x+w],sub,t,frame_index=frame_index)
                        metrics['cv_lock_wait_ms_total'] += tracker.last_lock_wait_ms
                        result.pop('binary')
                        result['bbox'] = (x,y,w,h)
                        result['outline'] = state['outlines'][vid]
                        data['vials'][vid] = result
                        writer.writerow(dict(elapsed_s=t,frame_index=frame_index,face=face,dwell=dwell,vial_id=vid,
                            detections=len(result['measured']),
                            known_group_count=sum(g['count'] or 0 for g in result['groups']),
                            unknown_groups=sum(g['count'] is None for g in result['groups']),
                            observed_distance_px=result['observed_distance_px'],gap_displacement_px=result['gap_displacement_px'],
                            points_json=json.dumps(result['measured']),groups_json=json.dumps(result['groups'])))
                # Preview snapshots own their arrays; publishing at most 5 Hz of wall time.
                now = time.perf_counter()
                if now-last_publish>=.2:
                    data['frame'] = gray.copy()
                    data['normalized'] = normal.copy()
                    data['background'] = None if rolling.background is None else state['cached_background'].copy()
                    data['geometry'] = state['geometry']
                    data['parameters'] = vars(params).copy()
                    data['background_window_s'] = window
                    data['contrast'] = None if rolling.background is None else state['contrast'].copy()
                    latest = data
                    last_publish = now
                metrics['frames_completed'] += 1
            except Exception as exc:
                metrics['failures'] += 1
                metrics['last_error'] = repr(exc)
            ms = (time.perf_counter()-start)*1000
            metrics['processing_ms_total'] += ms
            metrics['processing_ms_max'] = max(metrics['processing_ms_max'],ms)
            if latest and latest.get('elapsed_s') == t:
                output.flush()
                publish()
    finally:
        for state in faces.values(): state['background'].close()
        output.close()
        publish(final=True)


class FastTrackingPool:
    """Same lifecycle as FlyTrackingPool, but exports explicit group/centroid records."""
    fast_mode = True
    def __init__(self, output_dir, stamp, *, backend='thread', params=None, window=120, depth=8):
        if backend not in ('thread','process'): raise ValueError('Unknown fast tracking backend')
        self.backend = backend
        if not np.isfinite(window) or not 10<=window<=600: raise ValueError('Background window must be 10-600 s')
        self.params = params or FastParams()
        self.window = window
        self.path = str(Path(output_dir)/('fast_tracking_%s.csv'%stamp))
        if backend == 'process':
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            self.inbox,self.outbox = ctx.Queue(depth),ctx.Queue(2)
            self.worker = ctx.Process(target=_run,args=(self.inbox,self.outbox,self.params,window,self.path),daemon=True)
        else:
            self.inbox,self.outbox = queue.Queue(depth),queue.Queue(2)
            self.worker = threading.Thread(target=_run,args=(self.inbox,self.outbox,self.params,window,self.path),daemon=True)
        self.frames_submitted = self.frames_dropped = self.dwell_index = 0
        self._metrics = {}
        self.latest = {}
        self._offsets = {}
        self._closed = False
        self._started = False
        self._queues_closed = False

    def start(self):
        self.worker.start()
        self._started = True

    def _drain(self):
        if self._queues_closed: return
        while True:
            try: result = self.outbox.get_nowait()
            except queue.Empty: break
            self._metrics = result['metrics']
            self.latest = result['latest']

    def submit(self,gray,t,face,offset,geometry,frame_index=None):
        self._drain()
        if self._closed or not self.worker.is_alive():
            self.frames_dropped += 1
            return False
        changed = self._offsets.get(face) != offset
        if frame_index is None: frame_index=self.frames_submitted+self.frames_dropped
        item = (gray.copy(),t,face,self.dwell_index,dict(geometry) if changed else None,frame_index,time.perf_counter())
        try: self.inbox.put_nowait(item)
        except queue.Full:
            self.frames_dropped += 1
            return False
        self._offsets[face] = offset
        self.frames_submitted += 1
        return True

    def reset_dwell(self): self.dwell_index += 1
    def take_summaries(self): return {}  # dedicated CSV has different semantics from legacy behaviour rows

    def tracks(self):
        self._drain()
        if self.latest.get('dwell') != self.dwell_index: return {}
        return {int(vid): [[(x+d['bbox'][0],y+d['bbox'][1]) for x,y in path] for path in d['paths']]
                for vid,d in self.latest.get('vials',{}).items()}

    def stats(self):
        self._drain()
        result = dict(self._metrics,backend=self.backend,frames_submitted=self.frames_submitted,
                      frames_dropped=self.frames_dropped,csv=self.path,dwells=self.dwell_index)
        completed = result.get('frames_completed',0)
        attempted = self.frames_submitted+self.frames_dropped
        result['fraction_completed'] = completed/attempted if attempted else 0.
        result['processing_mean_ms'] = result.get('processing_ms_total',0)/max(completed+result.get('failures',0),1)
        result['queue_delay_mean_ms'] = result.get('queue_delay_ms_total',0)/max(completed+result.get('failures',0),1)
        result['cv_lock_wait_mean_ms'] = result.get('cv_lock_wait_ms_total',0)/max(completed+result.get('failures',0),1)
        result['pending_frames'] = self.frames_submitted-completed-result.get('failures',0)
        if self._started and not self._closed and not self.worker.is_alive():
            result['last_error'] = 'Fast tracking worker stopped unexpectedly'
        return result

    def close(self):
        if self._closed or not self._started: return
        self._closed = True
        # Drain every accepted frame and the final result; never join while a child is blocked publishing.
        while self.worker.is_alive():
            self._drain()
            try:
                self.inbox.put(None,timeout=.05)
                break
            except queue.Full: pass
        while self.worker.is_alive():
            self._drain()
            self.worker.join(timeout=.05)
        self._drain()
        if self.backend == 'process':
            self.inbox.cancel_join_thread()
            self.inbox.close(); self.outbox.close()
            self._queues_closed = True
