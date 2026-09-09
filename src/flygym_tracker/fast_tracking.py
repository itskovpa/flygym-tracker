"""Experimental centroid tracking: keep known merge counts, never invent split positions."""
from dataclasses import dataclass
import math
import time
import numpy as np
import cv2
from flygym_tracker.cv_setup import CV_LOCK


@dataclass(frozen=True)
class FastParams:
    threshold: float = .15
    min_area: int = 8
    max_single_area: int = 300
    max_speed: float = 150.
    max_gap_s: float = .25
    max_group_s: float = 1.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Tracking parameters must be finite')
        if not 0 < self.threshold < 1 or not 1 <= self.min_area <= self.max_single_area:
            raise ValueError('Require 0<threshold<1 and 1<=minimum area<=single-fly maximum')
        if self.max_speed <= 0 or self.max_gap_s < 0 or self.max_group_s <= 0:
            raise ValueError('Invalid tracking time or speed limit')


def _histogram_p90(hist):
    count = int(hist.sum())
    if not count:
        raise ValueError('Brightness reference mask is empty')
    position = .9 * (count-1)
    cdf = hist.cumsum()
    a = np.searchsorted(cdf, math.floor(position), side='right')
    b = np.searchsorted(cdf, math.ceil(position), side='right')
    return float(a + (b-a)*(position-math.floor(position)))


def uint8_percentile90(values):
    """Exact P90 without sorting individual uint8 samples."""
    return _histogram_p90(np.bincount(values.ravel(),minlength=256))


def brightness_reference(gray, mask, timings=None):
    """Exact linearly interpolated P90 of the masked uint8 image using its histogram."""
    start = time.perf_counter()
    with CV_LOCK:
        if timings is not None:
            timings['cv_lock_wait_ms_total'] = timings.get('cv_lock_wait_ms_total',0)+(time.perf_counter()-start)*1000
        hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel()
    return _histogram_p90(hist)


class GroupTracker:
    def __init__(self, params=None):
        self.params = params or FastParams()
        self.tracks = {}
        self.next_id = 1
        self.last_time = None
        self.last_frame = None
        self.observed_distance = self.gap_displacement = 0.
        self.last_lock_wait_ms = 0.

    def reset(self):
        self.tracks.clear()
        self.last_time = None
        self.last_frame = None

    def update(self, signal, mask, t, frame_index=None):
        p = self.params
        if self.last_time is not None and t <= self.last_time:
            raise ValueError('Tracking timestamps must increase')
        dt = 0 if self.last_time is None else t-self.last_time
        if frame_index is not None and self.last_frame is not None:
            if frame_index <= self.last_frame: raise ValueError('Frame indices must increase')
            if frame_index > self.last_frame+1:
                for track in self.tracks.values():
                    if track['state']=='measured': track['state']='missing'
                    track['path']=[]
        self.last_frame = frame_index
        self.last_time = t
        self.tracks = {k:v for k,v in self.tracks.items()
                       if t-v['seen'] <= p.max_gap_s and
                       (v['state'] != 'group' or t-v['measured_at'] <= p.max_group_s)}
        binary = np.ascontiguousarray((signal > p.threshold) & mask, dtype=np.uint8)
        start = time.perf_counter()
        with CV_LOCK:
            self.last_lock_wait_ms = (time.perf_counter()-start)*1000
            _, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
        ids = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= p.min_area)
        ids = ids[ids != 0]
        groups = {int(i): [] for i in ids}
        keys = list(self.tracks)
        if keys and len(ids):
            positions = np.array([self.tracks[k]['pos'] for k in keys])
            distances = np.sum((positions[:,None,:]-centers[ids][None,:,:])**2, axis=2)
            nearest = distances.argmin(axis=1)
            for row,key in enumerate(keys):
                x,y = np.rint(positions[row]).astype(int)
                inside = int(labels[y,x]) if 0<=y<labels.shape[0] and 0<=x<labels.shape[1] else 0
                if inside in groups:
                    groups[inside].append(key)
                elif distances[row,nearest[row]] <= (p.max_speed*dt)**2:
                    groups[int(ids[nearest[row]])].append(key)
        measured, unresolved = [], []
        used = set()
        for blob in ids:
            blob = int(blob)
            members = groups[blob]
            area = int(stats[blob, cv2.CC_STAT_AREA])
            pos = centers[blob].copy()
            if len(members)>1 or area>p.max_single_area:
                # A large blob present at startup has unknown multiplicity, not "one fly".
                unresolved.append(dict(x=float(pos[0]), y=float(pos[1]),
                                       count=len(members) if members else None, ids=members))
                for key in members:
                    tr = self.tracks[key]
                    tr.update(seen=t, state='group')
                    tr['path'] = []  # never draw an inferred trajectory through the group
                    used.add(key)
                continue
            key = members[0] if members else self.next_id
            if not members:
                self.next_id += 1
                self.tracks[key] = dict(pos=pos.copy(), measured_pos=pos.copy(), measured_at=t,
                                        seen=t, state='new', path=[])
            tr = self.tracks[key]
            distance = float(np.linalg.norm(pos-tr['measured_pos']))
            if tr['state'] == 'measured':
                self.observed_distance += distance
            elif tr['state'] != 'new':
                self.gap_displacement += distance
                tr['path'] = []
            tr.update(pos=pos, measured_pos=pos.copy(), measured_at=t, seen=t, state='measured')
            tr['path'] = (tr['path']+[tuple(pos)])[-30:]
            measured.append(dict(id=key,x=float(pos[0]),y=float(pos[1]),area=area))
            used.add(key)
        for key,tr in self.tracks.items():
            if key not in used:
                tr['state']='missing'
                tr['path']=[]
        return dict(measured=measured, groups=unresolved, binary=binary,
                    observed_distance_px=self.observed_distance,
                    gap_displacement_px=self.gap_displacement,
                    paths=[list(v['path']) for v in self.tracks.values() if len(v['path'])>1])
