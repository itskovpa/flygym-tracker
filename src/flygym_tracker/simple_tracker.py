"""Small silhouette/centroid tracker for tuning and throughput experiments.

Track numbers label fragments, not persistent animal identities. Ambiguous merges and
crossings can change assignments. Gap displacement is never added to observed path length.
"""
from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from flygym_tracker.cv_setup import CV_LOCK


@dataclass(frozen=True)
class SimpleTrackerParams:
    threshold: int = 140
    min_area: int = 8
    max_area: int = 300
    max_speed: float = 200.0
    max_gap_s: float = .25

    def __post_init__(self):
        if not 0 <= self.threshold <= 255 or not 1 <= self.min_area <= self.max_area:
            raise ValueError('Require threshold 0..255 and 1 <= minimum area <= maximum area')
        if not math.isfinite(self.max_speed) or not math.isfinite(self.max_gap_s) or self.max_speed <= 0 or self.max_gap_s < 0:
            raise ValueError('Speed must be positive; gap time must be nonnegative and finite')


def detect_centroids(gray, mask, params):
    """Dark pixels <= threshold, then connected component area filtering. No background model."""
    binary = np.ascontiguousarray((gray <= params.threshold) & mask, dtype=np.uint8)
    with CV_LOCK:
        _, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    accepted = (sizes >= params.min_area) & (sizes <= params.max_area)
    blobs = [(float(x), float(y), int(a)) for (x,y),a in zip(centers[1:][accepted], sizes[accepted])]
    return blobs, binary, labels, np.flatnonzero(accepted) + 1


@dataclass
class Fragment:
    number: int
    x: float
    y: float
    time: float
    frame: int
    points: deque = field(default_factory=lambda: deque(maxlen=200))


class CentroidTracker:
    def __init__(self, params=None):
        self.params = params or SimpleTrackerParams()
        self.active = {}
        self.next_id = 1
        self.observed_distance = 0.0
        self.observed_time = 0.0
        self.gap_displacement = 0.0
        self.gap_links = 0
        self.fragments = 0
        self.last_frame = None
        self.last_time = None

    def reset(self):
        """End links at rotation; keep session statistics and unique fragment numbers."""
        self.active.clear()
        self.last_frame = self.last_time = None

    def update(self, blobs, elapsed_s, frame_index):
        if self.last_frame is not None and (frame_index <= self.last_frame or elapsed_s < self.last_time):
            raise ValueError('Frames must advance; timestamps must not decrease')
        # A gap tolerance of zero still permits matching adjacent frames.
        for key, track in list(self.active.items()):
            if frame_index > track.frame + 1 and elapsed_s-track.time > self.params.max_gap_s:
                del self.active[key]
        candidates = []
        keys = list(self.active)
        centers = np.asarray([(x,y) for x,y,_ in blobs],dtype=float).reshape(-1,2)
        # Chunked vector distances avoid a Python loop for every candidate pair,
        # while bounding temporary memory if a low threshold generates many blobs.
        for start in range(0,len(keys),64):
            chunk = keys[start:start+64]
            tracks = [self.active[key] for key in chunk]
            xy = np.asarray([(track.x,track.y) for track in tracks])
            delta = xy[:,None,:]-centers[None,:,:]
            distance = np.hypot(delta[:,:,0],delta[:,:,1])
            gates = np.asarray([self.params.max_speed*(elapsed_s-track.time) for track in tracks])
            ti,bi = np.nonzero(distance <= gates[:,None])
            candidates.extend((float(distance[a,b]),chunk[a],int(b)) for a,b in zip(ti,bi))
        used_tracks, used_blobs, result = set(), set(), []
        for distance,key,i in sorted(candidates):
            if key in used_tracks or i in used_blobs:
                continue
            used_tracks.add(key)
            used_blobs.add(i)
            track = self.active[key]
            x,y,area = blobs[i]
            gap = frame_index > track.frame + 1
            dt = elapsed_s-track.time
            if gap:
                self.gap_displacement += distance
                self.gap_links += 1
            else:
                self.observed_distance += distance
                self.observed_time += dt
            track.x,track.y,track.time,track.frame = x,y,elapsed_s,frame_index
            track.points.append((x,y,gap))
            result.append(dict(fragment=key,x=x,y=y,area=area,gap=gap,distance=distance))
        for i,(x,y,area) in enumerate(blobs):
            if i not in used_blobs:
                track = Fragment(self.next_id,x,y,elapsed_s,frame_index)
                track.points.append((x,y,False))
                self.active[self.next_id] = track
                result.append(dict(fragment=self.next_id,x=x,y=y,area=area,gap=False,distance=0.0))
                self.next_id += 1
                self.fragments += 1
        self.last_frame,self.last_time = frame_index,elapsed_s
        return result

    def statistics(self):
        return dict(fragments=self.fragments, observed_distance_px=self.observed_distance,
                    gap_displacement_px=self.gap_displacement, gap_links=self.gap_links,
                    mean_link_speed_px_s=(self.observed_distance/self.observed_time if self.observed_time else 0.0))
