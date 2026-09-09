"""Cumulative, full-resolution pixel activity; owned by the pipeline thread."""
import numpy as np


class SpatialActivity:
    def __init__(self):
        self.faces = {}
        self.last_publish = 0.0
        self.last_motion = None

    def add(self, face, gray, motion):
        if face not in ('A', 'B'):
            return
        data = self.faces.get(face)
        if data is None or data['counts'].shape != gray.shape:
            background = gray.copy()
            background.setflags(write=False)
            data = dict(background=background, counts=np.zeros(gray.shape, dtype=np.uint64), frames=0)
            self.faces[face] = data
        self.last_motion = motion
        data['counts'] += motion
        data['frames'] += 1

    def snapshot(self, elapsed_s, force=False):
        if not self.faces or (not force and elapsed_s - self.last_publish < 10.0):
            return None
        self.last_publish = elapsed_s
        faces = {}
        for face, data in self.faces.items():
            counts = data['counts'].copy()
            counts.setflags(write=False)
            faces[face] = dict(background=data['background'], counts=counts, frames=data['frames'])
        return dict(elapsed_s=elapsed_s, faces=faces,
                    maximum=max(int(d['counts'].max()) for d in faces.values()))
