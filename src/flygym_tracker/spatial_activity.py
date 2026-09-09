"""Face-separated, registered pixel statistics; owned by the pipeline thread."""
import numpy as np


class SpatialActivity:
    def __init__(self, background_window_s=120):
        self.faces = {}
        self.background_window_s = background_window_s
        self.last_publish = 0.0
        self.last_motion = None

    def add(self, face, gray, motion, mask=None, offset=(0, 0), normalized=None, elapsed_s=None):
        """Sum stationary frames; motion=None means a reference frame with no frame pair.

        Offsets are the same integer translation used for the vial ROIs. Move samples
        back onto the first frame of this face without interpolation or edge wrapping.
        """
        if face not in ('A', 'B'):
            return
        data = self.faces.get(face)
        if data is None or data['counts'].shape != gray.shape:
            if data is not None:
                data['rolling'].close()
            background = gray.copy()
            background.setflags(write=False)
            data = dict(background=background, counts=np.zeros(gray.shape, np.uint64),
                        intensity_sum=np.zeros(gray.shape, np.uint64),
                        samples=np.zeros(gray.shape, np.uint64),
                        pair_samples=np.zeros(gray.shape, np.uint64),
                        normal_sum=np.zeros(gray.shape, np.float64), offset=tuple(offset), mask=mask,
                        roi=np.flatnonzero(np.ones(gray.shape, bool) if mask is None else mask),
                        frames=0, image_frames=0)
            from flygym_tracker.rolling_background import RollingBackground
            data['rolling'] = RollingBackground(len(data['roi']), self.background_window_s)
            self.faces[face] = data
        self.last_motion = motion  # Live inspector remains in current camera coordinates.
        dx, dy = (int(offset[i] - data['offset'][i]) for i in (0, 1))
        if mask is not None and not mask.flags.writeable and mask is data['mask'] and dx == 0 and dy == 0:
            roi = data['roi']
            values = gray.ravel()[roi]
            normal_values = values.astype(np.float64)/255 if normalized is None else normalized.ravel()[roi]
            data['intensity_sum'].ravel()[roi] += values
            data['samples'].ravel()[roi] += 1
            data['normal_sum'].ravel()[roi] += normal_values
            data['image_frames'] += 1
            data['rolling'].add(normal_values, data['image_frames']/20 if elapsed_s is None else elapsed_s)
            if motion is not None:
                data['counts'].ravel()[roi] += motion.ravel()[roi]
                data['pair_samples'].ravel()[roi] += 1
                data['frames'] += 1
            return
        height, width = gray.shape
        x0, x1 = max(0, -dx), min(width, width-dx)
        y0, y1 = max(0, -dy), min(height, height-dy)
        if x1 <= x0 or y1 <= y0:
            return
        target = np.s_[y0:y1, x0:x1]
        source = np.s_[y0+dy:y1+dy, x0+dx:x1+dx]
        valid = np.ones((y1-y0, x1-x0), bool) if mask is None else mask[source]
        values = gray[source]
        # Keep exact integer sums; normalize only published copies, never the history.
        np.add(data['intensity_sum'][target], values, out=data['intensity_sum'][target], where=valid)
        data['samples'][target] += valid
        normal_values = values.astype(np.float64)/255 if normalized is None else normalized[source]
        np.add(data['normal_sum'][target], normal_values, out=data['normal_sum'][target], where=valid)
        data['image_frames'] += 1
        aligned = np.full(gray.shape, np.nan, np.float32)
        np.copyto(aligned[target], normal_values, where=valid, casting='unsafe')
        data['rolling'].add(aligned.ravel()[data['roi']],
                            data['image_frames']/20 if elapsed_s is None else elapsed_s)
        if motion is not None:
            data['counts'][target] += motion[source] & valid
            data['pair_samples'][target] += valid
            data['frames'] += 1

    def snapshot(self, elapsed_s, force=False):
        if not self.faces or (not force and elapsed_s - self.last_publish < 10.0):
            return None
        self.last_publish = elapsed_s
        faces = {}
        for face, data in self.faces.items():
            valid = data['samples'] > 0
            mean = np.divide(data['intensity_sum'], data['samples'],
                             out=np.zeros(valid.shape, np.float64), where=valid)
            normal_mean = np.divide(data['normal_sum'], data['samples'],
                                    out=np.zeros_like(mean), where=valid)
            rate = np.divide(data['counts'], data['pair_samples'],
                             out=np.zeros_like(mean), where=data['pair_samples'] > 0)
            occupancy = np.zeros_like(mean)
            occupancy.ravel()[data['roi']] = data['rolling'].mean()
            published = dict(background=data['background'], counts=data['counts'].copy(),
                             frames=data['frames'], image_frames=data['image_frames'],
                             mean=mean, normal_mean=normal_mean, valid=valid, rate=rate,
                             occupancy=occupancy, occupancy_frames=data['rolling'].measured_frames,
                             background_samples=len(data['rolling'].samples),
                             background_learning=data['rolling'].background is None)
            for value in published.values():
                if isinstance(value, np.ndarray):
                    value.setflags(write=False)
            faces[face] = published
        return dict(elapsed_s=elapsed_s, faces=faces, background_window_s=self.background_window_s,
                    maximum=max(int(d['counts'].max()) for d in faces.values()),
                    rate_maximum=max(float(d['rate'].max()) for d in faces.values()))


    def set_background_window(self, seconds):
        if not np.isfinite(seconds) or not 10 <= seconds <= 600:
            raise ValueError('Background window must be 10 to 600 seconds')
        if self.background_window_s == float(seconds):
            return
        self.background_window_s = float(seconds)
        for data in self.faces.values():
            data['rolling'].set_window(seconds)
        self.last_publish = -np.inf

    def close(self):
        for data in self.faces.values():
            data['rolling'].close()


def relative_occupancy(data):
    return data['occupancy']
