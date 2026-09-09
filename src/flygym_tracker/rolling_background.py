"""Bounded rolling background estimation, separate from cumulative occupancy."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def bright_percentile(samples):
    # NumPy nanpercentile dispatches separately for each pixel column, which is
    # prohibitively slow on full frames. Sort each short (<=32) sample row in C.
    values = np.ascontiguousarray(np.stack(samples).T)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    values[~finite] = np.inf
    values.sort(axis=1)
    position = .9 * np.maximum(counts-1, 0)
    low = position.astype(np.intp)
    high = np.ceil(position).astype(np.intp)
    rows = np.arange(len(values))
    a, b = values[rows, low].astype(np.float64), values[rows, high].astype(np.float64)
    a[counts == 0] = b[counts == 0] = 0
    result = a + (b-a)*(position-low)
    result[counts == 0] = np.nan
    return result


class RollingBackground:
    def __init__(self, size, window_s=120, executor=None):
        self.window_s = float(window_s)
        self.samples = deque(maxlen=32)
        self.last_sample = -np.inf
        self.last_submit = -np.inf
        self.background = None
        self.background_time = -np.inf
        self.future = None
        self.future_time = -np.inf
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='flygym-background')
        self.owns_executor = executor is None
        self.total = np.zeros(size, np.float64)
        self.count = np.zeros(size, np.uint64)
        self.measured_frames = 0

    def set_window(self, seconds):
        if not np.isfinite(seconds) or not 10 <= seconds <= 600:
            raise ValueError('Background window must be 10 to 600 seconds')
        if self.window_s == float(seconds):
            return
        self.window_s = float(seconds)
        self.samples.clear()
        self.last_sample = self.last_submit = -np.inf
        self.background = None
        self.background_time = -np.inf
        # Old work may finish but must never install an estimate for the previous window.
        if self.future is not None:
            self.future.cancel()
        self.future = None
        # Previously corrected occupancy remains intact; only the background relearns.

    def add(self, values, elapsed_s):
        now = float(elapsed_s)
        while self.samples and self.samples[0][0] < now-self.window_s:
            self.samples.popleft()
        if self.background_time < now-self.window_s:
            self.background = None
        if self.future is not None and self.future.done():
            background = self.future.result()
            self.future = None
            if self.future_time >= now-self.window_s:
                self.background = background
                self.background_time = self.future_time
        # Measure against the previous estimate, before learning from this frame.
        if self.background is not None:
            valid = np.isfinite(values) & np.isfinite(self.background) & (self.background > 0)
            contrast = np.divide(self.background-values, self.background,
                                 out=np.zeros_like(self.background), where=valid)
            self.total += np.clip(contrast, 0, 1)
            self.count += valid
            self.measured_frames += int(np.any(valid))
        if now-self.last_sample >= self.window_s/32:
            sample = values.copy();sample.setflags(write=False)
            self.samples.append((now, sample));self.last_sample = now
        if len(self.samples) >= 4 and self.future is None and now-self.last_submit >= max(1, self.window_s/16):
            self.future_time = now
            self.future = self.executor.submit(bright_percentile, tuple(v for _,v in self.samples))
            self.last_submit = now

    def mean(self):
        return np.divide(self.total, self.count, out=np.zeros_like(self.total), where=self.count > 0)

    def close(self):
        if self.owns_executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
