"""The behavioural rows, re-binnable into per-vial timeseries. Pure: no Qt, no painting.

WHY THE STORE IS SEPARATE FROM THE PLOT. What the operator sets -- which parameter, what bin width,
cumulative or not -- is arithmetic on a table, and arithmetic that decides what a graph claims
should be testable without a screen. A widget that binned its own points inside `paintEvent` could
only ever be checked by looking at it.

=================================================================================================
RE-BINNING IS NOT RE-MEASURING, and the distinction is the whole reason `bin_seconds` here is a
DISPLAY control and not a measurement one.

`behaviour.csv` has one row per vial per DWELL -- roughly one every two seconds, because that is
how long the drum holds still. Choosing a 60 s bin here does not re-measure anything: it groups
rows that already exist and takes the median of each group. The underlying rows are untouched and
still on disk, so a bin width chosen for readability can never change the data.

THE AGGREGATOR IS THE MEDIAN, and it is chosen for the same reason `median_path_length` is: this
rig runs up to ~20 flies per vial, so a merged blob or a shredded track puts outliers in every
window, and a mean would follow them. `nan` rows are DROPPED from a bin rather than treated as
zero -- an empty vial has no height, which is not the same claim as "its flies are at the bottom".
A bin with nothing measurable in it yields no point at all, so a gap in the line is a gap in the
data rather than a dip toward zero.

CUMULATIVE MEANS RUNNING SUM OF THE BINNED VALUES. For a rate-like parameter (path length per
dwell) that is a genuine total-distance-so-far curve. For a level-like one (mean height) it is
not meaningful, and nothing here pretends otherwise -- the control is offered because the operator
asked for it and knows which parameters it suits.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: Parameters offered for plotting, in the order they are shown, as ``(field, label)``.
#: `median_path_length` FIRST because it is the rig owner's default: with up to 20 flies per vial
#: the mean is dominated by merges, so the median fragment length is the readable one.
#: ACTIVITY IS IN THE SAME LIST AS THE TRACKING PARAMETERS, from the operator's point of view
#: there is no reason for "how much did this vial move" to live in a different surface from "how
#: far did its flies walk" -- they are both per-vial numbers over the same run. They arrive by
#: different routes (activity per BIN from the accumulator, behaviour per DWELL from the tracker)
#: and `BehaviourSeries` does not care: a row is a row with an elapsed time, a face, a vial and
#: some fields.
PLOTTABLE = (
    ("median_path_length", "median track length (px)"),
    ("motion_px_sum", "activity: motion (px)"),
    ("active_fraction_mean", "activity: active fraction"),
    ("total_path_length", "total path length (px)"),
    ("median_speed", "median speed (px/s)"),
    ("mean_speed", "mean speed (px/s)"),
    ("p90_speed", "90th pct speed (px/s)"),
    ("frac_above_mid", "climbing score (fraction above mid)"),
    ("mean_height", "mean height (0 bottom, 1 top)"),
    ("median_height", "median height"),
    ("max_height", "max height"),
    ("est_n_flies_mean", "estimated flies per frame"),
    ("n_blobs_mean", "blobs per frame"),
    ("mean_fragment_frames", "mean fragment length (frames)"),
    ("n_tracks", "track fragments"),
)

PLOT_LABELS = dict(PLOTTABLE)

#: The fields that SCALE WITH ROI AREA -- the "extensive" ones, where a bigger hand-drawn ROI
#: inflates the number for the very same flies: a raw moving-pixel sum, a sum over every fly, and
#: counts. Dividing these by the vial's lit area (see `_area_factor`) makes vials whose ROIs were
#: drawn to different sizes comparable. Everything NOT listed here is left alone ON PURPOSE:
#: `active_fraction_mean` is ALREADY motion/area; speeds, heights and `median_path_length` are
#: per-fly or already a 0..1 fraction of the vial, so they do not depend on how big the box was
#: drawn and dividing THEM by area would only manufacture an uninterpretable quantity (a speed per
#: px^2, a height-fraction per px^2). The rig owner asked for "area-dependent ones only".
AREA_NORMALIZED_FIELDS = frozenset({
    "motion_px_sum",       # activity.csv: summed moving-pixel count over the bin
    "total_path_length",   # behaviour.csv: summed over every fly's track
    "n_blobs_mean",        # behaviour.csv: mean blob count per frame (more area -> more blobs)
    "est_n_flies_mean",    # behaviour.csv: mean estimated flies per frame
    "n_tracks",            # behaviour.csv: number of track fragments in the dwell
})

#: Bin widths offered for the DISPLAY, in seconds. 0 = "raw": every recorded row is its own point,
#: nothing grouped. The sub-second choices exist to watch a fast transient -- e.g. the ~7 s post-flip
#: startle -- but they only reveal detail the DATA already has: re-binning groups the rows that were
#: written, it cannot invent finer ones than the RECORDING bin (`binning.bin_seconds`) produced. To
#: see 200 ms structure the run must have been RECORDED at <= 200 ms, not just displayed at it.
BIN_CHOICES = (0, 0.2, 0.5, 1, 10, 30, 60, 300, 900)

VIALS_PER_FACE = 16
FACES = ("A", "B")


def _is_number(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    return ordered[middle] if n % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


class BehaviourSeries:
    """Accumulates behaviour rows and serves per-vial timeseries on demand.

    Rows are kept as they arrived; every display choice is applied at `series()` time, so changing
    the bin width or the parameter re-reads what is already held rather than needing the run to be
    repeated.
    """

    def __init__(self, max_rows: int = 200_000) -> None:
        #: ``(elapsed_s, face, vial_id, {field: value})`` per row.
        self._rows: List[Tuple[float, str, int, dict]] = []
        #: A 3-day run at ~2 s dwells over 32 vials is ~4 million rows; the file holds them all and
        #: this is for watching, so the oldest are dropped once the cap is reached.
        self._max_rows = int(max_rows)
        self.dropped_rows = 0
        #: ``(face, local_index) -> lit_area_px``, harvested from whatever rows carry `lit_area_px`
        #: (the activity rows do; the tracking rows do not). The lit area of a vial is FIXED for a
        #: run -- it is the ROI ∩ illumination mask -- so one value per vial is enough to normalise
        #: even the tracking metrics of the SAME vial. See `_area_factor`.
        self._vial_area: Dict[Tuple[str, int], int] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows = []
        self.dropped_rows = 0
        self._vial_area = {}

    def add(self, rows: Sequence[dict]) -> int:
        """Adopt behaviour rows. Returns how many were usable."""
        added = 0
        for row in rows or ():
            try:
                elapsed = float(row["elapsed_s"])
                face = str(row["face"])
                vial = int(row["vial_id"])
            except (KeyError, TypeError, ValueError):
                continue
            self._rows.append((elapsed, face, vial, dict(row)))
            # HARVEST THE VIAL'S LIT AREA when the row carries it (activity rows do). Constant per
            # vial, so the latest wins and one value serves the whole run -- including for this
            # vial's tracking metrics, whose own rows have no area of their own.
            area = row.get("lit_area_px")
            if area is not None:
                try:
                    area_px = int(area)
                except (TypeError, ValueError):
                    area_px = 0
                if area_px > 0:
                    self._vial_area[(face, self.local_index(face, vial))] = area_px
            added += 1
        overflow = len(self._rows) - self._max_rows
        if overflow > 0:
            del self._rows[:overflow]
            self.dropped_rows += overflow
        return added

    # -- queries ---------------------------------------------------------------------------------
    def local_index(self, face: str, vial_id: int) -> int:
        """0-based position of a vial WITHIN its face. Global ids are ``face_index*16 + local``."""
        return (int(vial_id) - 1) % VIALS_PER_FACE

    def area_reference(self) -> Optional[float]:
        """The MEDIAN per-vial lit area -- the yardstick every vial is rescaled to, or None if no
        area is known yet. Median, not mean, for the same reason the plotted points are medians:
        one enormous or tiny hand-drawn ROI must not drag the reference the other 31 are measured
        against."""
        areas = [a for a in self._vial_area.values() if a > 0]
        return _median(areas) if areas else None

    def _area_factor(self, face: str, vial_index: int) -> float:
        """``reference_area / this vial's lit area`` -- the constant a vial's EXTENSIVE metrics are
        multiplied by so it reads as if it had the median ROI. 1.0 when either area is unknown (no
        harm: the value is left exactly as recorded). Constant across the run, so multiplying every
        point by it is the same as re-measuring the vial at the median area, and the median vial is
        unchanged."""
        area = self._vial_area.get((face, vial_index))
        reference = self.area_reference()
        if not area or not reference:
            return 1.0
        return reference / float(area)

    def series(self, field: str, face: str, vial_index: int, *, bin_seconds: float = 10.0,
               cumulative: bool = False, normalize_area: bool = False) -> List[Tuple[float, float]]:
        """``[(elapsed_s, value), ...]`` for one vial, binned and optionally accumulated.

        Points are the MEDIAN of the rows in each bin, and rows whose value is `nan` are dropped
        rather than counted as zero -- so a bin with nothing measurable in it produces NO POINT,
        and a gap in the line reads as a gap in the data instead of a dip to the floor.

        `normalize_area` (only for `field in AREA_NORMALIZED_FIELDS`) rescales every point by
        `median_ROI_area / this vial's lit area`, so vials whose ROIs were hand-drawn to different
        sizes are comparable. It is a pure display multiplier -- the stored rows are untouched, and
        a metric that does not scale with area (a speed, a height, a per-fly figure) is never
        touched even when the flag is on.
        """
        width = max(1e-6, float(bin_seconds))
        buckets: Dict[int, List[float]] = {}
        for elapsed, row_face, vial, row in self._rows:
            if row_face != face or self.local_index(row_face, vial) != vial_index:
                continue
            value = row.get(field)
            if not _is_number(value):
                continue
            # `math.floor(elapsed / width + eps)`, NOT `elapsed // width`: floor division on floats
            # puts a value that is an exact multiple of the width into the WRONG bucket, because the
            # multiple is not exact in binary (0.4 / 0.2 == 2.0 but 0.4 // 0.2 == 1.0). At coarse bins
            # that never showed; at a 0.2 s display bin on 0.2 s data it silently merged every other
            # point. The epsilon is a fraction of one bin, far below any real spacing.
            bucket = int(math.floor(elapsed / width + 1e-9))
            buckets.setdefault(bucket, []).append(float(value))
        if not buckets:
            return []
        points = [((index + 0.5) * width, _median(values))
                  for index, values in sorted(buckets.items())]
        # AREA NORMALISATION is a constant per-vial factor, so median(k*v) == k*median(v): applying
        # it to the binned medians here is identical to scaling every raw row, and cheaper. Only the
        # extensive fields are eligible; everything else is returned exactly as measured.
        if normalize_area and field in AREA_NORMALIZED_FIELDS:
            factor = self._area_factor(face, vial_index)
            if factor != 1.0:
                points = [(t, value * factor) for t, value in points]
        if not cumulative:
            return points
        total = 0.0
        out = []
        for t, value in points:
            total += value
            out.append((t, total))
        return out

    def face_series(self, field: str, face: str, **kwargs) -> Dict[int, List[Tuple[float, float]]]:
        """`{vial_index: points}` for one face's sixteen vials."""
        return {index: self.series(field, face, index, **kwargs)
                for index in range(VIALS_PER_FACE)}

    def value_range(self, field: str, **kwargs) -> Optional[Tuple[float, float]]:
        """`(low, high)` across BOTH faces, or None if nothing is plottable.

        ONE RANGE FOR ALL 32 CELLS, deliberately. Per-cell autoscaling would draw an empty vial's
        noise with the same amplitude as a busy vial's real signal, and the grid exists precisely
        so vials can be compared with each other at a glance.
        """
        low = high = None
        for face in FACES:
            for index in range(VIALS_PER_FACE):
                for _t, value in self.series(field, face, index, **kwargs):
                    low = value if low is None else min(low, value)
                    high = value if high is None else max(high, value)
        if low is None:
            return None
        if high - low < 1e-9:
            # A flat series still needs a drawable band, or every point lands on one edge.
            return (low - 0.5, high + 0.5)
        return (low, high)

    def time_range(self) -> Optional[Tuple[float, float]]:
        if not self._rows:
            return None
        times = [row[0] for row in self._rows]
        return (min(times), max(times))
