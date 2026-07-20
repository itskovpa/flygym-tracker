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

#: Bin widths offered, in seconds. The smallest is below a dwell (~2 s), so "no binning" is
#: available: every dwell keeps its own point.
BIN_CHOICES = (1, 10, 30, 60, 300, 900)

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

    def __len__(self) -> int:
        return len(self._rows)

    def clear(self) -> None:
        self._rows = []
        self.dropped_rows = 0

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

    def series(self, field: str, face: str, vial_index: int, *, bin_seconds: float = 10.0,
               cumulative: bool = False) -> List[Tuple[float, float]]:
        """``[(elapsed_s, value), ...]`` for one vial, binned and optionally accumulated.

        Points are the MEDIAN of the rows in each bin, and rows whose value is `nan` are dropped
        rather than counted as zero -- so a bin with nothing measurable in it produces NO POINT,
        and a gap in the line reads as a gap in the data instead of a dip to the floor.
        """
        width = max(1e-6, float(bin_seconds))
        buckets: Dict[int, List[float]] = {}
        for elapsed, row_face, vial, row in self._rows:
            if row_face != face or self.local_index(row_face, vial) != vial_index:
                continue
            value = row.get(field)
            if not _is_number(value):
                continue
            buckets.setdefault(int(elapsed // width), []).append(float(value))
        if not buckets:
            return []
        points = [((index + 0.5) * width, _median(values))
                  for index, values in sorted(buckets.items())]
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
