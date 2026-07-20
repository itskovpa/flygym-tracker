"""Resumable CSV/XLSX logging for per-vial activity and run events (DESIGN.md §7).

CSV is the durable source of truth for every run, **regardless of `fmt`**: activity rows and
event rows are appended to plain CSV files the instant `log_activity`/`log_event` is called (each
call opens, writes, and closes the file — no long-lived handle, so a crash loses at most the
in-flight call). XLSX is a derived, human-friendly view. Appending to `.xlsx` row-by-row is
awkward — it is a zipped OOXML document, not a flat file with a true append mode — so instead each
`.xlsx` sibling is fully **regenerated from its CSV** on `flush()`/`close()`, limited to whichever
CSV files this instance actually wrote to since the last flush (only when `fmt` is `"xlsx"` or
`"both"`). CSV and XLSX therefore always agree by construction: the xlsx is nothing more than the
csv re-encoded through pandas/openpyxl.

Daily rolling (`activity_YYYYMMDD.csv`) is keyed off each record's own `bin_start_iso` — not
wall-clock time — so a batch is routed to the correct day file even if it is logged late, out of
order, or straddles a midnight boundary. `events.csv` does not roll; it is one file for the whole
run (DESIGN.md §7 lists no rolling scheme for events, unlike the explicit one for activity).

Resumability: re-opening an `ActivityLogger` against a directory that already has data appends to
the existing CSVs in `"a"` mode (no header duplication, no truncation of existing rows — append
mode cannot clobber what's already on disk). The logger does **not** de-duplicate rows against
what is already on disk; callers are expected to call `last_bin()` right after construction and
only submit records for bins after that point (see `last_bin` docstring).

Known xlsx limitation (inherent to the OOXML format, not fixable from here): xlsx has no distinct
int/float storage type. openpyxl serializes a whole-number float like `60.0` as `<v>60</v>` (no
decimal point), so a numeric column that is *entirely* whole numbers for a given day (e.g.
`elapsed_s` when every bin lands on an exact second) reads back from the `.xlsx` as int64 instead
of float64 — the value is exactly preserved (`60 == 60.0`), only the dtype label can differ.
Anything that needs the declared dtypes (`ActivityRecord`'s `float` vs `int` fields) exactly should
read the CSV, which is why it remains the source of truth.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from flygym_tracker.types import (
    ACTIVITY_COLUMNS,
    BEHAVIOUR_COLUMNS,
    EVENT_COLUMNS,
    ActivityRecord,
    EventRecord,
)

_VALID_FMT = {"csv", "xlsx", "both"}
_VALID_ROLLING = {"daily"}  # only mode implemented (DESIGN.md §7); kept as a parameter for
                            # forward compatibility so callers don't need to change call sites
                            # later.

# Columns that must never be type-inferred from their text when rebuilding the xlsx — a run_id,
# timestamp, or face label that happens to look numeric must not silently become an int/float.
_ACTIVITY_STR_COLUMNS = ["run_id", "bin_start_iso", "bin_end_iso", "face"]
_EVENT_STR_COLUMNS = ["run_id", "iso_time", "event", "detail"]
#: Behaviour columns that must stay text: a run_id or a face label that happens to look numeric
#: must not silently become an int.
_BEHAVIOUR_STR_COLUMNS = ["run_id", "iso_time", "face"]

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


class ActivityLogger:
    """Appends `ActivityRecord`/`EventRecord` rows to resumable, daily-rolling CSV (+ optional XLSX).

    Public API: `log_activity`, `log_event`, `last_bin`, `flush`, `close` — also usable as a
    context manager (`with ActivityLogger(...) as logger: ...`, which calls `close()` on exit).

    Parameters
    ----------
    output_dir:
        Directory for all output files; created (including parents) if missing.
    run_id:
        Written into every row and into `run_meta.json`; not otherwise interpreted.
    fmt:
        `"csv"`, `"xlsx"`, or `"both"`. CSV is always written internally as the source of truth
        regardless of this setting — `fmt` only controls whether a matching `.xlsx` is (re)built
        on `flush()`/`close()`.
    rolling:
        Only `"daily"` is implemented (DESIGN.md §7: "new file per day"); any other value raises.
    meta:
        Free-form dict stored verbatim under the `"meta"` key in `run_meta.json` (e.g. config
        snapshot, calibration hash, camera settings, versions) — opaque to the logger.
    """

    def __init__(
        self,
        output_dir,
        run_id: str,
        fmt: str = "both",
        rolling: str = "daily",
        meta: Optional[dict] = None,
    ) -> None:
        if fmt not in _VALID_FMT:
            raise ValueError(f"fmt must be one of {sorted(_VALID_FMT)}, got {fmt!r}")
        if rolling not in _VALID_ROLLING:
            raise ValueError(
                f"rolling must be one of {sorted(_VALID_ROLLING)}, got {rolling!r} "
                "(only 'daily' is implemented)"
            )

        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.fmt = fmt
        self.rolling = rolling

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._dirty_csvs: set = set()  # CSV Paths written since the last flush()
        self._closed = False

        #: EVERY FILE OF ONE RUN CARRIES THE RUN'S START TIME, as asked. Taken once, here, and
        #: never re-derived: naming each file when it happens to be written would scatter one
        #: run's outputs across the timeline and give the workbook a different stamp from the CSVs.
        #: A directory of several runs then sorts and groups by eye.
        started = datetime.now()
        self.started_iso = started.isoformat()
        self.stamp = started.strftime("%Y%m%d-%H%M%S")

        self._meta_path = self.output_dir / ("run_meta_%s.json" % self.stamp)
        self._run_meta = {
            "run_id": run_id,
            "start_iso": self.started_iso,
            "stop_iso": None,
            "meta": meta or {},
        }
        self._save_run_meta()

    # -- public API ----------------------------------------------------------------------------
    def update_meta(self, patch: dict) -> None:
        """Deep-merge `patch` into ``run_meta.json``'s ``meta`` block and re-save immediately.

        Exists so the snapshot can record what the run REALLY used, not merely what was loaded
        from the config file. The settings panel can be opened before the first frame, so the
        config captured at construction is a statement of intent that the operator may then
        change; a run whose data was measured at ``pixel_threshold`` 30.0 while its own metadata
        says 12.0 misreports its provenance to whoever reads that folder later.
        """
        def merge(dst: dict, src: dict) -> None:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    merge(dst[key], value)
                else:
                    dst[key] = value

        merge(self._run_meta["meta"], patch)
        self._save_run_meta()


    def log_activity(self, records: list) -> None:
        """Append rows, routing each record to the day file matching its own `bin_start_iso`.

        A single batch may straddle a day boundary; each record is routed independently so it
        always lands in the correct `activity_YYYYMMDD.csv`. No-op for an empty/None list.
        """
        if not records:
            return
        by_date: dict = {}
        for rec in records:
            date_key = self._date_key(rec.bin_start_iso)
            by_date.setdefault(date_key, []).append(rec.as_row())
        for date_key, rows in by_date.items():
            path = self._activity_csv_path(date_key)
            self._append_rows(path, ACTIVITY_COLUMNS, rows)
            self._dirty_csvs.add(path)

    def log_behaviour(self, rows: list) -> None:
        """Append fly-level rows to `behaviour.csv`. No-op for an empty list.

        A SEPARATE FILE, not extra columns on activity.csv, and the reason is that they are
        different measurements at different scales. An activity row exists for every vial in every
        bin and needs nothing but the ROI. A behaviour row exists only where the drum was still
        long enough for the tracker to see individual flies -- so on a widened activity table most
        rows would carry fourteen empty columns, and every analysis script already reading that
        file would have to be updated to ignore them.

        Rows are plain dicts (`fly_tracking.summarize` output plus the keys that identify them), so
        nothing here has to know what the tracker measures -- adding a parameter there adds a
        column here, provided it is listed in `BEHAVIOUR_COLUMNS`.
        """
        if not rows:
            return
        path = self.behaviour_path()
        self._append_rows(path, BEHAVIOUR_COLUMNS, rows)
        self._dirty_csvs.add(path)

    def log_event(self, event: Optional[EventRecord]) -> None:
        """Append one row to `events.csv`. No-op if `event` is None."""
        if event is None:
            return
        path = self._events_csv_path()
        self._append_rows(path, EVENT_COLUMNS, [event.as_row()])
        self._dirty_csvs.add(path)

    def activity_path(self, date_key: Optional[str] = None) -> Path:
        """This run's activity file for a day. Public so callers and tests never spell the
        naming scheme themselves -- it has changed once and may change again."""
        return self._activity_csv_path(date_key or self._today_key())

    def events_path(self) -> Path:
        return self._events_csv_path()

    def behaviour_path(self) -> Path:
        return self.output_dir / ("behaviour_%s.csv" % self.stamp)

    def meta_path(self) -> Path:
        return self._meta_path

    def workbook_path(self) -> Path:
        """Where `matrix_export` writes this run's one-sheet-per-parameter workbook."""
        return self.output_dir / ("flygym_%s.xlsx" % self.stamp)

    def run_meta(self) -> dict:
        """The run metadata as written to run_meta_<stamp>.json. Kept, as asked, and handed to the
        workbook so a sheet of numbers carries its start time and config with it."""
        return dict(self._run_meta)

    def last_bin(self) -> Optional[str]:
        """Max `bin_start_iso` already recorded in *today's* activity file, or None.

        "Today" is the real wall-clock date — i.e. the file this logger would currently be
        appending new records to — not the newest date ever logged across the whole run. After a
        daily rollover the previous day's file is complete and not relevant to resuming: a fresh
        day starts with nothing recorded, so `last_bin()` correctly returns None right after
        midnight even though yesterday's file is full of rows.

        Intended use: right after constructing a logger against a directory that may already have
        data (process restart), call `last_bin()` once and only pass `log_activity` records for
        bins strictly after it — the logger itself does not de-duplicate.
        """
        # EVERY ACTIVITY FILE FOR TODAY, not just this logger's own. Filenames now carry the
        # RUN's start stamp, so a restarted process writes to a new file -- and reading only its
        # own would report "nothing recorded today" while the previous process's rows sat beside
        # it. Resuming has to look at what is on disk for the day, whoever wrote it.
        paths = sorted(self.output_dir.glob("activity_*_%s.csv" % self._today_key()))
        if not paths:
            return None
        best: Optional[str] = None
        rows_iter = []
        for path in paths:
            with path.open("r", newline="", encoding="utf-8") as f:
                rows_iter.extend(list(csv.DictReader(f)))
        for row in rows_iter:
                v = row.get("bin_start_iso")
                if v and (best is None or v > best):
                    best = v
        return best

    def flush(self) -> None:
        """Regenerate the `.xlsx` sibling of every CSV touched since the last flush.

        No-op when `fmt == "csv"`. Safe to call with nothing pending (e.g. after only empty-list
        calls to `log_activity`).
        """
        if self.fmt not in ("xlsx", "both"):
            self._dirty_csvs.clear()
            return
        for csv_path in list(self._dirty_csvs):
            self._rewrite_xlsx(csv_path)
        self._dirty_csvs.clear()

    def close(self) -> None:
        """Flush pending xlsx regeneration and stamp `stop_iso` in run_meta.json. Idempotent."""
        if self._closed:
            return
        self.flush()
        self._run_meta["stop_iso"] = datetime.now().isoformat()
        self._save_run_meta()
        self._closed = True

    def __enter__(self) -> "ActivityLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- internals -------------------------------------------------------------------------------

    @staticmethod
    def _date_key(iso_str: str) -> str:
        """`"2026-07-18T10:00:00" -> "20260718"` — used for both file routing and `last_bin()`."""
        if not iso_str or not _ISO_DATE_RE.match(iso_str):
            raise ValueError(f"bin_start_iso does not look like an ISO 8601 timestamp: {iso_str!r}")
        return iso_str[:10].replace("-", "")

    @staticmethod
    def _today_key() -> str:
        return datetime.now().strftime("%Y%m%d")

    def _activity_csv_path(self, date_key: str) -> Path:
        # BOTH STAMPS. The run stamp groups a run's files together; the date key is what makes the
        # daily rolling work, so a three-day run still splits into one file per day WITHIN its run.
        return self.output_dir / f"activity_{self.stamp}_{date_key}.csv"

    def _events_csv_path(self) -> Path:
        return self.output_dir / f"events_{self.stamp}.csv"

    @staticmethod
    def _append_rows(path: Path, columns: list, rows: list) -> None:
        if not rows:
            return
        write_header = not (path.exists() and path.stat().st_size > 0)
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _schema_for(csv_path: Path):
        """Which schema a CSV in this directory follows, BY PREFIX rather than by exact name.

        TWO BUGS THIS FIXES, both mine and both caught by the logger's own tests once the files
        were renamed:

        * it matched the literal ``"events.csv"``, so once events carried the run stamp
          (`events_20260720-202020.csv`) it fell through to the ACTIVITY schema and the xlsx
          rebuild raised `KeyError` for twelve columns an event row has never had;
        * `behaviour_*.csv` was added to the rebuild set when `log_behaviour` was written, with no
          schema of its own -- so it would have been re-encoded as an activity table too.

        Matching on the prefix is what makes the naming scheme and the schema move together.
        """
        name = csv_path.name
        if name.startswith("events"):
            return EVENT_COLUMNS, _EVENT_STR_COLUMNS
        if name.startswith("behaviour"):
            return BEHAVIOUR_COLUMNS, _BEHAVIOUR_STR_COLUMNS
        return ACTIVITY_COLUMNS, _ACTIVITY_STR_COLUMNS

    def _rewrite_xlsx(self, csv_path: Path) -> None:
        """Re-derive `<csv_path>.xlsx` from `csv_path` in full. See module docstring."""
        if not (csv_path.exists() and csv_path.stat().st_size > 0):
            return
        columns, str_columns = self._schema_for(csv_path)
        df = pd.read_csv(
            csv_path,
            dtype={c: str for c in str_columns},
            keep_default_na=False,  # don't turn "" (e.g. EventRecord.detail) into NaN
        )
        df = df[columns]  # defensive: enforce exact column order regardless of file contents
        xlsx_path = csv_path.with_suffix(".xlsx")
        df.to_excel(xlsx_path, index=False, engine="openpyxl")

    def _save_run_meta(self) -> None:
        with self._meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._run_meta, f, indent=2, default=str)
