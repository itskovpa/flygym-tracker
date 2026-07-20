"""Tests for flygym_tracker.logger.ActivityLogger — schema, resume, daily rolling, xlsx parity.

See DESIGN.md §7 (output format) and §8 (test_logger.py requirements).
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from flygym_tracker.logger import ActivityLogger
from flygym_tracker.types import (
    ACTIVITY_COLUMNS,
    EVENT_COLUMNS,
    ActivityRecord,
    EventRecord,
)


def _activity_record(**overrides) -> ActivityRecord:
    base = dict(
        run_id="run1",
        bin_start_iso="2026-07-18T10:00:00",
        bin_end_iso="2026-07-18T10:01:00",
        elapsed_s=60.0,
        face="A",
        vial_id=1,
        row=0,
        col=0,
        present=True,
        n_stationary_frames=1750,
        n_rotating_frames=50,
        motion_px_sum=1234,
        active_fraction_mean=0.0456,
        lit_area_px=5000,
    )
    base.update(overrides)
    return ActivityRecord(**base)


# ---------------------------------------------------------------------------------------------
# Schema / column order / dtypes
# ---------------------------------------------------------------------------------------------


def test_log_activity_writes_columns_and_rows(tmp_path):
    records = [
        _activity_record(vial_id=1, col=0, bin_start_iso="2026-07-18T10:00:00",
                          bin_end_iso="2026-07-18T10:01:00", motion_px_sum=100),
        _activity_record(vial_id=2, col=1, bin_start_iso="2026-07-18T10:00:00",
                          bin_end_iso="2026-07-18T10:01:00", motion_px_sum=200),
        _activity_record(vial_id=1, col=0, bin_start_iso="2026-07-18T10:01:00",
                          bin_end_iso="2026-07-18T10:02:00", motion_px_sum=300),
        _activity_record(vial_id=2, col=1, bin_start_iso="2026-07-18T10:01:00",
                          bin_end_iso="2026-07-18T10:02:00", motion_px_sum=400),
    ]

    logger = ActivityLogger(output_dir=tmp_path, run_id="run1", fmt="csv",
                             meta={"versions": {"flygym_tracker": "0.1.0.dev0"}})
    logger.log_activity(records)
    logger.close()

    csv_path = logger.activity_path("20260718")
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ACTIVITY_COLUMNS
    assert len(df) == 4

    # exactly one row per (vial_id, bin_start_iso)
    pairs = list(zip(df["vial_id"], df["bin_start_iso"]))
    assert len(set(pairs)) == 4

    assert pd.api.types.is_integer_dtype(df["vial_id"])
    assert pd.api.types.is_integer_dtype(df["row"])
    assert pd.api.types.is_integer_dtype(df["col"])
    assert pd.api.types.is_integer_dtype(df["n_stationary_frames"])
    assert pd.api.types.is_integer_dtype(df["n_rotating_frames"])
    assert pd.api.types.is_integer_dtype(df["motion_px_sum"])
    assert pd.api.types.is_integer_dtype(df["lit_area_px"])
    assert pd.api.types.is_float_dtype(df["elapsed_s"])
    assert pd.api.types.is_float_dtype(df["active_fraction_mean"])
    assert pd.api.types.is_bool_dtype(df["present"])

    row0 = df.iloc[0]
    assert row0["run_id"] == "run1"
    assert row0["face"] == "A"
    assert bool(row0["present"]) is True
    assert int(row0["motion_px_sum"]) == 100
    assert row0["active_fraction_mean"] == pytest.approx(0.0456)

    meta_path = logger.meta_path()
    assert meta_path.exists()
    meta = json.loads(logger.meta_path().read_text(encoding="utf-8"))
    assert meta["run_id"] == "run1"
    assert meta["stop_iso"] is not None
    assert meta["meta"]["versions"]["flygym_tracker"] == "0.1.0.dev0"


def test_log_event_writes_schema(tmp_path):
    logger = ActivityLogger(output_dir=tmp_path, run_id="run1", fmt="csv")
    logger.log_event(EventRecord(
        run_id="run1", iso_time="2026-07-18T10:00:00", elapsed_s=0.0,
        event="rotation_start", detail="",
    ))
    logger.log_event(EventRecord(
        run_id="run1", iso_time="2026-07-18T10:00:05", elapsed_s=5.0,
        event="rotation_end", detail="face=A",
    ))
    logger.close()

    events_path = logger.events_path()
    assert events_path.exists()

    df = pd.read_csv(events_path, keep_default_na=False)
    assert list(df.columns) == EVENT_COLUMNS
    assert len(df) == 2
    assert df.iloc[0]["run_id"] == "run1"
    assert df.iloc[0]["event"] == "rotation_start"
    assert df.iloc[0]["detail"] == ""
    assert df.iloc[0]["elapsed_s"] == pytest.approx(0.0)
    assert df.iloc[1]["event"] == "rotation_end"
    assert df.iloc[1]["detail"] == "face=A"
    assert pd.api.types.is_float_dtype(df["elapsed_s"])


# ---------------------------------------------------------------------------------------------
# Resume: append without duplicating the header, no clobbering, last_bin() recovery
# ---------------------------------------------------------------------------------------------


def test_resume_appends_without_duplicate_header_and_last_bin(tmp_path):
    now = datetime.now().replace(microsecond=0)
    bin0, bin1, bin2 = now, now + timedelta(minutes=1), now + timedelta(minutes=2)
    bin0_iso, bin1_iso, bin2_iso = bin0.isoformat(), bin1.isoformat(), bin2.isoformat()

    logger1 = ActivityLogger(output_dir=tmp_path, run_id="resume_run", fmt="csv")
    assert logger1.last_bin() is None  # nothing recorded yet today
    logger1.log_activity([
        _activity_record(run_id="resume_run", vial_id=1,
                          bin_start_iso=bin0_iso, bin_end_iso=bin1_iso),
        _activity_record(run_id="resume_run", vial_id=1,
                          bin_start_iso=bin1_iso, bin_end_iso=bin2_iso),
    ])
    logger1.close()

    # A RESTART NOW WRITES ITS OWN FILE. Filenames carry the RUN's start stamp, so logger2 does
    # not append to logger1's file -- and that is the point of the stamp: one recording, one set
    # of files. What resuming still has to do is not RE-LOG bins that are already on disk, and
    # `last_bin` provides that by reading every activity file for the day whoever wrote it.
    logger2 = ActivityLogger(output_dir=tmp_path, run_id="resume_run", fmt="csv")
    assert logger2.last_bin() == bin1_iso, "a restart could not see the earlier run's rows"

    logger2.log_activity([
        _activity_record(run_id="resume_run", vial_id=1, bin_start_iso=bin2_iso,
                          bin_end_iso=(bin2 + timedelta(minutes=1)).isoformat()),
    ])
    logger2.close()

    header = ",".join(ACTIVITY_COLUMNS)
    files = sorted(tmp_path.glob("activity_*.csv"))
    # One file per run, unless both runs started inside the same SECOND -- the stamp's resolution.
    # Either way the header must appear once per file and no bin may be lost or duplicated, which
    # is what resuming actually has to guarantee.
    assert files, "no activity file was written at all"
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == header
        assert sum(1 for line in lines if line == header) == 1, "duplicated header in %s" % path.name

    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    assert [r["bin_start_iso"] for r in rows] == [bin0_iso, bin1_iso, bin2_iso]


def test_daily_rolling_by_record_date(tmp_path):
    day1, day1_end = "2026-01-01T23:59:00", "2026-01-02T00:00:00"
    day2, day2_end = "2026-01-02T00:05:00", "2026-01-02T00:06:00"

    logger = ActivityLogger(output_dir=tmp_path, run_id="rollover_run", fmt="csv")
    # One call, records for both days mixed together -- each must route independently.
    logger.log_activity([
        _activity_record(run_id="rollover_run", vial_id=1, bin_start_iso=day1, bin_end_iso=day1_end),
        _activity_record(run_id="rollover_run", vial_id=2, bin_start_iso=day1, bin_end_iso=day1_end),
        _activity_record(run_id="rollover_run", vial_id=1, bin_start_iso=day2, bin_end_iso=day2_end),
    ])
    logger.close()

    path1 = logger.activity_path("20260101")
    path2 = logger.activity_path("20260102")
    assert path1.exists()
    assert path2.exists()

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    assert len(df1) == 2
    assert len(df2) == 1
    assert set(df1["vial_id"].tolist()) == {1, 2}
    assert set(df2["vial_id"].tolist()) == {1}
    assert (df1["bin_start_iso"] == day1).all()
    assert (df2["bin_start_iso"] == day2).all()


# ---------------------------------------------------------------------------------------------
# fmt="both": xlsx regenerated from CSV on close(), must match exactly
# ---------------------------------------------------------------------------------------------


def test_fmt_both_xlsx_matches_csv(tmp_path):
    records = [
        _activity_record(run_id="both_run", vial_id=1, col=0,
                          bin_start_iso="2026-07-18T09:00:00", bin_end_iso="2026-07-18T09:01:00"),
        _activity_record(run_id="both_run", vial_id=2, col=1, present=False,
                          bin_start_iso="2026-07-18T09:00:00", bin_end_iso="2026-07-18T09:01:00"),
    ]
    event = EventRecord(run_id="both_run", iso_time="2026-07-18T09:00:00", elapsed_s=0.0,
                         event="calibration", detail="initial")

    logger = ActivityLogger(output_dir=tmp_path, run_id="both_run", fmt="both")
    logger.log_activity(records)
    logger.log_event(event)
    logger.close()

    csv_path = logger.activity_path("20260718")
    xlsx_path = csv_path.with_suffix(".xlsx")
    assert xlsx_path.exists()

    # check_dtype=False: xlsx (OOXML) has no distinct int/float storage type -- openpyxl
    # serializes a whole-number float like 60.0 as "60" (see logger.py module docstring /
    # ActivityLogger._rewrite_xlsx), so a column that is *entirely* whole numbers (e.g.
    # elapsed_s=60.0 for every bin) reads back as int64 instead of float64. The numeric value
    # is exactly preserved either way (60 == 60.0); only the pandas dtype label can differ.
    # assert_frame_equal still does full value-for-value comparison (with float tolerance),
    # so this remains a strict check of row/column/value agreement.
    df_csv = pd.read_csv(csv_path)
    df_xlsx = pd.read_excel(xlsx_path, engine="openpyxl")
    pd.testing.assert_frame_equal(df_csv, df_xlsx, check_dtype=False)

    events_csv = logger.events_path()
    events_xlsx = logger.events_path().with_suffix(".xlsx")
    assert events_xlsx.exists()
    df_events_csv = pd.read_csv(events_csv, keep_default_na=False)
    df_events_xlsx = pd.read_excel(events_xlsx, engine="openpyxl", keep_default_na=False)
    pd.testing.assert_frame_equal(df_events_csv, df_events_xlsx, check_dtype=False)


# ---------------------------------------------------------------------------------------------
# Robustness: empty lists, context manager
# ---------------------------------------------------------------------------------------------


def test_log_activity_empty_list_is_noop(tmp_path):
    logger = ActivityLogger(output_dir=tmp_path, run_id="empty_run", fmt="both")
    logger.log_activity([])
    logger.flush()
    logger.close()
    assert list(tmp_path.glob("activity_*.csv")) == []
    assert list(tmp_path.glob("activity_*.xlsx")) == []
    assert logger.meta_path().exists()


def test_context_manager_closes_and_stamps_stop_time(tmp_path):
    with ActivityLogger(output_dir=tmp_path, run_id="ctx_run", fmt="csv") as logger:
        assert logger.last_bin() is None

    meta = json.loads(logger.meta_path().read_text(encoding="utf-8"))
    assert meta["start_iso"] is not None
    assert meta["stop_iso"] is not None
