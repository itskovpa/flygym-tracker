"""A whole run with recording on: the video appears, and the measurement is byte-identical.

THE CLAIM BEING TESTED is the one the rig owner asked for -- recording "does not drain the cpu and
affect the ongoing processes as little as possible". The strong form of that, and the only form
worth testing, is that turning recording ON DOES NOT CHANGE THE MEASUREMENT AT ALL. Two runs over
identical frames, one recording and one not, must produce the same activity rows.
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import pytest

from flygym_tracker.calibration import build_two_face_calibration_from_polygons, save_calibration
from flygym_tracker.config import load_config

H, W = 240, 400
FPS = 20.0
N_FRAMES = 80


def _frame(i):
    """A fly climbing in vial 1, one sitting still in vial 2, vial 3 empty."""
    frame = np.full((H, W), 200, dtype=np.uint8)
    y = 190 - 3 * (i % 40)
    frame[y:y + 8, 55:65] = 40
    frame[120:128, 175:185] = 40
    return frame


class _Source:
    """Deterministic frames, so two runs differ only in whether they recorded."""

    fps = FPS

    def __init__(self):
        self.i = 0

    def open(self):
        pass

    def close(self):
        pass

    def read(self):
        from flygym_tracker.types import Frame

        if self.i >= N_FRAMES:
            return None
        # `t_wall_iso` IS NOT OPTIONAL. Leaving it off made every read raise, and the pipeline's
        # retry loop then span forever without advancing -- which looked exactly like the recorder
        # deadlocking, and is worth remembering the next time a run appears to hang.
        frame = Frame(index=self.i, t_monotonic=self.i / FPS, image=_frame(self.i),
                      t_wall_iso="2026-01-01T00:00:%06.3f" % (self.i / FPS))
        self.i += 1
        return frame


@pytest.fixture
def calib_dir(tmp_path):
    polygons = [[[40 + 60 * c, 60], [90 + 60 * c, 60], [90 + 60 * c, 210], [40 + 60 * c, 210]]
                for c in range(3)]
    calib, masks, _ = build_two_face_calibration_from_polygons(
        polygons, _frame(0), (W, H), faces=("A", "B"))
    out = str(tmp_path / "calib")
    save_calibration(calib, masks, out)
    return out


def _activity_rows(directory):
    rows = []
    for path in sorted(pathlib.Path(directory).glob("activity_*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _run(tmp_path, calib_dir, name, recording):
    """One full run through `RunWorker`, which is where the recorder is actually wired in."""
    from flygym_tracker.gui.run_controller import RunWorker

    output = tmp_path / name
    output.mkdir()
    config = load_config(overrides={"binning": {"bin_seconds": 1.0},
                                    "activity": {"pixel_threshold": 1.0},
                                    "rotation": {"detector": "adaptive"}})
    worker = RunWorker({"config": config, "calib_dir": calib_dir, "output_dir": str(output),
                        "source_factory": _Source, "recording": recording})
    summaries = []
    worker.finished.connect(summaries.append)
    worker.failed.connect(lambda message: pytest.fail("the run failed: %s" % message))
    worker.run()
    assert summaries, "the run produced no summary"
    return output, summaries[0]


# =============================================================================================
def test_recording_does_not_change_what_is_measured(qapp, tmp_path, calib_dir):
    """THE HEADLINE CLAIM. Identical frames in, identical activity rows out -- the recorder is a
    consumer of frames and nothing else. If this ever fails, recording has reached the measurement
    and no amount of it being fast would make that acceptable."""
    plain_dir, plain = _run(tmp_path, calib_dir, "plain", {"enabled": False})
    taped_dir, taped = _run(tmp_path, calib_dir, "taped",
                            {"enabled": True, "every_nth": 1, "scale": 1.0})

    assert plain["frames_processed"] == taped["frames_processed"] == N_FRAMES

    plain_rows, taped_rows = _activity_rows(plain_dir), _activity_rows(taped_dir)
    assert plain_rows, "the un-recorded run measured nothing, so this proves nothing"
    assert len(plain_rows) == len(taped_rows)
    for a, b in zip(plain_rows, taped_rows):
        for field in ("vial_id", "face", "elapsed_s", "motion_px_sum", "active_fraction_mean",
                      "lit_area_px", "n_stationary_frames", "n_rotating_frames"):
            assert a[field] == b[field], "recording changed %s: %r vs %r" % (field, a[field],
                                                                            b[field])


def test_the_video_is_written_beside_the_results_with_the_runs_stamp(qapp, tmp_path, calib_dir):
    output, summary = _run(tmp_path, calib_dir, "run",
                           {"enabled": True, "every_nth": 1, "scale": 1.0})

    videos = list(output.glob("video_*.avi"))
    assert len(videos) == 1, "expected one video, found %r" % [v.name for v in videos]
    stamp = videos[0].stem.split("video_")[1]
    # THE SAME STAMP AS THE CSVs, which is what groups one run's files together in a directory
    # holding a season of them.
    assert list(output.glob("activity_%s_*.csv" % stamp)), "the video carries a different stamp"
    assert summary["video"]["frames_written"] == N_FRAMES
    assert summary["video"]["error"] is None


def test_the_recorded_video_can_be_read_back(qapp, tmp_path, calib_dir):
    """A file that exists and a file that plays are different claims."""
    import cv2

    output, _summary = _run(tmp_path, calib_dir, "run",
                            {"enabled": True, "every_nth": 1, "scale": 1.0})
    video = next(output.glob("video_*.avi"))
    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    assert ok and frame is not None
    assert frame.shape[0] == H and frame.shape[1] == W


def test_the_sidecar_lines_the_video_up_with_the_measurement(qapp, tmp_path, calib_dir):
    """The video's own frame numbers are not a clock -- frames are skipped by request and dropped
    under load. The sidecar carries the same `elapsed_s` activity.csv uses, which is what makes
    "what was the fly doing at this point in the graph" answerable."""
    output, _summary = _run(tmp_path, calib_dir, "run",
                            {"enabled": True, "every_nth": 4, "scale": 1.0})
    sidecar = next(output.glob("video_*_frames.csv"))
    with open(sidecar, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == N_FRAMES // 4
    assert [int(row["video_frame"]) for row in rows[:3]] == [0, 1, 2]
    # Video frame 1 is the run's frame 4, i.e. 0.2 s in at 20 fps -- NOT 0.05 s, which is what
    # reading the file's own frame numbers as a timeline would give.
    assert float(rows[1]["elapsed_s"]) == pytest.approx(4 / FPS, abs=1e-3)


def test_no_video_appears_when_it_was_not_asked_for(qapp, tmp_path, calib_dir):
    output, summary = _run(tmp_path, calib_dir, "run", {"enabled": False})
    assert not list(output.glob("video_*")), "a video was written without being asked for"
    assert summary["video"] is None
