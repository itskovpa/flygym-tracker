"""A recording's frame number is not an experiment clock when frames were dropped."""
import csv

import cv2
import pytest

from flygym_tracker.frame_source import VideoFileSource
from flygym_tracker.logger import ActivityLogger
from flygym_tracker.pipeline import TrackerPipeline
from test_pipeline import _calibration, _config, _full_scene


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 40), False)
    assert writer.isOpened()
    for frame in _full_scene():
        writer.write(frame)
    writer.release()
    return path


def write_stamps(clip, stamps):
    with clip.with_name(clip.stem + "_frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_frame", "elapsed_s"])
        writer.writerows(enumerate(stamps))


@pytest.mark.parametrize("clock", ["auto", "index"])
def test_pipeline_auto_uses_recording_times_and_index_remains_explicit(clip, tmp_path, clock):
    stamps = [i / 10 + (5 if i >= 15 else 0) for i in range(30)]
    write_stamps(clip, stamps)
    source = VideoFileSource(str(clip))
    pipeline = TrackerPipeline(_config(), _calibration(tmp_path), source,
                               ActivityLogger(tmp_path / "results", "test", fmt="csv"), clock=clock)
    times = []
    pipeline.add_observer(lambda payload: times.append(payload["elapsed_s"]))
    summary = pipeline.run()
    assert summary["frames_processed"] == 30
    assert summary["observer_failures"] == 0
    assert times == (stamps if clock == "auto" else [i / 10 for i in range(30)])


def test_no_sidecar_keeps_existing_video_support(clip):
    with VideoFileSource(str(clip)) as source:
        assert source.recorded_elapsed_s(0) is None
        assert source.read().index == 0


@pytest.mark.parametrize("stamps", [[], [0], [float("nan")]*30,
                                    [float("inf")]*30, [-1]*30, list(range(30,0,-1))])
def test_invalid_sidecar_is_not_silently_replaced_with_nominal_timing(clip, stamps):
    write_stamps(clip, stamps)
    source = VideoFileSource(str(clip))
    with pytest.raises(ValueError, match="invalid recording timestamps"):
        source.open()
    assert source._cap is None


def test_sidecar_indices_must_match_frames(clip):
    write_stamps(clip, range(30))
    path = clip.with_name(clip.stem + "_frames.csv")
    path.write_text(path.read_text().replace("0,0", "7,0"))
    with pytest.raises(ValueError, match="indices"):
        VideoFileSource(str(clip)).open()
