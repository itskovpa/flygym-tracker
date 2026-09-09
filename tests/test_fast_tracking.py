import csv
import time
import numpy as np
import pytest
from flygym_tracker.fast_tracking import FastParams,GroupTracker,brightness_reference,uint8_percentile90
from flygym_tracker.fast_tracking_worker import FastTrackingPool


def test_histogram_matches_interpolated_percentile():
    gray=np.arange(100,dtype=np.uint8).reshape(10,10)
    mask=np.zeros_like(gray);mask[1:9,2:7]=255
    assert brightness_reference(gray,mask)==np.percentile(gray[mask>0],90)
    assert uint8_percentile90(gray[mask>0])==np.percentile(gray[mask>0],90)
    rng=np.random.default_rng(7)
    for n in (1,2,7,90,1001,24001):
        values=rng.integers(0,256,n,dtype=np.uint8)
        assert uint8_percentile90(values)==pytest.approx(np.percentile(values,90),abs=1e-12)
    with pytest.raises(ValueError):brightness_reference(gray,np.zeros_like(mask))


def test_merge_retains_count_and_separates_gap_displacement():
    y,x=np.mgrid[:80,:100];mask=np.ones(x.shape,bool);tracker=GroupTracker()
    for i,(a,b) in enumerate([(30,50),(34,46),(37,43),(40,40),(37,43),(35,47),(31,51)]):
        signal=np.maximum(.7*np.exp(-((x-a)**2+(y-40)**2)/18),.7*np.exp(-((x-b)**2+(y-40)**2)/18))
        result=tracker.update(signal,mask,i*.05)
        if i==0:
            ids={p['id'] for p in result['measured']}
            assert len(ids)==2
        if i==3:
            assert not result['measured']
            assert result['groups'][0]['count']==2
            observed=result['observed_distance_px']
    assert {p['id'] for p in result['measured']}==ids
    assert result['gap_displacement_px']==pytest.approx(2.)
    assert observed>0
    before=tracker.observed_distance
    tracker.reset();tracker.update(signal,mask,2.)
    assert tracker.observed_distance==before


def test_initial_large_blob_is_unknown_not_a_single_fly():
    signal=np.ones((40,40),np.float32)*.5
    result=GroupTracker().update(signal,np.ones(signal.shape,bool),0)
    assert result['measured']==[]
    assert result['groups'][0]['count'] is None


def test_skipped_source_frame_is_gap_not_observed_motion():
    tracker=GroupTracker();mask=np.ones((30,30),bool)
    first=np.zeros((30,30),np.float32);first[10:14,10:14]=.5
    tracker.update(first,mask,0,frame_index=10)
    result=tracker.update(np.roll(first,2,axis=1),mask,.1,frame_index=12)
    assert result['observed_distance_px']==0
    assert result['gap_displacement_px']==2
    assert result['paths']==[]


@pytest.mark.parametrize('backend',['thread','process'])
def test_worker_drains_accepted_frames_and_reports_drops(tmp_path,backend):
    pool=FastTrackingPool(tmp_path,backend,backend=backend,window=10,depth=2)
    pool.start()
    frame=np.full((40,40),100,np.uint8);geometry={1:((0,0,40,40),np.ones((40,40),bool))}
    for i in range(35):pool.submit(frame,float(i)*.4,'A',(0,0),geometry)
    pool.reset_dwell()
    pool.close();pool.close()
    stats=pool.stats()
    assert stats['frames_submitted']+stats['frames_dropped']==35
    assert stats['frames_completed']==stats['frames_submitted']
    assert stats['failures']==0
    assert stats['pending_frames']==0
    assert stats['processing_mean_ms']>=0
    assert (tmp_path/f'fast_tracking_{backend}.csv').exists()


def test_parameter_validation():
    with pytest.raises(ValueError):FastParams(threshold=float('nan'))
    with pytest.raises(ValueError):FastParams(min_area=400,max_single_area=300)


def test_fast_preview_uses_its_own_frame_and_subtraction(qapp):
    from flygym_tracker.gui.activity_heatmap import ActivityHeatmapPanel
    snapshot={'fast_tracking':dict(frame=np.full((20,20),100,np.uint8),
        contrast=np.full((20,20),.25,np.float32),face='A',elapsed_s=1.,vials={},
        stats=dict(frames_completed=10,frames_submitted=12,frames_dropped=2,pending_frames=2))}
    panel=ActivityHeatmapPanel(snapshot)
    panel.mode_box.setCurrentIndex(4)
    assert panel.heatmap.image.pixelColor(10,10).red()==100
    assert 'dropped 2' in panel.range_label.text()
    panel.mode_box.setCurrentIndex(5)
    assert panel.heatmap.image.pixelColor(10,10).red()==128
    panel.mode_box.setCurrentIndex(4)
    assert panel.heatmap.image.pixelColor(10,10).red()==100
    panel.close()
