import numpy as np
import pytest
from flygym_tracker.simple_tracker import CentroidTracker, SimpleTrackerParams, detect_centroids


def test_absolute_threshold_area_and_mask():
    image=np.full((20,20),200,np.uint8)
    image[2:4,2:4]=80
    image[8:12,8:12]=50
    image[15,15]=20
    mask=np.ones(image.shape,bool)
    mask[8:12,8:12]=False
    blobs,*_=detect_centroids(image,mask,SimpleTrackerParams(threshold=80,min_area=2,max_area=8))
    assert blobs==[(2.5,2.5,4)]


def test_gap_displacement_is_separate_and_speed_uses_real_time():
    tracker=CentroidTracker(SimpleTrackerParams(max_speed=50,max_gap_s=.4))
    assert tracker.update([(0,0,10)],0,0)[0]['fragment']==1
    tracker.update([(2,0,10)],.1,1)
    tracker.update([],.2,2)
    result=tracker.update([(5,0,10)],.3,3)
    assert result[0]['fragment']==1 and result[0]['gap']
    assert tracker.observed_distance==2
    assert tracker.gap_displacement==3
    assert tracker.statistics()['mean_link_speed_px_s']==20


def test_one_to_one_assignment_and_expiration():
    tracker=CentroidTracker(SimpleTrackerParams(max_gap_s=.1))
    tracker.update([(0,0,10),(10,0,10)],0,0)
    linked=tracker.update([(4,0,10)],.05,1)
    assert len(linked)==1
    tracker.update([],.1,2)
    result=tracker.update([(4,0,10)],.3,3)
    assert result[0]['fragment']==3


def test_rotation_resets_links_without_adding_jump_distance():
    tracker=CentroidTracker()
    tracker.update([(0,0,10)],0,0)
    tracker.reset()
    tracker.update([(20,0,10)],1,10)
    assert tracker.fragments==2
    assert tracker.observed_distance==tracker.gap_displacement==0


def test_gate_and_monotonic_frames():
    tracker=CentroidTracker(SimpleTrackerParams(max_speed=10))
    tracker.update([(0,0,10)],0,0)
    assert tracker.update([(10,0,10)],.1,1)[0]['fragment']==2
    with pytest.raises(ValueError): tracker.update([],.1,1)
