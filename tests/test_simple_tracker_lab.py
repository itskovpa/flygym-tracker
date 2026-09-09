import json
import cv2
import numpy as np
from flygym_tracker.gui.simple_tracker_lab import SimpleTrackerLab


def test_lab_parameter_mouse_step_reprocesses_and_rotation_resets(qapp,tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from flygym_tracker.gui.stepper import StepperField
    from test_pipeline import _calibration, _full_scene
    from flygym_tracker.calibration import save_calibration
    # Use the repository's synthetic pipeline calibration written by its fixture helper.
    calib=_calibration(tmp_path)
    calib.to_json(str(tmp_path/'calibration.json'))
    video=tmp_path/'lab.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(40,40),False)
    scene=_full_scene()
    for frame in scene:writer.write(frame)
    writer.release()
    ann=tmp_path/'annotations.json'
    ann.write_text(json.dumps(dict(video=str(video),frames=[dict(index=i,elapsed_s=i/10,face='A',state='stationary' if i<3 else 'rotating') for i in range(len(scene))])))
    lab=SimpleTrackerLab(str(ann),str(tmp_path))
    lab.show();qapp.processEvents()
    box=lab.inputs['threshold']
    stepper=box.parentWidget()
    assert isinstance(stepper,StepperField)
    before=box.value()
    QTest.mouseClick(stepper.up,Qt.MouseButton.LeftButton,pos=stepper.up.rect().center())
    assert box.value()==before+1
    lab.advance();lab.advance();lab.advance()
    assert not any(t.active for t in lab.trackers.values())
    assert 'reset' in lab.status.text()
    lab.close()
