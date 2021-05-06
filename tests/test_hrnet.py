
from pathlib import Path
import pytest

from torch.cuda.amp import autocast
import torch as th
import numpy as np

from ml import cv, logging
from ml.vision.models import yolo4, yolo5, yolo5l, yolo5x, rfcn
from ml.vision.models.detection import yolo
from ml.vision.datasets.coco import COCO80_CLASSES
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml.vision.ops import xyxys2xyxysc, xcycwh2xyxy, xcycwh2xywh, xyxy2xcycwh

from .fixtures import *

@pytest.fixture
def img():
    return 'assets/bus_zidane_tiles.jpg'

# @pytest.mark.essential
@pytest.mark.parametrize("pretrained", [True])
@pytest.mark.parametrize("arch", ['litehrnet_30_coco_384x288'])
def test_posenet(pretrained, arch, img):
    from ml.vision.models.pose import posenet, inference
    print()
    model = posenet(pretrained=pretrained, arch=arch, force_reload=False, unload_after=True, fp16=False)
    # print(model.tag, model)

    # YOLOv5 as the detector
    from ml.vision.models import yolo5x, yolo5
    detector = yolo5(pretrained=True, 
                     classes=80,
                     chkpt='yolov5x', 
                     tag='v3.1', 
                     s3=None, 
                     fuse=True, 
                     force_reload=False)

    # pose inference of 17 keypoints with HRNet
    """
        COCO keypoint indexes::
        0: 'nose',
        1: 'left_eye',
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',
        6: 'right_shoulder',
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',
        10: 'right_wrist',
        11: 'left_hip',
        12: 'right_hip',
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',
        16: 'right_ankle'
    """    
    results, vis_img = inference(detector, model, img, vis=True)
    # print(results)
    cv.save(vis_img, f'export/{Path(img).name}')