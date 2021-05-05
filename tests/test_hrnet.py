
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

# @pytest.mark.essential
@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("arch", ['litehrnet_30_coco_384x288'])
def test_posenet(pretrained, arch):
    from ml.vision.models.pose import posenet
    print()
    model = posenet(pretrained=pretrained, arch=arch, force_reload=False, unload_after=True)
    # print(model.tag, model)