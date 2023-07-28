import pytest

import torch

from .fixtures import *

## Set operations on detections

@pytest.mark.essential
def test_dets_select():
    from ml.vision.ops import dets_select
    dets = torch.ones(4, 3)
    dets[:, -1] = torch.tensor([0,1,2,3])
    selection = dets_select(dets, [1, 3])
    assert (selection == torch.tensor([False, True, False, True])).all()