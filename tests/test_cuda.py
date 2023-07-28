import pytest
import numpy as np
import torch as th

from ml import logging

@pytest.fixture
def CPU():
    return th.device('cpu')

@pytest.fixture
def GPU():
    return th.device(0)

@pytest.fixture
def IMG():
    H, W = 720, 1280
    return th.rand(1, 3, H, W)

# @pytest.mark.essential
def test_custom_roi_pool(IMG, GPU):
    from ml.vision.ops import (
        roi_align,
        roi_pool,
    )
    H, W = IMG.shape[-2:]
    feats = th.rand(1, 1, IMG.shape[-2] // 16, IMG.shape[-1] // 16)
    output_size = (7, 7)
    rois = th.rand(1000, 4)
    rois[:, 2:] += rois[:, :2]
    rois *= th.Tensor([W, H, W, H])
    rois = rois.to(GPU)
    feats = feats.to(GPU)

    fh, fw = list(feats.shape)[-2:]
    h, w = list(IMG.shape)[-2:]
    scales = (fh/h, fw/w)
    try:
        print(roi_pool.__module__)
        rois_feats = roi_pool(feats, [rois], output_size, scales).squeeze()
        logging.info(f"RoI pooled features: {rois_feats.shape}")
        rois_feats = roi_align(feats, [rois], output_size, scales).squeeze()
        logging.info(f"RoI aligned features: {rois_feats.shape}")
    except Exception as e:
        assert False, f"Missing custom CUDA kernel: {e}"