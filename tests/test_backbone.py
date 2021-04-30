import os
from pathlib import Path

import pytest
import numpy as np
import torch as th
from torchvision.transforms import functional as TF
from torchvision import transforms
from ml.vision.ops import clip_boxes_to_image
from ml import cv, nn
import ml

from .fixtures import *

@pytest.fixture
def dev():
    return th.device('cuda') if th.cuda.is_available() else th.device('cpu')

@pytest.fixture
def normalize():
    mean = [0.442, 0.406, 0.38] 
    std = [0.224, 0.217, 0.211]
    return transforms.Normalize(mean=mean, std=std)

@pytest.fixture
def cwd():
    return Path(__file__).parent.parent

@pytest.fixture
def chkpt_img(cwd):
    return cwd.parent / 'checkpoints/backbone/kinetics400-x101_32x8d_wsl-62.58.pth'

@pytest.fixture
def padding():
    return (0.70, 0.35)

@pytest.fixture
def dets():
    return th.Tensor([[150., 246., 348., 654.],
                      [151., 227., 197., 338.],
                      [ 70.,  43., 128., 198.],
                      [221., 302., 439., 712.],
                      [168., 274., 269., 490.],
                      [ 59.,  51., 122., 215.]])

@pytest.fixture
def model(dev):
    from ml.vision.models.backbone import resnext101
    model = resnext101(pretrained=True, groups=32, width_per_group=8)
    model.eval()
    model.to(dev)
    return model

def call_backbone_spatial(model, batch):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            r = model(batch)[-2]
            torch.cuda.synchronize()
            #print(f"r.grad_fn={r.grad_fn}")
            #print(f"i.grad_fn={i.grad_fn}")
            return r

def call_backbone(model, batch):
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            feats = model(batch)[-1]
            torch.cuda.synchronize()
            # print(f"feats.grad_fn={feats.grad_fn}")
            return feats

@pytest.mark.essential
@pytest.mark.parametrize("batch_size", [10])
def test_resnext101_spatial_feats(benchmark, model, normalize, dev, batch_size):
    spatial_transform = transforms.Compose([normalize])
    batch = []
    for n in range(batch_size):
        frame = th.rand((3, 720, 1280), dtype=th.float32)
        frame = spatial_transform(frame).to(dev)
        batch.append(frame)

    batch = th.stack(batch)
    torch.cuda.synchronize()
    spatial_feats = benchmark(call_backbone_spatial, model, batch)
    assert spatial_feats.shape[0] == batch_size  

@pytest.mark.essential
@pytest.mark.parametrize("streams", [2, 4])
def test_resnext101_feats(benchmark, model, dev, normalize, dets, padding, streams):
    im_transform = transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       normalize, ])
    frames = th.rand((streams, 3, 720, 1280), dtype=th.float32)
    
    H, W = frames.shape[-2:]
    width = dets[:, 2] - dets[:, 0] + 1
    height = dets[:, 3] - dets[:, 1] + 1
    paddingW = width * padding[0]
    paddingH = height * padding[1]
    dets[:, 0] -= paddingW
    dets[:, 1] -= paddingH
    dets[:, 2] += paddingW
    dets[:, 3] += paddingH
    dets = clip_boxes_to_image(dets, (H, W))

    batch = []
    for s in range(streams):
        for box in dets.round().long():
            x1, y1, x2, y2 = box.tolist()
            batch.append(im_transform(frames[s, :, y1:y2, x1:x2]))

    batch = th.stack(batch).to(dev)
    torch.cuda.synchronize()
    feats = benchmark(call_backbone, model, batch)
    assert feats.shape[0] == batch.shape[0]