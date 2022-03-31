from pathlib import Path

import pytest

import torch as th
from ml.vision import transforms

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
def dets():
    return th.Tensor([[150., 246., 348., 654.],
                      [151., 227., 197., 338.],
                      [ 70.,  43., 128., 198.],
                      [221., 302., 439., 712.],
                      [168., 274., 269., 490.],
                      [ 59.,  51., 122., 215.]])

@pytest.fixture
def model(dev):
    from ml.vision.models.backbone import convnext_small
    model = convnext_small(pretrained=True)
    model.eval()
    model.to(dev)
    return model

def call_backbone_spatial(model, batch):
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=True):
            r = model(batch)[-2]
            th.cuda.synchronize()
            #print(f"r.grad_fn={r.grad_fn}")
            #print(f"i.grad_fn={i.grad_fn}")
            return r

def call_backbone(model, batch):
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=True):
            feats = model(batch)[-1]
            th.cuda.synchronize()
            # print(f"feats.grad_fn={feats.grad_fn}")
            return feats

@pytest.mark.essential
@pytest.mark.parametrize("batch_size", [10])
def test_convnext_small_spatial_feats(benchmark, model, normalize, dev, batch_size):
    spatial_transform = transforms.Compose([normalize])
    batch = []
    for n in range(batch_size):
        frame = th.rand((3, 720, 1280), dtype=th.float32)
        frame = spatial_transform(frame).to(dev)
        batch.append(frame)

    batch = th.stack(batch)
    th.cuda.synchronize()
    spatial_feats = benchmark(call_backbone_spatial, model, batch)
    assert spatial_feats.shape[0] == batch_size