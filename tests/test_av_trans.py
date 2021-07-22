from ml import av
from ml.av.transforms import Resize, Compose
from ml.av.transforms import functional as F
from ml.av import io
from ml import nn
import torch as th
import numpy as np
import pytest

from .fixtures import img, vid

@pytest.mark.essential
def test_letterbox_pt(size=640, stride=32):
    im = th.ones((3, 220, 300), dtype=th.uint8)
    H, W = im.shape[-2:]
    resized, meta = av.utils.letterbox(im, size, stride=stride)
    sH = int(round(size / W * H))
    dH = (size - sH) % stride
    top, bottom = int(round(dH / 2 - 0.1)), int(round(dH / 2 + 0.1))
    # print(tuple(resized.shape), meta, top, bottom)
    assert resized.shape == (3, sH + top + bottom, size)
    assert meta['shape'] == im.shape[-2:]
    assert meta['offset'] == (top, 0)
    assert meta['ratio'] == (size / W, size / W)
    assert F.is_tensor(im)
    assert F.is_tensor(resized)

@pytest.mark.essential
def test_letterbox_cv2(size=640, stride=32):
    im = th.ones((220, 300, 3), dtype=th.uint8).numpy()
    H, W = im.shape[:2]
    resized, meta = av.utils.letterbox(im, size, stride=stride)
    sH = int(round(size / W * H))
    dH = (size - sH) % stride
    top, bottom = int(round(dH / 2 - 0.1)), int(round(dH / 2 + 0.1))
    # print(tuple(resized.shape), meta, top, bottom)
    assert resized.shape == (sH + top + bottom , size, 3)
    assert meta['shape'] == im.shape[:2]
    assert meta['offset'] == (top, 0)
    assert meta['ratio'] == (size / W, size / W)
    assert F.is_cv2(im)
    assert F.is_cv2(resized)

@pytest.mark.essential
def test_resize(img):
    default = F.resize(img, size=480)
    shorter = F.resize(img, size=480, constraint='shorter')
    longer = F.resize(img, size=480, constraint='longer')

    assert default.shape == shorter.shape
    assert shorter.shape[-2:] == (640, 480)
    assert longer.shape[-2:] == (480, 360)

"""
# @pytest.mark.essential
def test_ToCV():
    image = th.randn(3, 240, 320)
    trans = Compose([
        ToCV()
    ])
    pic = trans(image)
    assert isinstance(pic, np.ndarray)
    assert pic.shape == (240, 320, 3)

    from PIL import Image
    image = Image.new('RGB', (320, 240))
    pic = trans(image)
    assert isinstance(pic, np.ndarray)
    assert pic.shape == (240, 320, 3)
"""

@pytest.mark.essential
def test_Resize():
    from PIL import Image
    image = Image.new('RGB', (320, 240))
    trans = Compose([
        Resize(224, constraint='longer')
    ])
    pic = trans(image)
    assert pic.mode == image.mode
    assert pic.size == (224, int(240 / 320 * 224))
   
    trans = Compose([
        Resize(224, constraint='shorter')
    ])
    pic = trans(image)
    assert pic.mode == image.mode
    assert pic.size == (int(320 / 240 * 224), 224)