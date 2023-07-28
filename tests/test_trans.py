import pytest

from ml import logging

from ml.vision import utils
from ml.vision.transforms import Resize
from ml.vision.transforms.functional import resize

import numpy as np

import torch

from .fixtures import img 

@pytest.mark.essential
def test_gen_patches():
    H = 128
    W = 320
    patch_size = (H, W)
    im = torch.randint(0, 255, (3, 720, 1280), dtype=torch.uint8)
    img_patches, boxes = utils.gen_patches(im, resize=640, patch_size=patch_size)
    rows = 640 / W
    cols = 640 / H
    assert img_patches.shape[0] == rows * cols
    assert img_patches.shape[1] == 3
    assert img_patches.shape[2] == H
    assert img_patches.shape[3] == W
    assert len(boxes) == len(img_patches)

@pytest.mark.essential
def test_resize(img):
    default = resize(img, size=480)
    shorter = resize(img, size=480, constraint='shorter')
    longer = resize(img, size=480, constraint='longer')

    assert default.shape == shorter.shape
    assert shorter.shape[-2:] == (640, 480)
    assert longer.shape[-2:] == (480, 360)

@pytest.mark.essential
def test_resize_preserve_aspect_ratio_shorter_edge():
    # Test resizing by the shorter edge with preserving aspect ratio
    # Smaller image with height < width
    img = torch.rand(3, 100, 200)  # C, H, W
    size = 64
    result = resize(img, size, constraint='shorter')
    assert result.shape[-2:] == (size, 128)  # Width should be scaled down to 128

    # Smaller image with height > width
    img = torch.rand(3, 200, 100)  # C, H, W
    size = 64
    result = resize(img, size, constraint='shorter')
    assert result.shape[-2:] == (128, size)  # Height should be scaled down to 128

@pytest.mark.essential
def test_resize_preserve_aspect_ratio_longer_edge():
    # Test resizing by the longer edge with preserving aspect ratio
    # Larger image with height < width
    img = torch.rand(3, 100, 200)  # C, H, W
    size = 300
    result = resize(img, size, constraint='longer')
    assert result.shape[-2:] == (150, 300)  # Height should be scaled up to 150

    # Larger image with height > width
    img = torch.rand(3, 200, 100)  # C, H, W
    size = 300
    result = resize(img, size, constraint='longer')
    assert result.shape[-2:] == (300, 150)  # Width should be scaled up to 150

@pytest.mark.essential
def test_resize_exact_size():
    # Test resizing to an exact size without preserving aspect ratio
    img = torch.rand(3, 200, 100)  # C, H, W
    size = (64, 128)
    result = resize(img, size)
    assert result.shape[-2:] == size