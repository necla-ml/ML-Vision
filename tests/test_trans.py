import pytest

from ml.vision.transforms import Resize, ToCV, Compose
import torch as th
import numpy as np

@pytest.mark.essential
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