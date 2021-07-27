from ml import av, logging
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
    im = th.randint(0, 255, (3, 220, 300), dtype=th.uint8)
    H, W = im.shape[-2:]
    resized, meta = av.letterbox(im, size, stride=stride)
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

#@pytest.mark.essential
def test_letterbox_cv2(size=640, stride=32):
    im = th.randint(0, 255, (220, 300, 3), dtype=th.uint8).numpy()
    H, W = im.shape[:2]
    resized, meta = av.letterbox(im, size, stride=stride)
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

def test_resize_cmp(img):
    """Compare bilinear resize across frameworks to verify if half_pixel_centers is implemented.
    FIXME: inconsist with opencv
    """
    from ml.av.backend import opencv as cv
    import tensorflow as tf
    import numpy as np
    import torch as th

    osz = (810, 1080)
    rsz = (480,  640)
    #osz = (20,  20)
    #rsz = (8,  8)

    img = th.randint(0, 255, (3, *osz), dtype=th.uint8)
    img_cv = img.permute(1, 2, 0).numpy()[:, :, ::-1]
    img_tf = tf.constant(img_cv, dtype=tf.uint8)
    resized_cv = F.resize(img_cv, size=rsz[0])
    resized = F.resize(img, size=rsz[0])

    logging.info(f"tf img shape={tuple(img_tf.shape)}, dtype={img_tf.dtype}, sum={img_tf.numpy().astype(np.uint8).sum(axis=(0,1))}")
    #resized_tf = tf.compat.v1.image.resize_bilinear(img_tf, (640, 480), align_corners=False, half_pixel_centers=True)
    resized_tf = tf.image.resize(img_tf, rsz, method=tf.image.ResizeMethod.BILINEAR)
    resized_tf = tf.squeeze(resized_tf).numpy().round()


    logging.info(f"tv im after resize: {tuple(resized.shape)}, {resized.dtype}, sum={resized.sum(dim=(1, 2))}")
    logging.info(f"cv im after resize: {tuple(resized_cv.shape)}, {resized_cv.dtype}, sum={resized_cv.sum(axis=(0, 1))}")
    logging.info(f"tf im after resize: {tuple(resized_tf.shape)}, {resized_tf.dtype}, sum={resized_tf.sum(axis=(0, 1))}")

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

def test_ToCV():
    from ml.av.transforms import ToCV
    image = th.randn(3, 240, 320)
    trans = Compose([
        ToCV()
    ])
    pic = trans(image)
    assert isinstance(pic, np.ndarray)
    assert pic.shape == (240, 320, 3)

    from ml.av.backend.pil import Image
    image = Image.new('RGB', (320, 240))
    pic = trans(image)
    assert isinstance(pic, np.ndarray)
    assert pic.shape == (240, 320, 3)

