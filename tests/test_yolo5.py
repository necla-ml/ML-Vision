from pathlib import Path
import pytest

from torch.cuda.amp import autocast
import torch as th
import numpy as np

from ml import logging
from ml.vision.models import yolo4, yolo5, yolo5l, yolo5x, rfcn
from ml.vision.models.detection import yolo
from ml.vision.datasets.coco import COCO80_CLASSES
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml.vision.ops import xyxys2xyxysc, xcycwh2xyxy, xcycwh2xywh, xyxy2xcycwh

from .fixtures import *

@pytest.fixture
def detector(tag):
    detector = yolo5x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False)
    assert detector.module.tag == tag
    detector.eval()
    return detector.to('cuda' if th.cuda.is_available() else 'cpu')


@pytest.mark.essential
@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("amp", [True, False])
def test_detect(benchmark, detector, fps, amp):
    size = 640
    frame_qb = th.randint(255, (720, 1056, 3), dtype=th.uint8).numpy()
    frame_latham = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame_qb] * fps + [frame_latham] * fps
    cfg = dict(
        cls_thres = 0.0001,
        # cls_thres = 1,
        nms_thres = 0.1,
        agnostic = False,
        merge = True,
    )
    
    amp = autocast(enabled=amp)
    with th.no_grad():
        with amp:
            # dets, pooled = detector.detect(frames, **cfg)
            dets, pooled = benchmark(detector.detect, frames, amp=amp, **cfg)
            print(dets[0].shape, dets[0].dtype, pooled[0].shape, pooled[0].dtype)


@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("amp", [True, False])
def test_model_preprocess(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame_qb = th.randint(255, (720, 1056, 3), dtype=th.uint8).numpy()
    frame_latham = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame_qb] * fps + [frame_latham] * fps
    with amp:
        # batch, metas = yolo.preprocess(frames, size)
        batch, metas = benchmark(yolo.preprocess, frames, size)


@pytest.mark.essential
@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("amp", [True, False])
def test_model_forward(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame_qb = th.randint(255, (720, 1056, 3), dtype=th.uint8).numpy()
    frame_latham = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame_qb] * fps + [frame_latham] * fps
    with amp:
        batch, metas = yolo.preprocess(frames, size)
        # batch, metas = benchmark(yolo.preprocess, frames, size)

    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.no_grad():
        with amp:
            # predictions = model(batch)
            predictions = benchmark(model, batch)
            # print('predictions:', predictions.shape)
            cfg = dict(
                conf_thres = 0.0001,
                iou_thres = 0.1,
                agnostic = False,
                merge = True,
            )


@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("amp", [True, False])
def test_model_postprocess(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame_qb = th.randint(255, (720, 1056, 3), dtype=th.uint8).numpy()
    frame_latham = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame_qb] * fps + [frame_latham] * fps
    with amp:
        batch, metas = yolo.preprocess(frames, size)
        # batch, metas = benchmark(yolo.preprocess, frames, size)
    
    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.no_grad():
        with amp:
            predictions = model(batch)
            # predictions = benchmark(model, batch)
            #print('predictions:', predictions.shape)
            cfg = dict(
                conf_thres = 0.0001,
                iou_thres = 0.1,
                agnostic = False,
                merge = True,
            )
            # dets = yolo.postprocess(predictions, metas, **cfg)
            dets = benchmark(yolo.postprocess, predictions, metas, **cfg)
    
    dtype = th.float32
    for dets_f in dets:
        if dets_f is not None:
            dtype = dets_f.dtype
            break
    dets = list(map(lambda det: th.empty(0, 6, dtype=dtype, device=param.device) if det is None else det, dets))
    #print('dets:', [tuple(dets_f.shape) for dets_f in dets])
    with th.no_grad():
        with amp:
            features = [feats.to(dets[0]) for feats in model.features]
            pooled = detector.pooler(features, dets, metas)
            # pooled = benchmark(detector.pooler, features, dets, metas)


@pytest.mark.parametrize("fps", [5, 10])
@pytest.mark.parametrize("amp", [True, False])
def test_model_pooling(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame_qb = th.randint(255, (720, 1056, 3), dtype=th.uint8).numpy()
    frame_latham = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame_qb] * fps + [frame_latham] * fps
    with amp:
        batch, metas = yolo.preprocess(frames, size)
        # batch, metas = benchmark(yolo.preprocess, frames, size)
    
    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.no_grad():
        with amp:
            predictions = model(batch)
            # predictions = benchmark(model, batch)
            #print('predictions:', predictions.shape)
            cfg = dict(
                conf_thres = 0.0001,
                iou_thres = 0.1,
                agnostic = False,
                merge = True,
            )
            dets = yolo.postprocess(predictions, metas, **cfg)
            # dets = benchmark(yolo.postprocess, predictions, metas, **cfg)
    
    dtype = th.float32
    for dets_f in dets:
        if dets_f is not None:
            dtype = dets_f.dtype
            break
    dets = list(map(lambda det: th.empty(0, 6, dtype=dtype, device=param.device) if det is None else det, dets))
    #print('dets:', [tuple(dets_f.shape) for dets_f in dets])
    with th.no_grad():
        with amp:
            features = [feats.to(dets[0]) for feats in model.features]
            # pooled = detector.pooler(features, dets, metas)
            pooled = benchmark(detector.pooler, features, dets, metas)


def test_torchjit(tile_img):
    # TODO 
    import torchvision
    # An instance of your model.
    model = torchvision.models.resnet18()

    logging.info(f"Compiling pretrained detection model")
    sm = torch.jit.script(model)
    logging.info(f"Compiled pretrained detection model")
    
    '''
    path = Path(path)
    img = cv.imread(path)
    img2 = cv.resize(img, scale=0.5)
    model_dir = None # "/tmp/ml/checkpoints"
    detector = yolo5x(pretrained=True, pooling=True, fuse=True, model_dir=model_dir, force_reload=not True)
    
    # dets, pooled = detector.detect([img, img2], size=640, conf_thres=0.4, iou_thres=0.5)
    dets, pooled = detector.detect([img, img2], size=736, conf_thres=0.35, iou_thres=0.5)
    dets, pooled = detector.detect([img, img2], size=736, conf_thres=0.001, iou_thres=0.65)
    features = detector.features
    print('images:', [(tuple(img.shape), img.mean()) for img in [img, img2]], 
            'dets:', [tuple(det.shape) for det in dets], 
            'pooled:', [tuple(feats.shape) for feats in pooled],
            'features:', [tuple(feats.shape) for feats in features])
    assert len(dets) == 2
    assert dets[0].shape[1] == 4+1+1
    cv.render(img, dets[0], score_thr=0.35, classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}-yolo5.jpg")
    cv.render(img2, dets[1], score_thr=0.35, classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}2-yolo5.jpg")
    '''