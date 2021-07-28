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

TAG_SZ = {
    'v1.0':736,
    'v2.0':672,
    'v3.0':640,
    'v5.0':640,
}

@pytest.fixture
def tag():
    # YOLOv5 version tag
    return 'v5.0'

@pytest.fixture
def detector(tag):
    detector = yolo5x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False)
    assert detector.tag == tag
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
            predictions = model(batch)[0]
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
            predictions = model(batch)[0]
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
    import torch
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

# FIXME: not maintained since YOLOv5, to remove
def test_yolo4(tile_img):
    from ml import cv
    path = tile_img
    path = Path(path)
    img = cv.imread(path)
    img2 = cv.resize(img, scale=0.5)
    detector = yolo4(pooling=True, fuse=True)
    (dets, pooled), features = detector.detect([img, img2], size=608), detector.features
    print('images:', [(tuple(img.shape), img.mean()) for img in [img, img2]], 
            'dets:', [tuple(det.shape) for det in dets], 
            'pooled:', [tuple(feats.shape) for feats in pooled],
            'features:', [tuple(feats.shape) for feats in features])
    assert len(dets) == 2
    assert dets[0].shape[1] == 4+1+1
    detector.render(img, dets[0], classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}-yolo4.jpg")
    detector.render(img2, dets[1], classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}2-yolo4.jpg")

# @pytest.mark.essential
def test_detection_tv(detector, tile_img):
    from ml.av import io, utils
    from ml.av.transforms import functional as TF
    path = Path(tile_img)
    img = io.load(path)
    h, w = img.shape[-2:]
    img2 = TF.resize(img, (h//2, w//2))
    # print(detector)
    
    dets, pooled = detector.detect([img, img2], size=TAG_SZ[detector.tag], cls_thres=0.49, nms_thres=0.5)
    # dets, pooled = detector.detect([img, img2], size=sz, cls_thres=0.35, nms_thres=0.5)
    # dets, pooled = detector.detect([img, img2], size=sz, cls_thres=0.01, nms_thres=0.65)
    features = detector.features
    '''
    print('images:', [(tuple(img.shape), img.mean()) for img in [img, img2]], 
            'dets:', [tuple(det.shape) for det in dets], 
            'pooled:', [tuple(feats.shape) for feats in pooled],
            'features:', [tuple(feats.shape) for feats in features])
    print(dets)
    '''
    assert len(dets) == 2
    assert dets[0].shape[1] == 4+1+1

    dets0, dets1 = dets[0], dets[1]
    labels0 = [f"{COCO80_CLASSES[int(c)]} {s:.2f}" for s, c in dets0[:, -2:]]
    labels1 = [f"{COCO80_CLASSES[int(c)]} {s:.2f}" for s, c in dets1[:, -2:]]
    print(f"lables0: {labels0}")
    print(f"lables1: {labels1}")
    img = utils.draw_bounding_boxes(img, dets0, labels=COCO80_CLASSES)
    img2 = utils.draw_bounding_boxes(img2, dets1, labels=COCO80_CLASSES)
    io.save(img, f"export/{path.name[:-4]}-yolo5.png")
    io.save(img2, f"export/{path.name[:-4]}2-yolo5.png")

'''
def test_yolo5_store(sku_img, wp_img):
    from ml.vision.datasets.widerperson import WIDERPERSON_CLASSES
    WIDERPERSON_CLASSES[0] = 'object'
    sku_img, wp_img = Path(sku_img), Path(wp_img)
    img = cv.imread(sku_img)
    img2 = cv.imread(wp_img)
    model_dir = None # "/tmp/ml/checkpoints"
    detector = yolo5(name='yolov5x-store', pretrained=True, bucket='eigen-pretrained', key='detection/yolo/yolov5x-store.pt',
                    classes=len(WIDERPERSON_CLASSES), pooling=True, fuse=True, model_dir=model_dir, force_reload=not True)
    # dets, pooled = detector.detect([img, img2], size=640, conf_thres=0.4, iou_thres=0.5)
    dets, pooled = detector.detect([img, img2], size=736, conf_thres=0.35, iou_thres=0.5)
    # dets, pooled = detector.detect([img, img2], size=736, conf_thres=0.001, iou_thres=0.65)
    features = detector.features
    print('images:', [(tuple(img.shape), img.mean()) for img in [img, img2]], 
            'dets:', [tuple(det.shape) for det in dets], 
            'pooled:', [tuple(feats.shape) for feats in pooled],
            'features:', [tuple(feats.shape) for feats in features])
    assert len(dets) == 2
    assert dets[0].shape[1] == 4+1+1
    cv.render(img, dets[0], score_thr=0.35, classes=WIDERPERSON_CLASSES, path=f"export/{sku_img.name[:-4]}-yolo5.jpg")
    cv.render(img2, dets[1], score_thr=0.35, classes=WIDERPERSON_CLASSES, path=f"export/{wp_img.name[:-4]}2-yolo5.jpg")
'''
