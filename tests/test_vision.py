from pathlib import Path
import pytest
import torch

from ml import logging
from ml.vision.models import yolo4, yolo5, yolo5l, yolo5x, rfcn
from ml.vision.datasets.coco import COCO80_CLASSES
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml.vision.ops import xyxys2xyxysc, xcycwh2xyxy, xcycwh2xywh, xyxy2xcycwh

from .fixtures import *

## Set operations on detections

@pytest.mark.essential
def test_dets_select():
    from ml.vision.ops import dets_select
    dets = torch.ones(4, 3)
    dets[:, -1] = torch.tensor([0,1,2,3])
    selection = dets_select(dets, [1, 3])
    assert (selection == torch.tensor([False, True, False, True])).all()

## Test Box format conversions

@pytest.mark.essential
def test_xyxys2xyxysc(xyxys):
    xyxysc = xyxys2xyxysc(xyxys)
    assert torch.is_tensor(xyxysc)
    assert xyxysc[xyxysc[:, -1] == 0].shape[0] == 3
    assert xyxysc[xyxysc[:, -1] == 1].shape[0] == 0
    assert xyxysc[xyxysc[:, -1] == 2].shape[0] == 3
    assert xyxysc[xyxysc[:, -1] == 3].shape[0] == 0
    for c, dets in enumerate(xyxys):
        assert dets is None or len(dets) == 0 or (dets == xyxysc[xyxysc[:, -1] == c][:, :5]).all()
    print()
    print('xyxys:', xyxys)
    print('xyxysc:', xyxysc)

@pytest.mark.essential
def test_xyxysc2xyxys(xyxysc):
    from ml.vision.ops.utils import xyxysc2xyxys
    xyxys = xyxysc2xyxys(xyxysc, 4)
    assert len(xyxys) == 4
    assert len(xyxys[1]) == 0
    assert len(xyxys[3]) == 0
    assert all(map(torch.is_tensor, xyxys))
    print()
    for c, dets in enumerate(xyxys):
        if c in [0, 2]:
            print(c, dets, xyxysc[xyxysc[:, -1] == c][:, :5])
            assert (dets == xyxysc[xyxysc[:, -1] == c][:, :5]).all()
        elif c in [1, 3]:
            assert len(dets) == 0
        else:
            assert False

@pytest.mark.essential
def test_xcycwh2xyxy(xcycwh):
    print()
    print("xcycwh:", xcycwh)
    xyxy = xcycwh2xyxy(xcycwh)
    print("xyxy:", xyxy)
    assert (xcycwh[:, 2] == xyxy[:, 2] - xyxy[:, 0] + 1).all()
    assert (xcycwh[:, 3] == xyxy[:, 3] - xyxy[:, 1] + 1).all()
    assert (xcycwh[:, 0] == (xyxy[:, 0] + xyxy[:, 2]) // 2).all()
    assert (xcycwh[:, 1] == (xyxy[:, 1] + xyxy[:, 3]) // 2).all()
    xcycwh2xyxy(xcycwh, inplace=True)
    assert (xyxy == xcycwh).all()
    
@pytest.mark.essential
def test_xcycwh2xywh(xcycwh):
    print()
    print("xcycwh:", xcycwh)
    xywh = xcycwh2xywh(xcycwh)
    print("xywh:", xywh)
    assert (xcycwh[:, 2] == xywh[:, 2]).all()
    assert (xcycwh[:, 3] == xywh[:, 3]).all()
    x2 = xywh[:, 0] + xywh[:, 2] - 1
    y2 = xywh[:, 1] + xywh[:, 3] - 1
    assert (xcycwh[:, 0] == (xywh[:, 0] + x2) // 2).all()
    assert (xcycwh[:, 1] == (xywh[:, 1] + y2) // 2).all()
    xcycwh2xywh(xcycwh, inplace=True)
    assert (xcycwh == xywh).all()

@pytest.mark.essential
def test_xyxy2xcycwh(xyxy):
    print()
    print("xyxy:", xyxy)
    xcycwh = xyxy2xcycwh(xyxy)
    print("xcycwh:", xcycwh)
    w = xyxy[:, 2] - xyxy[:, 0] + 1
    h = xyxy[:, 3] - xyxy[:, 1] + 1
    print("h:", h)
    assert (xcycwh[:, 2] == w).all()
    assert (xcycwh[:, 3] == h).all()
    xc = (xyxy[:, 0] + xyxy[:, 2]) // 2
    yc = (xyxy[:, 1] + xyxy[:, 3]) // 2
    assert (xcycwh[:, 0] == xc).all()
    assert (xcycwh[:, 1] == yc).all()
    xyxy2xcycwh(xyxy, inplace=True)
    assert (xcycwh == xyxy).all()

## Test YOLO

@pytest.mark.essential
def test_multiscale_fusion_align():
    from ml.vision import ops
    pooler = MultiScaleFusionRoIAlign(3)
    features = [
       torch.randn(2, 256, 76, 60),
       torch.randn(2, 512, 38, 30), 
       torch.randn(2, 1024, 19, 15)
    ]
    metas = [dict(
        shape=(1080, 810),
        offset=(0, (608-810/1080*608) % 64),
        ratio=(608/1080,)*2,
    ), dict(
        shape=(540, 405),
        offset=(0, (608-405/540*608) % 64),
        ratio=(608/540,)*2,
    )]

    boxes = torch.rand(6, 4) * 256
    boxes[:, 2:] += boxes[:, :2]
    rois = [boxes, boxes * 2]
    pooled = pooler(features, rois, metas)
    logging.info(f"RoI aligned features: {tuple(feats.shape for feats in pooled)}")
    assert list(pooled[0].shape) == [len(rois[0]), 1024+512+256, 3, 3]

# @pytest.mark.essential
def test_rfcn(tile_img):
    from ml import cv
    path = Path(tile_img)
    img = cv.imread(path)
    img2 = cv.resize(img, scale=0.5)
    img = cv.imread(path)
    model_dir = None # "/tmp/ml/checkpoints"
    detector = rfcn(pooling=2, model_dir=model_dir, force_reload=True)
    assert detector.with_rpn
    rois, dets, pooled = detector.detect(img, return_rpn=True)
    print('dets:', [tuple(det.shape) for det in dets], dets)
    print('rois:', [tuple(roi.shape) for roi in rois])
    print('pooled:', [tuple(feats.shape) for feats in pooled])
    cv.render(img, dets[0], score_thr=0.01, classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}-rfcn.jpg")

## Test Tracking

def test_yolo_deep_sort(video):
    import numpy as np
    from ml.vision.models.tracking.dsort import DeepSort
    from ml import av
    model, size = yolo4, 608
    model, size = yolo5x, 736
    detector = model(pretrained=True, fuse=True, pooling=True)
    pooler = MultiScaleFusionRoIAlign(3)
    tracker = DeepSort(max_feat_dist=0.2,
                       nn_budget=100, 
                       max_iou_dist=0.7,    # 0.7
                       max_age=15,          # 30 (FPS)
                       n_init=3)            # 3

    video = Path(video)
    if video.suffix in ['.mp4', '.avi']:
        s = av.open(video)
        v = s.decode(video=0)
        print(f"Tracking video: {video}")
    else:
        s = None
        if video.is_file():
            files = [video]
        elif video.is_dir():
            files = sorted([f for f in video.iterdir() if f.is_file()])
        v = [cv.imread(f) for f in files]
        print(f"Tracking {len(files)} frames in {video}")
    export = Path(f'export/{video.stem}-{model.__name__}')
    export.mkdir(parents=True, exist_ok=True)
    assert export.exists()

    print(f"Saving to {export / 'tracking.mp4'}")
    media = av.open(f"{export}/tracking.mp4", 'w')
    stream = media.add_stream('h264', 15)
    stream.bit_rate = 2000000
    for i, frame in enumerate(v):
        if not isinstance(frame, np.ndarray):
            frame = frame.to_rgb().to_ndarray()[:,:,::-1]
        
        if i == 0:
            stream.height = frame.shape[0]
            stream.width = frame.shape[1]
        dets, features = detector.detect([frame], size=size)
            
        # Track person only
        person = dets[0][:, -1] == 0
        persons = dets[0][person]
        features[0] = features[0][person]

        assert len(dets) == 1
        assert len(persons) == features[0].shape[0]
        assert dets[0].shape[1] == 4+1+1
        # assert features[0].shape[1] == 256+512+1024
        assert features[0].shape[1] == 320+640+1280

        if len(dets[0]) > 0:
            D = 1
            for s in features[0].shape[1:]:
                D *= s
            tracker.update(persons, features[0].view(len(features[0]), D))
            if i < 60:
                logging.info(f"[{i}] dets[0]: {dets[0].shape}, feats: {[tuple(feats.shape) for feats in features]}")
                cv.render(frame, dets[0], path=export / 'dets' / f"frame{i:03d}.jpg")
            else:
                break

        snapshot = tracker.snapshot()
        logging.info(f"[{i}] snapshot[0]: {snapshot and list(zip(*snapshot))[0] or len(snapshot)}")
        frame = cv.render(frame, snapshot, path=f"export/{video.stem}-{model.__name__}/tracking/frame{i:03d}.jpg")
        if media is not None:
            shape = frame.shape
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packets = stream.encode(frame)
            print('encoded:', packets, frame)
            media.mux(packets)
    if media is not None:
        packets = stream.encode(None)
        media.mux(packets)
        media.close()

def test_yolo5x_store_deep_sort(video):
    import numpy as np
    from ml.vision.models.tracking.dsort import DeepSort
    from ml.vision.datasets.widerperson import WIDERPERSON_CLASSES
    WIDERPERSON_CLASSES[0] = 'object'
    model, size = yolo5, 736
    detector = model(name='yolov5x-store', pretrained=True, bucket='eigen-pretrained', key='detection/yolo/yolov5x-store.pt',
                    classes=len(WIDERPERSON_CLASSES), pooling=True, fuse=True, model_dir=None, force_reload=not True)
    pooler = MultiScaleFusionRoIAlign(3)
    tracker = DeepSort(max_feat_dist=0.2,
                       nn_budget=100, 
                       max_iou_dist=0.7,    # 0.7
                       max_age=15,          # 30 (FPS)
                       n_init=3)            # 3

    from ml import av
    s = av.open(video)
    v = s.decode()
    video = Path(video)
    export = Path(f'export/{video.stem}-{model.__name__}')
    export.mkdir(exist_ok=True)
    assert export.exists()

    print(f"Tracking video: {video}")
    print(f"Saving to {export / 'tracking.mp4'}")
    media = av.open(f"{export}/tracking.mp4", 'w')
    stream = media.add_stream('h264', 15)
    stream.bit_rate = 2000000
    for i, frame in enumerate(v):
        if i == 0:
            stream.height = frame.height
            stream.width = frame.width

        frame = frame.to_rgb().to_ndarray()[:,:,::-1]
        dets, features = detector.detect([frame], size=size)

        # Track person only
        person = (0 < dets[0][:, -1]) & (dets[0][:, -1] < 4)
        persons = dets[0][person]
        features[0] = features[0][person]

        assert len(dets) == 1
        assert len(persons) == features[0].shape[0]
        assert dets[0].shape[1] == 4+1+1
        assert features[0].shape[1] == 320+640+1280
        if len(dets[0]) > 0:
            D = 1
            for s in features[0].shape[1:]:
                D *= s
            tracker.update(persons, features[0].view(len(features[0]), D))
            if i < 60:
                logging.info(f"[{i}] dets[0]: {dets[0].shape}, feats: {[tuple(feats.shape) for feats in features]}")
                cv.render(frame, dets[0], classes=WIDERPERSON_CLASSES, path=export / 'dets' / f"frame{i:03d}.jpg")
            else:
                break
        
        snapshot = tracker.snapshot()
        logging.info(f"[{i}] snapshot[0]: {snapshot and list(zip(*snapshot))[0] or len(snapshot)}")
        frame = cv.render(frame, snapshot, classes=WIDERPERSON_CLASSES, path=f"export/{video.stem}-{model.__name__}/tracking/frame{i:03d}.jpg")
        #frame = detector.render(frame, snapshot)

        if media is not None:
            shape = frame.shape
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packets = stream.encode(frame)
            print('encoded:', packets, frame)
            media.mux(packets)
    if media is not None:
        packets = stream.encode(None)
        media.mux(packets)
        media.close()

def test_rfcn_deep_sort(video):
    import numpy as np
    from ml.vision.models.tracking.dsort import DeepSort
    model, size = rfcn, 608
    detector = model(pooling=2, model_dir="/tmp/checkpoints", force_reload=not True)
    tracker = DeepSort(max_feat_dist=0.2,
                       nn_budget=100, 
                       max_iou_dist=0.7,    # 0.7
                       max_age=15,          # 30 (FPS)
                       n_init=3)            # 3

    from ml import av
    s = av.open(video)
    v = s.decode()
    video = Path(video)
    export = Path(f'export/{video.stem}-{model.__name__}')
    export.mkdir(exist_ok=True)
    assert export.exists()

    print(f"Tracking video: {video}")
    print(f"Saving to {export / 'tracking.mp4'}")
    media = av.open(f"{export}/tracking.mp4", 'w')
    stream = media.add_stream('h264', 15)
    stream.bit_rate = 2000000
    for i, frame in enumerate(v):
        if i == 0:
            stream.height = frame.height
            stream.width = frame.width

        frame = frame.to_rgb().to_ndarray()[:,:,::-1]
        dets, features = detector.detect([frame], size=size)
        if True:
            # Track person only
            person = dets[0][:, -1] == 0
            dets[0] = dets[0][person]
            features[0] = features[0][person]

        assert len(dets) == 1
        assert len(dets[0]) == features[0].shape[0]
        assert dets[0].shape[1] == 4+1+1
        # assert features[0].shape[1] == 256+512+1024
        assert features[0].shape[1] == 1024

        if len(dets[0]) > 0:
            D = 1
            for s in features[0].shape[1:]:
                D *= s
            tracker.update(dets[0], features[0].view(len(features[0]), D))
            if i < 60:
                logging.info(f"[{i}] dets[0]: {dets[0].shape}, feats: {[tuple(feats.shape) for feats in features]}")
                detector.render(frame, dets[0], path=export / 'dets' / f"frame{i:03d}.jpg")
            else:
                break
        
        snapshot = tracker.snapshot()
        logging.info(f"[{i}] snapshot[0]: {snapshot and list(zip(*snapshot))[0] or len(snapshot)}")
        frame = detector.render(frame, snapshot, path=f"export/{video.stem}-{model.__name__}/tracking/frame{i:03d}.jpg")
        #frame = detector.render(frame, snapshot)

        if media is not None:
            shape = frame.shape
            frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
            packets = stream.encode(frame)
            print('encoded:', packets, frame)
            media.mux(packets)
    if media is not None:
        packets = stream.encode(None)
        media.mux(packets)
        media.close()