import pytest
import torch as th

@pytest.fixture
def vid():
    return th.randint(0, 255, (10, 1080, 810, 3), dtype=th.uint8)

@pytest.fixture
def xyxy():
    boxes = th.randint(100, (4, 4))
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes

@pytest.fixture
def xcycwh(xyxy):
    w = xyxy[:, 2] - xyxy[:, 0] + 1 
    h = xyxy[:, 3] - xyxy[:, 1] + 1 
    xyxy[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) // 2
    xyxy[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) // 2
    xyxy[:, 2] = w
    xyxy[:, 3] = h
    return xyxy

@pytest.fixture
def xyxys():
    dets = [th.randn(3, 5) for c in range(4)]
    dets[0][:, 2:4] += dets[0][:, 0:2]
    dets[2][:, 2:4] += dets[2][:, 0:2]
    dets[1] = None
    dets[3] = th.randn(0, 5)
    return dets

@pytest.fixture
def xyxysc():
    dets = []
    for c in range(4):
        boxes = th.randn(3, 6) * 1280
        boxes[:, -1] = c
        boxes[:, 2:4] += boxes[:, 0:2]
        dets.append(boxes)
    dets[1] = th.randn(0, 6)
    dets[3] = th.randn(0, 6)
    return th.cat(dets)

@pytest.fixture
def tile_img():
    return 'assets/bus_zidane_tiles.jpg'
    # TODO download and cache test images
    # return '../yolov3/data/samples/tiles.jpg'
    #return '../yolov3/data/samples/bus.jpg'
    #return '../yolov3/data/samples/zidane.jpg'
    #return '/zdata/projects/shared/datasets/WiderPerson/Images/005014.jpg'
    #return '/zdata/projects/shared/datasets/SKU110K/images/test_100.jpg'

@pytest.fixture
def sku_img():
    'assets/sku110k-test_100.jpg',

@pytest.fixture
def wp_img():
    'assets/wider_person-005014.jpg',

@pytest.fixture
def img():
    return th.randint(0, 255, (3, 1080, 810), dtype=th.uint8)
    
@pytest.fixture
def video():
    import os
    return '/zdata/projects/shared/datasets/kinetics400/frames-5fps/val/abseiling/GwlcmI36imo_000127_000137'
    return os.path.join(os.environ['HOME'], 'Videos', 'GwlcmI36imo_000127_000137.mp4')
    return os.path.join(os.environ['HOME'], 'Videos', 'store720p-short.264')
    return os.path.join(os.environ['HOME'], 'Videos', 'calstore-concealing.mp4')

@pytest.fixture
def model_dir():
    return '/tmp/ml'