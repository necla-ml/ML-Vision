import pytest

from torch.cuda.amp import autocast
import torch as th

from ml.vision.models import yolo7x
from ml.vision.models.detection import yolo

from .fixtures import *

@pytest.fixture
def tag():
    return 'main'

@pytest.fixture
def url():
    return 'https://hbr.org/resources/images/article_assets/2015/03/MAR15_18_91531630.jpg'

@pytest.fixture
def shape():
    return 3, 720, 1280

@pytest.fixture
def batch_size():
    return 16

@pytest.fixture
def image(url):
    from PIL import Image
    import requests
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@pytest.fixture
def transform(shape):
    import ml.vision.transforms as T 
    return T.Compose([T.Resize(shape[1:]), T.ToTensor()])

@pytest.fixture
def batch(image, transform, batch_size):
    return th.stack([transform(image)] * batch_size).to(th.uint8)

@pytest.fixture
def detector(tag):
    detector = yolo7x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False, unload_after=True)
    detector.eval()
    return detector.to('cuda' if th.cuda.is_available() else 'cpu')

def test_model_eval(benchmark, detector):
    def detect(detector):
        mode = detector.training
        detector.eval()
        detector.train(mode)
    benchmark(detect, detector)

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True])
@pytest.mark.parametrize("batch_preprocess", [True, False])
def test_detect(benchmark, batch, detector, fps, amp, batch_preprocess):
    if batch_preprocess:
        frames = batch[:fps]
    else:
        frames = [batch[0].permute(1, 2, 0).numpy()] * fps
    cfg = dict(
        cls_thres = 0.0001,
        # cls_thres = 1,
        nms_thres = 0.1,
        agnostic = False,
        merge = True,
        batch_preprocess = batch_preprocess
    )
    
    amp = autocast(enabled=amp)
    with th.no_grad():
        with amp:
            dets, pooled = benchmark(detector.detect, frames, amp=amp, **cfg)
            print(dets[0].shape, dets[0].dtype, pooled[0].shape, pooled[0].dtype)

@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True, False])
def test_model_batched_preprocess(benchmark, amp, fps):
    # NOTE: benchmark behaves oddly when input frames are explicitly moved to GPU
    amp = autocast(enabled=amp)
    size = 640
    frame = th.randint(
        255, (3, 720, 1280),
        dtype=th.uint8,
        #device=th.device('cuda' if th.cuda.is_available() else 'cpu')
    )
    frames = th.stack([frame] * fps)
    with amp:
        batch, metas = benchmark(yolo.batched_preprocess, frames, size)
    assert batch.size(0) == fps

@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True, False])
def test_model_preprocess(benchmark, amp, fps):
    amp = autocast(enabled=amp)
    size = 640
    frame = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame] * fps
    with amp:
        batch, metas = benchmark(yolo.preprocess, frames, size)
    assert batch.size(0) == fps

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True, False])
def test_model_forward(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame] * fps
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


@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True, False])
def test_model_postprocess(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame= th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame] * fps
    with amp:
        batch, metas = yolo.preprocess(frames, size)
    
    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.no_grad():
        with amp:
            predictions = model(batch)[0]
            cfg = dict(
                conf_thres = 0.0001,
                iou_thres = 0.1,
                agnostic = False,
                merge = True,
            )
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


@pytest.mark.parametrize("fps", [8, 16])
@pytest.mark.parametrize("amp", [True, False])
def test_model_pooling(benchmark, detector, fps, amp):
    amp = autocast(enabled=amp)
    size = 640
    frame = th.randint(255, (720, 1280, 3), dtype=th.uint8).numpy()
    frames = [frame] * fps
    with amp:
        batch, metas = yolo.preprocess(frames, size)
    
    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.no_grad():
        with amp:
            predictions = model(batch)[0]
            cfg = dict(
                conf_thres = 0.0001,
                iou_thres = 0.1,
                agnostic = False,
                merge = True,
            )
            dets = yolo.postprocess(predictions, metas, **cfg)
    
    dtype = th.float32
    for dets_f in dets:
        if dets_f is not None:
            dtype = dets_f.dtype
            break
    dets = list(map(lambda det: th.empty(0, 6, dtype=dtype, device=param.device) if det is None else det, dets))
    with th.no_grad():
        with amp:
            features = [feats.to(dets[0]) for feats in model.features]
            pooled = benchmark(detector.pooler, features, dets, metas)

