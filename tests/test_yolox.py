import torch 
from torch.cuda.amp import autocast

import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ml.vision.models.detection.yolox import preprocess, postprocess

from .fixtures import *

@pytest.fixture
def batch_size():
    return 16

@pytest.fixture
def shape():
    return 3, 720, 1280

@pytest.fixture
def dev():
    return th.device('cuda' if th.cuda.is_available() else 'cpu')

@pytest.fixture
def args(shape, dev):
    return th.rand(1, *shape, device=dev)

@pytest.fixture
def url():
    return 'https://hbr.org/resources/images/article_assets/2015/03/MAR15_18_91531630.jpg'

@pytest.fixture
def image(url):
    from PIL import Image
    import requests
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@pytest.fixture
def transform(shape): 
    return T.Compose([T.Lambda(lambda x: TF.pil_to_tensor(x)), T.Resize(shape[1:], antialias=True)])

@pytest.fixture
def batch(image, transform, batch_size, dev):
    return th.stack([transform(image)] * batch_size).to(th.uint8)

@pytest.fixture
def tag():
    return 'main'

@pytest.fixture
def name():
    return 'yolox'

@pytest.fixture
def detector(tag, dev):
    from ml.vision.models import yolox_x
    detector = yolox_x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False)
    assert detector.module.tag == tag
    detector.eval()
    return detector.to(dev)

def test_model_eval(benchmark, detector):
    def detect(detector):
        mode = detector.training
        detector.eval()
        detector.train(mode)
    benchmark(detect, detector)

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("batch_preprocess", [True, False])
def test_detect(benchmark, batch, detector, fps, amp, batch_preprocess):
    if batch_preprocess:
        frames = batch[:fps]
    else:
        frames = [batch[0]] * fps
    cfg = dict(
        cls_thres = 0.0001,
        # cls_thres = 1,
        nms_thres = 0.1,
        agnostic = False,
        merge = True,
        batch_preprocess = batch_preprocess
    )
    
    amp = autocast(enabled=amp)
    with th.inference_mode():
        with amp:
            dets = benchmark(detector.detect, frames, amp=amp, **cfg)
            print(dets[0].shape, dets[0].dtype)

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("size", [(640, 640)])
def test_model_preprocess(benchmark, amp, fps, size, batch):
    amp = autocast(enabled=amp)
    with amp:
        batch, metas = benchmark(preprocess, batch[:fps], size)
    assert batch.size(0) == fps

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("size", [(640, 640)])
def test_model_forward(benchmark, detector, fps, amp, size, batch):
    amp = autocast(enabled=amp)
    with amp:
        batch, metas = preprocess(batch[:fps], size)
        # batch, metas = benchmark(yolo.preprocess, frames, size)

    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.inference_mode():
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

@pytest.mark.essential
@pytest.mark.parametrize("fps", [8])
@pytest.mark.parametrize("amp", [True, False])
@pytest.mark.parametrize("size", [(640, 640)])
def test_model_postprocess(benchmark, detector, fps, amp, size, batch):
    amp = autocast(enabled=amp)
    with amp:
        batch, metas = preprocess(batch[:fps], size)
    
    model = detector.module
    param = next(model.parameters())
    batch = batch.to(param)
    with th.inference_mode():
        with amp:
            predictions = model(batch)[0]
            cfg = dict(
                conf_thre = 0.0001,
                nms_thre = 0.1,
                class_agnostic = False,
            )
            dets = benchmark(postprocess, predictions, metas, **cfg)

@pytest.mark.essential
def test_yolox_preprocess(batch):
    # Test when the input is a single tensor
    img = batch[:2]
    input_size = (640, 640)
    pad_value = 114
    resized_img, ratio = preprocess(img, input_size, pad_value)

    assert isinstance(resized_img, torch.Tensor)
    assert isinstance(ratio, float)
    assert resized_img.shape[-2:] == input_size

    # Test when the input is a list of tensors
    img_list = [
        torch.randn(3, 500, 600),  # Random 3-channel image with height=500 and width=600
        torch.randn(3, 300, 200),  # Random 3-channel image with height=300 and width=200
        torch.randn(3, 700, 800),  # Random 3-channel image with height=700 and width=800
    ]
    resized_img, ratio = preprocess(img_list, input_size, pad_value)

    assert isinstance(resized_img, torch.Tensor)
    assert len(resized_img) == len(img_list)
    assert all(resized_img[i].shape[-2:] == input_size for i in range(len(img_list)))