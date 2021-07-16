
from pathlib import Path
import pytest

from torch.cuda.amp import autocast
import torch as th

from ml.vision.models import detr

from .fixtures import *

# @pytest.mark.essential
# @pytest.mark.parametrize("deformable", [False, True])
# @pytest.mark.parametrize("backbone", ['resnet50'])
# def test_detr(backbone, deformable):
#     from ml.vision.models.detection.detr import detr
#     print()
#     model = detr(pretrained=True, backbone=backbone, deformable=deformable, unload_after=True)
#     print(model.tag, model)

@pytest.fixture
def image():
    from PIL import Image 
    import requests
    import torchvision.transforms as T

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    transform = T.Compose([
        T.ToTensor()
    ])
    im = transform(im)
    return im


@pytest.mark.essential
@pytest.mark.parametrize("fps", [5, 10, 15, 20])
@pytest.mark.parametrize("amp", [False])
@pytest.mark.parametrize("deformable", [False])
@pytest.mark.parametrize("backbone", ['resnet50', 'resnet101'])
def test_detect(benchmark, fps, amp, deformable, backbone, image):
    # setup model
    from ml.vision.models.detection.detector import detr
    resize = 400
    detector = detr(
        pretrained=True,
        backbone=backbone,
        deformable=deformable,
        resize=resize
    )
    detector.eval()
    detector.to('cuda' if th.cuda.is_available() else 'cpu')

    # mean-std normalize the input image (batch-size: 1)
    frames = [image] * fps

    amp = autocast(enabled=amp)
    with th.no_grad():
        with amp:
            # dets, pooled = detector.detect(frames, **cfg)
            dets, rec = benchmark(detector.detect, frames)
            print(dets[0].shape)
