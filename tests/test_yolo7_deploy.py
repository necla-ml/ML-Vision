import pytest
import torch as th

from ml import deploy, logging
import ml.vision.transforms as T
import ml.vision.transforms.functional as TF
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
    return T.Compose([T.Resize(shape[1:]), T.ToTensor()])

@pytest.fixture
def batch(image, transform, batch_size, dev):
    return th.stack([transform(image)] * batch_size).to(th.uint8)

@pytest.fixture
def tag():
    return 'main'

@pytest.fixture
def name():
    return 'yolo7s'

@pytest.fixture
def detector(tag, dev):
    from ml.vision.models import yolo7s
    detector = yolo7s(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False)
    detector.eval()
    return detector.to(dev)

@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("strict", [True, False])
def test_deploy_trt(benchmark, batch, detector, dev, B, fp16, int8, strict, name):
    batch = TF.resize(batch, (384, 640)).float()
    h, w = batch.shape[2:]
    kwargs = {}
    if int8:
        import os
        from pathlib import Path
        from ml import hub
        from ml.vision.datasets.coco import download

        def preprocessor(size=(384, 640)):
            from PIL import Image
            from torchvision import transforms
            trans = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor()])

            H, W = size
            def preprocess(image_path, *shape):
                r'''Preprocessing for TensorRT calibration
                Args:
                    image_path(str): path to image
                    channels(int):
                '''
                image = Image.open(image_path)
                logging.debug(f"image.size={image.size}, mode={image.mode}")
                image = image.convert('RGB')
                C = len(image.mode)
                im = trans(image)
                assert im.shape == (C, H, W)
                return im

            return preprocess

        int8_calib_max = 5000
        int8_calib_batch_size = 64 
        cache = f'{name}-COCO2017-val-{int8_calib_max}-{int8_calib_batch_size}.cache'
        cache_path = Path(os.path.join(hub.get_dir(), cache))
        kwargs['int8_calib_cache'] = str(cache_path)
        kwargs['int8_calib_data'] = download(split='val2017', reload=False)
        kwargs['int8_calib_preprocess_func'] = preprocessor()
        kwargs['int8_calib_max'] = int8_calib_max
        kwargs['int8_calib_batch_size'] = int8_calib_batch_size

    engine = deploy.build(f"{name}-bs{B}_{h}x{w}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}",
                          detector,
                          [batch.shape[1:]],
                          backend='trt', 
                          reload=not True,
                          batch_size=B,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
                          onnx_simplify=True,
                          **kwargs
                          )
    
    preds, *features = benchmark(engine.predict, batch[:B].to(dev), sync=True)
    assert len(features) == 3
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=fp16):
            torch_preds, torch_features = detector(batch[:B].to(dev))
    logging.info(f"outputs trt norm={preds.norm().item()}, torch norm={torch_preds.norm().item()}")
    if fp16 or int8:
        pass
        # th.testing.assert_allclose(torch_preds.float(), preds.float(), rtol=2e-02, atol=4e-02)
    else:
        th.testing.assert_allclose(torch_preds.float(), preds.float(), rtol=1e-03, atol=4e-04)
        for torch_feats, feats in zip(torch_features, features):
            th.testing.assert_allclose(torch_feats.float(), feats.float(), rtol=1e-03, atol=4e-04)