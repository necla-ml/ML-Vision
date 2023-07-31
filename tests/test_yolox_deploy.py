# import pycuda.autoinit
import pytest
import numpy as np
import torch as th

from ml import deploy, logging
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from ml.vision.models.detection import yolox

from .fixtures import *

@pytest.fixture
def batch_size():
    return 32

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

# @pytest.mark.essential
@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("shape", [(640, 640)])
def test_deploy_onnx(benchmark, name, batch, detector, dev, B, shape):
    spec = [[3, *shape]]
    engine = deploy.build(name,
                          detector,
                          spec=spec,
                          backend='onnx', 
                          reload=True)
    
    batch, metas = yolox.preprocess(batch, input_size=shape)
    outputs = benchmark(engine.predict, batch[:B])
    meta_preds, *features = outputs
    with th.inference_mode():
        torch_meta_preds, torch_features = detector(batch[:B].to(dev))
    
    for torch_preds, preds in zip(torch_meta_preds, meta_preds):
        np.testing.assert_allclose(torch_preds.cpu().numpy(), preds, rtol=1e-03, atol=3e-04)
        th.testing.assert_close(torch_preds, th.from_numpy(preds).to(dev), rtol=1e-03, atol=3e-04)
    for torch_feats, feats in zip(torch_features, features):
        np.testing.assert_allclose(torch_feats.cpu().numpy(), feats, rtol=1e-03, atol=3e-04)
        th.testing.assert_close(torch_feats, th.from_numpy(feats).to(dev), rtol=1e-03, atol=3e-04)

# @pytest.mark.essential
@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("int8", [False])
@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("shape", [(640, 640)])
def test_deploy_trt(benchmark, batch, detector, dev, B, fp16, int8, strict, shape, name):
    # preprocess
    batch, metas = yolox.preprocess(batch, input_size=shape)
    h, w = batch.shape[2:]
    kwargs = {}
    if int8:
        import os
        from pathlib import Path
        from ml import hub
        from ml.vision.datasets.coco import download

        def preprocessor(size=(640, 640)):
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
        kwargs['int8_calib_preprocess_func'] = preprocessor(shape)
        kwargs['int8_calib_max'] = int8_calib_max
        kwargs['int8_calib_batch_size'] = int8_calib_batch_size

    spec = [[3, *shape]]
    engine = deploy.build(f"{name}-bs{B}_{h}x{w}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}",
                          detector,
                          spec=spec,
                          backend='trt', 
                          reload=not True,
                          batch_size=B,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
                          **kwargs
                          )
    
    preds, *features = benchmark(engine.predict, batch[:B].to(dev), sync=True)
    with th.inference_mode():
        with th.cuda.amp.autocast(enabled=fp16):
            torch_preds, torch_features = detector(batch[:B].to(dev))
    logging.info(f"outputs trt norm={preds.norm().item()}, torch norm={torch_preds.norm().item()}")
    if fp16 or int8:
        pass
    else:
        th.testing.assert_close(torch_preds.float(), preds.float(), rtol=1e-03, atol=4e-04)
        for torch_feats, feats in zip(torch_features, features):
            th.testing.assert_close(torch_feats.float(), feats.float(), rtol=1e-03, atol=4e-04)

# @pytest.mark.essential
@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("batch_preprocess", [True, False])
@pytest.mark.parametrize('fp16', [True, False])
@pytest.mark.parametrize("int8", [False])
@pytest.mark.parametrize("shape", [(640, 640)])
def test_detect_trt(benchmark, name, batch, detector, B, batch_preprocess, fp16, int8, shape):
    if batch_preprocess:
        frames = batch[:B]
    else:
        frames = [batch[0]] * B
    cfg = dict(
        cls_thres = 0.0001,
        # cls_thres = 1,
        nms_thres = 0.1,
        agnostic = False,
        merge = True,
        batch_preprocess = batch_preprocess
    )
    spec = [3, *shape]
    detector.deploy(name,
                    batch_size=B,
                    spec=spec,
                    fp16=fp16,
                    int8=int8,
                    backend='trt',
                    reload=not True)
    
    dets = benchmark(detector.detect, frames, **cfg)

    assert len(dets) == len(frames)
    # TODO: assert output with torch output