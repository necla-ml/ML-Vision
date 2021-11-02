# import pycuda.autoinit
import pytest
import numpy as np
import torch as th

from ml import deploy, logging
import ml.vision.transforms as T
import ml.vision.transforms.functional as TF
from .fixtures import *

@pytest.fixture
def batch_size():
    return 40

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
    return 'v6.0'

@pytest.fixture
def name():
    return 'yolo5x'

@pytest.fixture
def detector(tag, dev):
    from ml.vision.models import yolo5x
    detector = yolo5x(pretrained=True, tag=tag, pooling=1, fuse=True, force_reload=False)
    assert detector.module.tag == tag
    detector.eval()
    return detector.to(dev)

@pytest.mark.parametrize("B", [1])
def test_deploy_onnx(benchmark, batch, detector, dev, B):
    module = detector.module
    module.model[-1].export = True
    engine = deploy.build('yolo5x',
                          detector,
                          [batch.shape[1:]],
                          backend='onnx', 
                          reload=True)
    
    #outputs = engine.predict(batch[:B])
    #for output in outputs:
    #    print(output.shape)
    outputs = benchmark(engine.predict, batch[:B])
    # print('outputs:', [o.shape for o in outputs])
    meta_preds, features = outputs[0:3], outputs[3:]
    with th.no_grad():
        torch_meta_preds, torch_features = detector(batch[:B].to(dev))
        # print('torch:', [o.shape for o in torch_meta_preds], [feats.shape for feats in torch_features])
    # logging.info(f"outputs onnx shape={tuple(outputs[0].shape)}, torch shape={tuple(torch_outputs.shape)}")
    
    for torch_preds, preds in zip(torch_meta_preds, meta_preds):
        np.testing.assert_allclose(torch_preds.cpu().numpy(), preds, rtol=1e-03, atol=3e-04)
        th.testing.assert_allclose(torch_preds, th.from_numpy(preds).to(dev), rtol=1e-03, atol=3e-04)
    for torch_feats, feats in zip(torch_features, features):
        np.testing.assert_allclose(torch_feats.cpu().numpy(), feats, rtol=1e-03, atol=3e-04)
        th.testing.assert_allclose(torch_feats, th.from_numpy(feats).to(dev), rtol=1e-03, atol=3e-04)

@pytest.mark.parametrize("B", [8])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("int8", [True, False])
@pytest.mark.parametrize("strict", [True, False])
def test_deploy_trt(benchmark, batch, detector, dev, B, fp16, int8, strict, name):
    # FIXME pytorch cuda initialization must be ahead of pycuda
    module = detector.module
    module.model[-1].export = True
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

    engine = deploy.build(f"yolo5x-bs{B}_{h}x{w}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}",
                          detector,
                          [batch.shape[1:]],
                          backend='trt', 
                          reload=not True,
                          batch_size=B,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
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

@pytest.mark.parametrize("B", [40])
@pytest.mark.parametrize("batch_preprocess", [True, False])
@pytest.mark.parametrize('fp16', [True, False])
@pytest.mark.parametrize("int8", [True, False])
def test_detect_trt(benchmark, batch, detector, B, batch_preprocess, fp16, int8):
    if batch_preprocess:
        frames = batch[:B]
    else:
        frames = [batch[0].permute(1, 2, 0).numpy()] * B
    cfg = dict(
        cls_thres = 0.0001,
        # cls_thres = 1,
        nms_thres = 0.1,
        agnostic = False,
        merge = True,
        batch_preprocess = batch_preprocess
    )
    spec = [3, 384, 640]
    detector.deploy('yolo5x',
                    batch_size=B,
                    spec=spec,
                    fp16=fp16,
                    int8=int8,
                    backend='trt',
                    reload=not True)
    
    dets, pooled = benchmark(detector.detect, frames, **cfg)

# @pytest.mark.essential
def test_detection_tv(detector, tile_img, B=5, fp16=True):
    from pathlib import Path
    from ml.av import io, utils
    from ml.av.transforms import functional as TF
    from ml.vision.datasets.coco import COCO80_CLASSES
    path = Path(tile_img)
    img = io.load(path)
    h, w = img.shape[-2:]
    
    module = detector.module
    module.model[-1].export = True
    """
    engine = deploy.build(f"yolo5x-bs{B}_{h}x{w}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}",
                          detector,
                          [img.shape],
                          backend='trt', 
                          reload=not True,
                          batch_size=B,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
                          )
    """
    import math
    scale = YOLO5_TAG_SZ[detector.tag]
    if w > h:
        spec = (3, 32 * math.ceil(h / w * scale / 32), scale)
    else:
        spec = (3, scale, 32 * math.ceil(w / h * scale / 32))
    print(f"spec={spec}") 
    detector.deploy('yolo5x', 
                    batch_size=B, 
                    spec=spec, 
                    fp16=fp16, 
                    backend='trt', 
                    reload=not True)
    dets, pooled = detector.detect([img], size=scale, cls_thres=0.49, nms_thres=0.5)
    assert len(dets) == 1
    assert dets[0].shape[1] == 4+1+1

    dets0 = dets[0]
    labels0 = [f"{COCO80_CLASSES[int(c)]} {s:.2f}" for s, c in dets0[:, -2:]]
    print(f"lables0: {labels0}")
    img = utils.draw_bounding_boxes(img, dets0, labels=COCO80_CLASSES)
    io.save(img, f"export/{path.name[:-4]}-yolo5x_trt.png")