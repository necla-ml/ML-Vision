# import pycuda.autoinit
import pytest
import numpy as np
import torch as th

from ml import deploy, logging
# from ml.vision.models import yolo4, yolo5, yolo5l, yolo5x, rfcn
from ml.vision.models.detection import yolo
from ml.vision.datasets.coco import COCO80_CLASSES
from ml.vision.ops import MultiScaleFusionRoIAlign
from ml.vision.ops import xyxys2xyxysc, xcycwh2xyxy, xcycwh2xywh, xyxy2xcycwh

from .fixtures import *

@pytest.fixture
def batch_size():
    return 10

@pytest.fixture
def shape():
    return 3, 384, 640
    return 3, 640, 640

@pytest.fixture
def dev():
    return th.device('cuda' if th.cuda.is_available() else 'cpu')

@pytest.fixture
def args(shape, dev):
    return th.rand(1, *shape, device=dev)

@pytest.fixture
def batch(batch_size, shape, dev):
    return th.rand(batch_size, *shape)

@pytest.fixture
def tag():
    return 'v5.0'

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

@pytest.mark.parametrize("B", [5, 10])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("int8", [False])
@pytest.mark.parametrize("strict", [False])
def test_deploy_trt(benchmark, batch, detector, dev, B, fp16, int8, strict):
    # FIXME pytorch cuda initialization must be ahead of pycuda
    module = detector.module
    module.model[-1].export = True
    h, w = batch.shape[2:]
    engine = deploy.build(f"yolo5x-bs{B}_{h}x{w}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}",
                          detector,
                          [batch.shape[1:]],
                          backend='trt', 
                          reload=not True,
                          batch_size=B,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
                          )

    outputs = benchmark(engine.predict, batch[:B].to(dev), sync=True)
    # print('outputs:', [output.shape for output in outputs])
    meta_preds, features = outputs[:3], outputs[3:]
    assert len(outputs) == 6
    assert len(meta_preds) == 3
    assert len(features) == 3
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=fp16):
            torch_meta_preds, torch_features = detector(batch[:B].to(dev))
            # print('torch:', [o.shape for o in torch_meta_preds], [feats.shape for feats in torch_features])
    logging.info(f"outputs trt norm={[preds.norm().item() for preds in meta_preds]}, torch norm={[preds.norm().item() for preds in torch_meta_preds]}")
    if fp16:
        #th.testing.assert_allclose(torch_output, th.from_numpy(output[:B]).to(dev), rtol=1e-02, atol=3e-02)
        pass
    else:
        for torch_preds, preds in zip(torch_meta_preds, meta_preds):
            th.testing.assert_allclose(torch_preds.float(), preds.float(), rtol=1e-03, atol=4e-04)
        for torch_feats, feats in zip(torch_features, features):
            th.testing.assert_allclose(torch_feats.float(), feats.float(), rtol=1e-03, atol=4e-04)

# @pytest.mark.essential
def test_detection_tv(detector, tile_img, B=5, fp16=True):
    from pathlib import Path
    from ml.av import io, utils
    from ml.av.transforms import functional as TF
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