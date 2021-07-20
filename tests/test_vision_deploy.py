from pathlib import Path
import pytest
import torch as th

from ml import logging
from ml.vision.models import yolo5x
from ml.vision.datasets.coco import COCO80_CLASSES

from .fixtures import *

@pytest.mark.parametrize("amp", [True])
@pytest.mark.parametrize("int8", [True])
def test_yolo5(benchmark, tile_img, tag, amp, int8):
    from ml import cv
    path = Path(tile_img)
    img = cv.imread(path)
    img2 = cv.resize(img, scale=0.5)

    detector = yolo5x(pretrained=True, tag='v3.0', pooling=True, fuse=True, force_reload=not True)
    detector.to('cuda')
    assert detector.module.tag == tag

    detector.deploy('yolo5x',
                    batch_size=10,
                    spec=(3, 640, 640),
                    backend='trt',
                    reload=not True,
                    #dynamic_axes={'input_0': {0: 'batch_size'}},  # {0: 'batch_size', 2: 'height'}
                    amp=amp,
                    int8=int8
                    # min_shapes=[(3, 320, 640)],
                    # max_shapes=[(3, 640, 640)]
                    )
    dets, pooled = benchmark(detector.detect, [img, img2] * 5, size=640, cls_thres=0.4, nms_thres=0.5)
    # assert len(dets) == 30
    # assert dets[0].shape[1] == 4+1+1

    detector = yolo5x(pretrained=True, tag='v3.0', pooling=True, fuse=True, force_reload=not True)
    with th.no_grad():
        with th.cuda.amp.autocast(enabled=True):
            torch_dets, torch_pooled = detector.detect([img, img2], size=640, cls_thres=0.4, nms_thres=0.5)
    assert len(torch_dets) == 2
    assert torch_dets[0].shape[1] == 4+1+1
    for i, (torch_output, output) in enumerate(zip(torch_dets, dets)):
        logging.info(f"output[{i}] shape={tuple(output.shape)}, trt norm={output.norm()}, torch norm={torch_output.norm()}")
        #th.testing.assert_allclose(torch_output, output, rtol=1e-03, atol=3e-04)

    #cv.render(img, dets[0], score_thr=0.00035, classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}-yolo5_trt{amp and '_amp' or ''}.jpg")
    #cv.render(img2, dets[1], score_thr=0.00035, classes=COCO80_CLASSES, path=f"export/{path.name[:-4]}2-yolo5_trt{amp and '_amp' or ''}.jpg")
