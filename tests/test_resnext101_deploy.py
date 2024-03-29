import pytest
import numpy as np
import torch as th
from ml import (
    deploy, 
    logging
)

@pytest.fixture
def batch_size():
    return 16

@pytest.fixture
def shape():
    return 3, 224, 224

@pytest.fixture
def dev():
    return th.device('cuda') if th.cuda.is_available() else th.device('cpu')

@pytest.fixture
def args(shape, dev):
    return th.rand(1, *shape, device=dev)

@pytest.fixture
def batch(batch_size, shape):
    return th.rand(batch_size, *shape)

@pytest.fixture
def x101_32x8d(dev):
    from torchvision.models import get_model_weights
    from torchvision.models.resnet import _resnet
    from torchvision.models.resnet import Bottleneck
    from torchvision.ops.misc import FrozenBatchNorm2d
    kwargs = {}
    frozen = True
    kwargs['groups'] = gs = kwargs.get('groups', 32)
    kwargs['width_per_group'] = gw = kwargs.get('width_per_group', 8)
    kwargs['norm_layer'] = kwargs.get('norm_layer', FrozenBatchNorm2d if frozen else None)
    arch = f"resnext101_{gs}x{gw}d"
    weights = get_model_weights(arch).DEFAULT
    model = _resnet(Bottleneck, [3, 4, 23, 3], weights, True, **kwargs)
    model.to(dev).eval()
    print(model)
    return model

@pytest.fixture
def x101_32x8d_wsl(dev):
    from torchvision.ops.misc import FrozenBatchNorm2d
    kwargs = {}
    frozen = True
    kwargs['groups'] = gs = kwargs.get('groups', 32)
    kwargs['width_per_group'] = gw = kwargs.get('width_per_group', 8)
    kwargs['norm_layer'] = kwargs.get('norm_layer', FrozenBatchNorm2d if frozen else None)
    model = th.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl', **kwargs)
    model.to(dev).eval()
    return model

@pytest.fixture
def backbone_x101_32x8d_wsl(dev):
    from ml.vision.models.backbone import resnext101_wsl
    model = resnext101_wsl(groups=32, width_per_group=8)
    model.to(dev).eval()
    return model

@pytest.mark.parametrize("B", [2])
def test_x101_amp(benchmark, x101_32x8d, dev, batch, B):
    model = x101_32x8d
    with th.inference_mode():
        with th.cuda.amp.autocast(enabled=False):
            outputs_fp32 = model(batch[:B].to(dev)).float()
        with th.cuda.amp.autocast():
            outputs_amp = model(batch[:B].to(dev)).float()

    for i, (output_fp32, output_amp) in enumerate(zip(outputs_fp32, outputs_amp)):
        logging.info(f"output[{i}] shape={tuple(output_fp32.shape)}, norm_fp32={output_fp32.norm()}, norm_amp={output_amp.norm()}")
        th.testing.assert_close(output_amp, output_fp32, rtol=2e-02, atol=8e-03)

@pytest.mark.parametrize("B", [2])
def test_x101_wsl_amp(benchmark, x101_32x8d_wsl, dev, batch, B):
    model = x101_32x8d_wsl
    with th.inference_mode():
        with th.cuda.amp.autocast(enabled=False):
            outputs_fp32 = model(batch[:B].to(dev)).float()
        with th.cuda.amp.autocast():
            outputs_amp = model(batch[:B].to(dev)).float()
    
    # for i, (output_fp32, output_amp) in enumerate(zip(outputs_fp32, outputs_amp)):
    #     logging.info(f"output[{i}] shape={tuple(output_fp32.shape)}, norm_fp32={output_fp32.norm()}, norm_amp={output_amp.norm()}")
    #     th.testing.assert_close(output_amp, output_fp32, rtol=2e-02, atol=8e-03)

@pytest.mark.parametrize("B", [2])
def test_backbone_x101_wsl_amp(benchmark, backbone_x101_32x8d_wsl, dev, batch, B):
    model = backbone_x101_32x8d_wsl
    with th.inference_mode():
        with th.cuda.amp.autocast(enabled=False):
            outputs_fp32 = model(batch[:B].to(dev))#[1:2]
        with th.cuda.amp.autocast():
            outputs_amp = model(batch[:B].to(dev))#[1:2]
    #assert len(outputs_fp32) == 5
    #assert len(outputs_amp) == 5
    for i, (output_fp32, output_amp) in enumerate(zip(outputs_fp32, outputs_amp)):
        # logging.info(f"output[{i}] shape={tuple(output_fp32.shape)}, norm_fp32={output_fp32.norm(dim=(2,3)).norm(dim=1)}, norm_amp={output_amp.norm(dim=(2,3)).norm(dim=1)}")
        logging.info(f"output[{i}] shape={tuple(output_fp32.shape)}, norm_fp32={output_fp32.norm()}, norm_amp={output_amp.norm()}")
        th.testing.assert_close(output_amp, output_fp32, rtol=2.5e-01, atol=9e-1)

@pytest.mark.parametrize("B", [1, 2])
def test_deploy_onnx(benchmark, backbone_x101_32x8d_wsl, dev, batch, B):
    engine = deploy.build('resnext101_32x8d_wsl',
                            backbone_x101_32x8d_wsl,
                            [batch.shape[1:]],
                            backend='onnx', 
                            reload=True)
    
    outputs = benchmark(engine.predict, batch[:B])
    spatial_feats, scene_feats = outputs[-2][:B], outputs[-1][:B]
    assert spatial_feats.shape == (B, 2048, 7, 7)
    assert scene_feats.shape == (B, 2048)
    with th.inference_mode():
        torch_outputs = backbone_x101_32x8d_wsl(batch[:B].to(dev))
    for i, (torch_output, output) in enumerate(zip(torch_outputs, outputs)):
        # logging.info(f"output[{i}] shape={tuple(output.shape)}")
        np.testing.assert_allclose(torch_output.cpu().numpy(), output, rtol=1e-03, atol=3e-04)
        th.testing.assert_close(torch_output, th.from_numpy(output).to(dev), rtol=1e-03, atol=3e-04)

@pytest.mark.parametrize("B", [16])
@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.parametrize("int8", [False])
@pytest.mark.parametrize("strict", [False])
@pytest.mark.parametrize("max_inp_size", [(224, 224)])
@pytest.mark.parametrize("min_inp_size", [(224, 224)])
def test_deploy_trt(benchmark, batch, backbone_x101_32x8d_wsl, dev, B, fp16, int8, strict, min_inp_size, max_inp_size):
    from ml import hub
    # dynamic/static input
    minH, minW = min_inp_size
    maxH, maxW = max_inp_size
    min_shapes = [(3, minH, minW)]
    max_shapes = [(3, maxH, maxW)]
    spec = [[3, minH, minW]]
    dynamic_axes={'input_0': {0: 'batch_size'}}
    if maxH != minH:
        spec[0][1] = -1
        dynamic_axes['input_0'][2] = 'height'
    if maxW != minW:
        spec[0][2] = -1
        dynamic_axes['input_0'][3] = 'width'

    # int 8 configs
    int8_calib_data = 'data/ILSVRC2012/val'
    int8_calib_max = 5000 * 4
    int8_calib_batch_size = max(B, 512 * 4)
    prefix = 'x101_32x8d_wsl'
    name = f"{prefix}-bs{B}_{maxW}x{maxH}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}{strict and '_strict' or ''}"
    cache = f"ILSVRC2012-val-{int8_calib_batch_size}-{int8_calib_max}"
    int8_calib_cache = f"{hub.get_dir()}/{cache}.cache"
    engine = deploy.build(name,
                          backbone_x101_32x8d_wsl,
                          spec=spec,
                          backend='trt',
                          reload=not True,
                          batch_size=B,
                          dynamic_axes=dynamic_axes,
                          min_shapes=min_shapes,
                          max_shapes=max_shapes,
                          fp16=fp16,
                          int8=int8,
                          strict_type_constraints=strict,
                          int8_calib_cache=int8_calib_cache,
                          int8_calib_data=int8_calib_data,
                          int8_calib_max=int8_calib_max,
                          int8_calib_batch_size=int8_calib_batch_size)

    outputs = benchmark(engine.predict, batch[:B].to(dev), sync=True)
    spatial_feats, scene_feats = outputs[-2], outputs[-1]
    assert len(outputs) == 5
    # assert spatial_feats.shape == (B, 2048, 23, 40), f""
    assert scene_feats.shape == (B, 2048)
    with th.inference_mode():
        with th.cuda.amp.autocast(enabled=fp16):
            torch_outputs = backbone_x101_32x8d_wsl(batch[:B].to(dev))
    for i, (torch_output, output) in enumerate(zip(torch_outputs, outputs)):
        logging.info(f"output[{i}] shape={tuple(output.shape)}, trt norm={output.norm()}, torch norm={torch_output.norm()}")
        if fp16:
            if int8:
                th.testing.assert_close(torch_output, output, rtol=15.2, atol=15.2)
            else:
                th.testing.assert_close(torch_output, output, rtol=1.9, atol=1.9)
        else:
            if int8:
                th.testing.assert_close(torch_output, output, rtol=15.2, atol=15.2)
            else:
                th.testing.assert_close(torch_output, output, rtol=1e-03, atol=3e-04)
