import torch
import torch.nn.functional as F
import pytest

from .fixtures import xyxy

@pytest.fixture
def roi_align_module():
    from ml.vision.ops import MultiScaleFusionRoIAlign
    output_size = (7, 7)
    sampling_ratio = -1
    aligned = False
    return MultiScaleFusionRoIAlign(output_size, sampling_ratio=sampling_ratio, aligned=aligned)

@pytest.fixture
def boxes(xyxy):
    return [xyxy.float(), xyxy.float()]

@pytest.fixture
def metas():
    metas = [{"ratio": (1.0, 1.0), "offset": (0, 0), "shape": (64, 64)},
             {"ratio": (2.0, 2.0), "offset": (0, 0), "shape": (32, 32)}]
    return metas

@pytest.fixture
def x():
    x = [torch.randn(1, 3, 64, 64), torch.randn(1, 3, 32, 32)]
    return x

@pytest.fixture(params=[False, True])
def autocast(request):
    return request.param

@pytest.mark.essential
def test_forward_on_cuda(roi_align_module, x, boxes, metas, autocast):
    # Move fixtures to cuda
    x = [t.to("cuda") for t in x]
    boxes = [box.to("cuda") for box in boxes]

    with torch.cuda.amp.autocast(enabled=autocast):
        aligned = roi_align_module.forward(x, boxes, metas)
    assert len(aligned) == len(boxes)
    assert all(feature.is_cuda for feature in aligned)  # Asserting CUDA tensors
    # Add more assertions as per your requirements

@pytest.mark.essential
def test_forward_on_cpu(roi_align_module, x, boxes, metas):
    # Move fixtures to CPU
    x_cpu = [t.to("cpu") for t in x]
    boxes_cpu = [box.to("cpu") for box in boxes]

    aligned = roi_align_module.forward(x_cpu, boxes_cpu, metas)
    assert len(aligned) == len(boxes_cpu)
    assert all(not feature.is_cuda for feature in aligned)  # Asserting CPU tensors
    # Add more assertions as per your requirements

@pytest.mark.essential
def test_benchmark_forward_on_cuda(benchmark, roi_align_module, x, boxes, metas, autocast):
    # Move fixtures to cuda
    x = [t.to("cuda") for t in x]
    boxes = [box.to("cuda") for box in boxes]

    def forward_fn():
        torch.cuda.synchronize()
        aligned = roi_align_module.forward(x, boxes, metas)
        torch.cuda.synchronize()
        return aligned

    with torch.cuda.amp.autocast(enabled=autocast):
        aligned = benchmark(forward_fn)

    assert len(aligned) == len(boxes)
    assert all(feature.is_cuda for feature in aligned)  # Asserting CUDA tensors
    # Add more assertions as per your requirements

@pytest.mark.essential
def test_benchmark_forward_on_cpu(benchmark, roi_align_module, x, boxes, metas):
    # Move fixtures to CPU
    x_cpu = [t.to("cpu") for t in x]
    boxes_cpu = [box.to("cpu") for box in boxes]

    def forward_fn():
        aligned = roi_align_module.forward(x_cpu, boxes_cpu, metas)
        return aligned

    aligned = benchmark(forward_fn)

    assert len(aligned) == len(boxes_cpu)
    assert all(not feature.is_cuda for feature in aligned)  # Asserting CPU tensors
    # Add more assertions as per your requirements