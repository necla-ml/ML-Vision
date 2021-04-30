import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from ...extension import _lazy_import
from ... import math
from .utils import boxes2rois

## Adapted from torchvision

class _RoIPoolFunction(Function):
    @staticmethod
    def forward(ctx, input, rois, output_size, spatial_scale):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.input_shape = input.size()
        _C = _lazy_import()
        output, argmax = _C.roi_pool_forward(
            input, rois, *spatial_scale, *output_size)
        ctx.save_for_backward(rois, argmax)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        bs, ch, h, w = ctx.input_shape
        _C = _lazy_import()
        grad_input = _C.roi_pool_backward(
            grad_output, rois, argmax, *spatial_scale, *output_size,
            bs, ch, h, w)
        return grad_input, None, None, None

def roi_pool(input, boxes, output_size, spatial_scale=1.0):
    """
    Performs Region of Interest (RoI) Pool operator described in Fast R-CNN
    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in x1,y1,x2,y2
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float | tuple): a scaling factor that maps the input coordinates to
            the box coordinates where both sides may take different scales. Default: 1.0
    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    rois = boxes
    if not isinstance(rois, th.Tensor):
        rois = boxes2rois(rois)

    if type(spatial_scale) not in [tuple, list]:
        spatial_scale = float(spatial_scale)
        spatial_scale = (spatial_scale, spatial_scale)

    return _RoIPoolFunction.apply(input, rois, output_size, spatial_scale)

class RoIPool(nn.Module):
    """
    See roi_pool
    """
    def __init__(self, output_size, spatial_scale):
        super(RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input, rois):
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr

## Pure Python implementation below

def roi_pool_pth(batch, rois, output_size, spatial_scale=1):
    r"""Allow RoI pooling to have different scales on both sides and follow float32 precision as PyTorch.
    """

    if type(spatial_scale) is float:
        spatial_scale_x, spatial_scale_y = spatial_scale, spatial_scale
    else:
        spatial_scale_x, spatial_scale_y = spatial_scale

    if not th.is_tensor(rois):
        rois = boxes2rois(rois)
    
    pooled_height, pooled_width = output_size
    output = th.zeros(len(rois), batch.shape[1], *output_size, device=batch.device)
    for i, roi in enumerate(rois):
        b = int(roi[0].item())
        features = batch[b]
        H, W = features.shape[-2:]
        x1, y1, x2, y2 = roi[1:]
        
        # XXX py3 rounding to even while py2 rounding half up as C++/CUDA implementations
        roi_start_w = math.round(x1.item() * spatial_scale_x)
        roi_start_h = math.round(y1.item() * spatial_scale_y)
        roi_end_w = math.round(x2.item() * spatial_scale_x)
        roi_end_h = math.round(y2.item() * spatial_scale_y)

        # RoI width/height >= 1
        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        roi_height = max(roi_end_h - roi_start_h + 1, 1)
        bin_size_h = th.tensor(roi_height, dtype=th.float32).div_(pooled_height)
        bin_size_w = th.tensor(roi_width, dtype=th.float32).div_(pooled_width)
        for ph in range(pooled_height):
            hstart = math.floor(ph * bin_size_h)
            hend = math.ceil((ph + 1) * bin_size_h)
            hstart = min(H, max(0, hstart + roi_start_h))
            hend = min(H, max(0, hend + roi_start_h))
            for pw in range(pooled_width):
                wstart = math.floor(pw * bin_size_w)
                wend = math.ceil((pw + 1) * bin_size_w)
                wstart = min(W, max(0, wstart + roi_start_w))
                wend = min(W, max(0, wend + roi_start_w))

                # empty when ROI start at the limit
                if hend <= hstart or wend <= wstart:
                    output[i, :, ph, pw] = 0
                else:
                    output[i, :, ph, pw] = features[:, hstart:hend, wstart:wend].max(dim=-1)[0].max(dim=-1)[0]
    return output


def roi_pool_ma(batch, rois, output_size, spatial_scale=1):
    r"""Allow RoI pooling to have different scales on both sides and use double precision.
    """

    if type(spatial_scale) is float:
        spatial_scale_x, spatial_scale_y = spatial_scale, spatial_scale
    else:
        spatial_scale_x, spatial_scale_y = spatial_scale

    if not th.is_tensor(rois):
        rois = boxes2rois(rois)
    
    pooled_height, pooled_width = output_size
    output = th.zeros(len(rois), batch.shape[1], *output_size, device=batch.device)
    for i, roi in enumerate(rois):
        b = int(roi[0].item())
        features = batch[b]
        H, W = features.shape[-2:]
        x1, y1, x2, y2 = roi[1:]
        
        # XXX py3 rounding to even while py2 rounding half up as C++/CUDA implementations
        roi_start_w = round(x1.item() * spatial_scale_x)
        roi_start_h = round(y1.item() * spatial_scale_y)
        roi_end_w = round(x2.item() * spatial_scale_x)
        roi_end_h = round(y2.item() * spatial_scale_y)

        # RoI width/height >= 1
        roi_width = max(roi_end_w - roi_start_w + 1, 1)
        roi_height = max(roi_end_h - roi_start_h + 1, 1)
        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width
        for ph in range(pooled_height):
            hstart = math.floor(ph * bin_size_h)
            hend = math.ceil((ph + 1) * bin_size_h)
            hstart = min(H, max(0, hstart + roi_start_h))
            hend = min(H, max(0, hend + roi_start_h))
            for pw in range(pooled_width):
                wstart = math.floor(pw * bin_size_w)
                wend = math.ceil((pw + 1) * bin_size_w)
                wstart = min(W, max(0, wstart + roi_start_w))
                wend = min(W, max(0, wend + roi_start_w))

                # empty when ROI start at the limit
                if hend <= hstart or wend <= wstart:
                    output[i, :, ph, pw] = 0
                else:
                    output[i, :, ph, pw] = features[:, hstart:hend, wstart:wend].max(dim=-1)[0].max(dim=-1)[0]
    return output


def roi_pool_clip(batch, rois, output_size, spatial_scale=1):
    r"""Allow RoI pooling to have different scales on both sides.
    """

    if type(spatial_scale) is float:
        spatial_scale_x, spatial_scale_y = spatial_scale, spatial_scale
    else:
        spatial_scale_x, spatial_scale_y = spatial_scale

    if not th.is_tensor(rois):
        rois = boxes2rois(rois)
    
    output = []
    for roi in rois:
        b = int(roi[0])
        features = batch[b]
        H, W = features.shape[-2:]
        x1, y1, x2, y2 = roi[1:]
        roi_start_w = round(x1.item() * spatial_scale_x)
        roi_start_h = round(y1.item() * spatial_scale_y)
        roi_end_w = round(x2.item() * spatial_scale_x)
        roi_end_h = round(y2.item() * spatial_scale_y)

        roi_start_w = max(0, roi_start_w)
        roi_start_w = min(W, roi_start_w)
        roi_start_h = max(0, roi_start_h)
        roi_start_h = min(H, roi_start_h)
        roi_end_w = max(0, roi_end_w)
        roi_end_w = min(W, roi_end_w)
        roi_end_h = max(0, roi_end_h)
        roi_end_h = min(H, roi_end_h)

        #print(f"{features.shape}, ({y1*spatial_scale_y:.2f}, {y2*spatial_scale_y:.2f}, {x1*spatial_scale_x:.2f}, {x2*spatial_scale_x:.2f}), ({roi_start_h}, {roi_end_h}, {roi_start_w}, {roi_end_w})")
        if roi_start_h >= H or roi_start_w >= W:
            # XXX nothing to pool
            pooled = th.zeros((features.shape[0], *output_size), device=batch.device)
        else:
            RoI = features[:, roi_start_h:roi_end_h+1, roi_start_w:roi_end_w+1]
            pooled = F.adaptive_max_pool2d(RoI, output_size)
        output.append(pooled)
    return th.stack(output)


# TODO
# - batch processing
# - lack of GPU version
# - inconsistent with PyTorch version in boundary conditions
class RoIPoolMa(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale_x, spatial_scale_y):
        super(RoIPoolMa, self).__init__()
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale_x = float(spatial_scale_x)
        self.spatial_scale_y = float(spatial_scale_y)
        #self.adaptive_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, feature, rois, debug=False):
        num_channels, data_height, data_width = feature.size()
        num_rois = rois.shape[0]
        outputs = th.zeros(num_rois, num_channels, self.pooled_height, self.pooled_width)

        for roi_ind, roi in enumerate(rois):
            # np.round() to even using Banker's algorithm different from C++/CUDA rounding away from zero
            # roi and spatial_scale are double in Python not float32 in C++/CUDA
            roi_start_w = np.round(roi[0] * self.spatial_scale_x).astype(int)
            roi_start_h = np.round(roi[1] * self.spatial_scale_y).astype(int)
            roi_end_w = np.round(roi[2] * self.spatial_scale_x).astype(int)
            roi_end_h = np.round(roi[3] * self.spatial_scale_y).astype(int)

            # pooling start/end difference >= 1
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)
            bin_size_w = float(roi_width) / float(self.pooled_width)
            bin_size_h = float(roi_height) / float(self.pooled_height)

            if debug and roi_ind in [1]:
                print(f"ROI: {roi}")
                print(f"Scaled ROI: ({roi[1]*self.spatial_scale_y:.2f}, {roi[3]*self.spatial_scale_y:.2f}, "
                                    f"{roi[0]*self.spatial_scale_x:.2f}, {roi[2]*self.spatial_scale_x:.2f}), "
                                    f"({roi_start_h}, {roi_end_h}, {roi_start_w}, {roi_end_w})")

            for ph in range(self.pooled_height):
                hstart = int(np.floor(ph * bin_size_h))
                hend = int(np.ceil((ph + 1) * bin_size_h))
                hstart = min(data_height, max(0, hstart + roi_start_h))
                hend = min(data_height, max(0, hend + roi_start_h))
                for pw in range(self.pooled_width):
                    wstart = int(np.floor(pw * bin_size_w))
                    wend = int(np.ceil((pw + 1) * bin_size_w))
                    wstart = min(data_width, max(0, wstart + roi_start_w))
                    wend = min(data_width, max(0, wend + roi_start_w))

                    # empty when ROI start at the limit
                    is_empty = (hend <= hstart) or(wend <= wstart)
                    if is_empty:
                        outputs[roi_ind, :, ph, pw] = 0
                    else:
                        outputs[roi_ind, :, ph, pw] = th.max(
                            th.max(feature[:, hstart:hend, wstart:wend], 1, keepdim=True)[0], 2, keepdim=True)[0].view(-1)

                    if debug and roi_ind in [1]:
                        #print(f"ROI: {roi}")
                        #print(f"Scaled ROI: ({roi[1]*self.spatial_scale_y:.2f}, {roi[3]*self.spatial_scale_y:.2f}, "
                        #                    f"{roi[0]*self.spatial_scale_x:.2f}, {roi[2]*self.spatial_scale_x:.2f}), "
                        #                    f"({roi_start_h}, {roi_end_h}, {roi_start_w}, {roi_end_w})")
                        #print(f"[{roi_ind}, :, {ph}, {pw}]={outputs[roi_ind, :, ph, pw]}")
                        #print(f"[{roi_ind}][{ph},{pw}] ({hstart}, {hend}, {wstart}, {wend}), ({bin_size_h:.2f}, {bin_size_w:.2f})") 
                        #print(f"feats: {feature.shape}, {feature[:, hstart:hend, wstart:wend].view(-1)}")
                        pass

        #outputs = self.adaptive_max_pool(outputs).squeeze()
        return outputs