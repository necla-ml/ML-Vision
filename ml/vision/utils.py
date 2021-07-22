from typing import Tuple, Union, List, Optional

from torch.nn import functional as F
from torchvision.utils import (
    make_grid,
    save_image,
    draw_bounding_boxes,        # 0.9.0+
    # draw_segmentation_masks,    # 0.10.0+
)
import torch

from ml import logging
from .transforms import functional as TF

@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    with_coordinates: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Make a grid of images.

    Params:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``1``.
        padding (int, optional): amount of padding. Default: ``50``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
        with_coordinates (bool, optional): if True, return cell coordinates in the grid
    Returns:
        tuple of grid image and list of coordinates of individual images (tuple(Tensor, list(tuple)))
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (th.is_tensor(tensor) or
            (isinstance(tensor, list) and all(th.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = th.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = th.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = th.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    # NOTE: if uncommented, list with single image will not be padded
    # if tensor.size(0) == 1:
    #     return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = py_min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    coordinates = []
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#th.Tensor.copy_
            x1, y1 = x * width + padding,  y * height + padding
            x2, y2 =  x1 + tensor.size(3), y1 + tensor.size(2)
            coordinates.append((x1, y1, x2, y2))
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    
    return grid, coordinates if with_coordinates else grid

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    r"""Resize and pad to a multiple of strides for object detection.

    src: utils.augmentation.letterbox
    
    Args:
        im (Tensor[CHW, dtype=uint8], ndarray[HWC, dtype=uint8]): RGB tensor or BGR cv2 in numpy HWC
        new_shape (int, Tuple[int, int]): target size of resize and padding
        color (Tuple[int, int, int]): color to pad the borders
        auto (bool): True forr minimal rectangle
        scaleFill: stretch to fill

    Returns:
        resized (Tensor[CHW, uint8], ndarray[HWC, uint8]): scaled image with potential padding to a multiple of 32 w.r.t. the longer side
        meta (Dict): { original shape, offset, ratio } 
    """
    
    # Resize and pad image while meeting stride-multiple constraints
    if TF.is_tensor(im):         
        shape = im.shape[-2:]
    elif TF.is_cv2(im):
        shape = im.shape[:2]
    else:
        raise ValueError(f"image type and format not Tensor or CV2")
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) w.r.t longer edge
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:                                # minimum padding to rectangle
        dw, dh = dw % stride, dh % stride   # wh padding
    elif scaleFill:                         # stretch w/o padding
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[0] / shape[0], new_shape[1] / shape[1]  # rH, rW

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if TF.is_tensor(im):
        if shape[::-1] != new_unpad:  # resize
            im = TF.resize(im, new_unpad[::-1], interpolation=TF.InterpolationMode.BILINEAR)
        im = F.pad(im, mode='constant', pad=(left, right, top, bottom), value=sum(color)/len(color) if isinstance(color, tuple) else color).div(255)
    else:
        import cv2
        from .. import cv
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # border padding
        im = cv.toTorch(im)
        
    return im, dict(
        shape=shape,        # HxW
        offset=(top, left), # offH, offW
        ratio=ratio,        # rH, rW
    )