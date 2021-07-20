"""Functional transformation depending on the input format.

Supported formats:
- Tensor (RGB, CHW)
- PIL (RGB, HWC)
- cv2 (BGR, HWC)

Conversions:
- toTensor
- toPIL
- toCV
"""

from typing import Union
from enum import Enum
from ml import logging
from torchvision.transforms.functional import *
import torch as th

try:
    import accimage
except Exception as e:
    accimage = None

def is_tensor(img):
    return th.is_tensor(img) and img.shape[0] in (1, 3) and img.dtype == th.uint8

def is_accimage(img):
    return accimage is not None and isinstance(img, accimage.Image)

def is_pil(img):
    from ml.av.backend.pil import Image
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def resize(img: Tensor, size: Union[int, Tuple[int, int]], constraint: str='shorter', interpolation: InterpolationMode=InterpolationMode.BILINEAR, **kwargs) -> Tensor:
    '''Resize input image of PIL/accimage, OpenCV BGR or torch tensor.

    Args:
        size(Tuple[int], int): tuple of height and width or length on both sides following torchvision resize semantics
    '''

    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        img (PIL Image or Tensor): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is ``InterpolationMode.BILINEAR``. If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        constraint(str): resize by the shorter (ImageNet) or longer edge (YOLO)
    Returns:
        PIL Image or Tensor: Resized image.
    """
    from torchvision.transforms import functional as F
    if is_tensor(img):
        H, W = img.shape[-2:]
    else:
        # PIL
        W, H = img.size
    if isinstance(size, int):
        # with aspect ratio preserved
        if constraint == 'longer':
            if H < W:
                h, w = int(H / W * size), size
            else:
                h, w = size, int(W / H * size)
        else:
            return F.resize(img, size, interpolation, **kwargs)
    else:
        h, w = size
    return F.resize(img, (h, w), interpolation, **kwargs)

def letterbox(img, size=640, color=114, minimal=True, stretch=False, upscaling=True):
    """Resize and pad to the new shape.
    Args:
        img(BGR): CV2 BGR image
        size[416 | 512 | 608 | 32*]: target long side to resize to in multiples of 32
        color(tuple): Padding color
        minimal(bool): Padding up to the short side or not
        stretch(bool): Scale the short side without keeping the aspect ratio
        upscaling(bool): Allows to scale up or not
    """
    # Resize image to a multiple of 32 pixels on both sides 
    # https://github.com/ultralytics/yolov3/issues/232
    color = isinstance(color, int) and (color,) * img.shape[-1] or color
    shape = img.shape[:2]
    if isinstance(size, int):
        size = (size, size)

    r = py_min(size[0] / shape[0], size[1] / shape[1])
    if not upscaling: 
        # Only scale down but no scaling up for better test mAP
        r = py_min(r, 1.0)

    # Compute padding
    ratio = r, r
    pw = int(round(shape[1] * r))
    ph = int(round(shape[0] * r))
    new_unpad = pw, ph  # actual size to scale to (w, h)
    dw, dh = size[1] - new_unpad[0], size[0] - new_unpad[1]         # padding on sides

    if minimal: 
        # Padding up to 64 for the short side
        dw, dh = dw % 64, dh % 64
    elif stretch:  
        # Stretch the short side to the exact target size
        dw, dh = 0.0, 0.0
        new_unpad = size
        ratio = size[0] / shape[0], size[1] / shape[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = resize(img, (new_unpad[::-1]))

    # Fractional to integral padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    resized = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return resized, dict(
        shape=shape,        # HxW
        offset=(top, left), # H, W
        ratio=ratio,        # H, W
    )