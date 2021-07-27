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
import numpy as np
import torch as th
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import *

try:
    import accimage
except Exception as e:
    accimage = None

def is_tensor(img):
    return th.is_tensor(img) and img.shape[0] in (1, 3) and img.dtype == th.uint8

def is_cv2(img):
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            return img.shape[-1] in (1, 3) and img.dtype == np.uint8
        else:
            return img.ndim == 2 and img.dtype == np.uint8

def is_accimage(img):
    return accimage is not None and isinstance(img, accimage.Image)

def is_pil(img):
    from ml.av.backend.pil import Image
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def resize(img, size, interpolation=InterpolationMode.BILINEAR, constraint='shorter', **kwargs):
    '''Resize input image of PIL/accimage, OpenCV BGR or torch tensor.
    Args:
        size(Tuple[int], int): tuple of height and width or length on both sides following torchvision resize semantics
    '''

    r"""Resize the input image to the given size.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        img (PIL Image or Tensor or cv2 numpy): Image to be resized.
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
    if is_tensor(img):
        H, W = img.shape[-2:]
    elif is_cv2(img):
        H, W = img.shape[:2]
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
            if H < W:
                w, h = int(W / H * size), size
            else:
                w, h = size, int(H / W * size)
            # return TF.resize(img, size, interpolation, **kwargs)
    else:
        h, w = size
    
    from ml import logging
    if is_cv2(img):
        from ml.av.backend import opencv as cv
        logging.info(f"cv img shape={tuple(img.shape)}, dtype={img.dtype}, sum={img.sum(axis=(0,1))}")
        return cv.resize(img, (w, h), interpolation=cv.INTER_LINEAR)
    else:
        if is_tensor(img):
            logging.info(f"tensor img shape={tuple(img.shape)}, dtype={img.dtype}, sum={img.sum(dim=(1,2))}")
        else:
            import numpy as np
            img_np = np.array(img)
            logging.info(f"PIL img shape={tuple(img_np.shape)}, dtype={img_np.dtype}, sum={img_np.sum(axis=(0,1))}")
        return TF.resize(img, (h, w), interpolation, **kwargs)