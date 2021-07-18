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