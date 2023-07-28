from typing import Sequence

import torch as th
from torchvision.transforms import InterpolationMode

from ml.vision.transforms import functional as F

class Resize(th.nn.Module):
    """Resize the input image to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number if lossy is True.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            Otherwise, the longer edge is matched.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` only mode. This can help making the output for PIL images and tensors
            closer.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.
    """

    def __init__(self, size, constraint='shorter', interpolation=InterpolationMode.BILINEAR, antialias=False):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.constraint = constraint
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL.Image, cv2 BGR HWC, RGB HWC Tensor): image to resize.
        Returns:
            Resized image in the same input format.
        """
        return F.resize(img, self.size, constraint=self.constraint, interpolation=self.interpolation, antialias=self.antialias)

    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1}), constraint={2}, antialias={3}'.format(self.size, interpolate_str, self.constraint, self.antialias)