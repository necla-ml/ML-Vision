from collections.abc import Sequence
import torch as th

from ml import cv

'''
In need of co-transformation on both input and target with consistent RNG.
'''

class ToCV(th.nn.Module):
    """Convert a ``PIL Image`` to numpy. This transform does not support torchscript.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks. See the `references`_ for implementing the transforms for image masks.
    .. _references: https://github.com/pytorch/vision/tree/master/references/segmentation
    """

    def __init__(self, format='BGR'):
        self.format = format

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if th.is_tensor(pic):
            return cv.fromTorch(pic)
        else:
            from PIL.Image import Image
            try:
                import accimage
            except Exception as e:
                accimage = None
            assert isinstance(pic, (Image, Image if accimage is None else accimage.Image)), "Assume input is a list of PIL.Image"
            assert pic.mode == 'RGB', "Assume input images in RGB format"
            return cv.pil_to_cv(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'

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
    """

    def __init__(self, size, constraint='shorter', interpolation=cv.INTER_LINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.constraint = constraint
        self.interpolation = interpolation

    def forward(self, img):
        """
        Args:
            img (PIL.Image, cv2 BGR HWC, RGB HWC Tensor): image to resize.
        Returns:
            Resized image in the same input format.
        """
        # return F.resize(img, self.size, self.interpolation)
        return cv.resize(img, self.size, constraint=self.constraint, interpolation=self.interpolation)

    def __repr__(self):
        from torchvision.transforms.transforms import _pil_interpolation_to_str
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)