"""Functional transformation depending on the input format.
"""
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def resize(img, size, interpolation=InterpolationMode.BILINEAR, constraint='shorter', antialias=False, **kwargs):
    '''Resize input image of torch tensor.
    Args:
        size(Tuple[int], int): tuple of height and width or length on both sides following torchvision resize semantics
    '''

    r"""Resize the input image to the given size.
    The image is is expected to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
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
        antialias (bool, optional): antialias flag. If ``img`` is PIL Image, the flag is ignored and anti-alias
            is always used. If ``img`` is Tensor, the flag is False by default and can be set to True for
            ``InterpolationMode.BILINEAR`` only mode. This can help making the output for PIL images and tensors
            closer.

            .. warning::
                There is no autodiff support for ``antialias=True`` option with input ``img`` as Tensor.
    Returns:
        Tensor: Resized image.
    """
    H, W = img.shape[-2:]
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
    else:
        h, w = size

    return TF.resize(img, (h, w), interpolation, antialias=antialias, **kwargs)