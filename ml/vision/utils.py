import random
from colorsys import rgb_to_hsv
from typing import Tuple, Union, List, Optional

from torch.nn import functional as F
from torchvision.utils import (
    make_grid,
    save_image,
    draw_bounding_boxes,        # 0.9.0+
    draw_segmentation_masks,    # 0.10.0+
)
import torch as th
import numpy as np

from ml import logging
import ml.vision.transforms as T
from .transforms import functional as TF

BLACK    = (  0,   0,   0)
BLUE     = (255,   0,   0)
GREEN    = (  0, 255,   0)
RED      = (  0,   0, 255)
MAROON   = (  0,   0, 128)
YELLOW   = (  0, 255, 255)
WHITE    = (255, 255, 255)
FG       = GREEN
BG       = BLACK
COLORS91 = [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(91)]

PALETTE_RGB = [
    (204,73,196),
    (100,205,76),
    (107,60,194),
    (196,221,63),
    (111,115,206),
    (203,171,58),
    (61,38,94),
    (180,211,121),
    (214,69,116),
    (101,211,154),
    (209,69,48),
    (105,190,189),
    (215,128,63),
    (85,119,152),
    (192,166,116),
    (139,62,116),
    (82,117,57),
    (213,137,206),
    (54,60,54),
    (205,215,188),
    (106,39,44),
    (174,178,219),
    (131,89,48),
    (197,134,139)
]

from random import shuffle
shuffle(PALETTE_RGB)
PALETTE_RGB = [tuple([c / 255 for c in C]) for C in PALETTE_RGB]
PALETTE_HSV = [rgb_to_hsv(*c) for c in PALETTE_RGB]

def rgb(i, integral=False):
    if integral:
        return tuple(map(lambda v: int(255 * v), PALETTE_RGB[i % len(PALETTE_RGB)]))
    else:
        return PALETTE_RGB[i % len(PALETTE_RGB)]

def hsv(i, s=1, v=1):
    return PALETTE_HSV[i % len(PALETTE_HSV)]

def pts(pts):
    r"""
    Args:
        pts list of x and y or (x, y) tuples
    """
    import numpy as np
    if type(pts[0]) is int:
        pts = [[[pts[2*i], pts[2*i+1]]]for i in range(len(pts) // 2)]
    elif type(pts[0]) is tuple:
        pts = [[list(p)] for p in pts]
    return np.array(pts)

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
    if shape[::-1] != new_unpad:  # resize
        im = TF.resize(im, new_unpad[::-1], interpolation=TF.InterpolationMode.BILINEAR)
    if TF.is_tensor(im):
        #logging.info(f"resized tensor img shape={tuple(im.shape)}, dtype={im.dtype}, sum={im.sum(dim=(1,2))}")
        im = F.pad(im, mode='constant', pad=(left, right, top, bottom), value=sum(color)/len(color) if isinstance(color, tuple) else color)
    else:
        from ml.av.backend import opencv as cv
        #logging.info(f"resized cv img shape={tuple(im.shape)}, dtype={im.dtype}, sum={im.sum(axis=(0,1))}")
        im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # border padding
        
    return im, dict(
        shape=shape,        # HxW
        offset=(top, left), # offH, offW
        ratio=ratio,        # rH, rW
    )

def gen_patches(img, patch_size=(144, 320), resize=(720, 1280)):
    """
    Args:
        img: Tensor(C, H, W)
        patch_size: Tuple([H, W])
        resize: Tuple([H, W])
    Returns:
        img_patches: Tensor(P, C, H ,W)
        normalized cordinates: Tensor(P, x1, y1, x2, y2)
    """
    C = img.shape[0]

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(resize, int):
        resize = (resize, resize)

    pH, pW = patch_size
    rH, rW = resize

    assert rH % pH == 0 and rH >= pH, 'Cannot divide height into equal patches'
    assert rW % pW == 0 and rW >= pW, 'Cannot divide width into equal patches'

    trans = T.Compose([T.ToPILImage(), T.Resize(resize, constraint='longer'), T.ToTensor()])
    img_patches = trans(img).unfold(1, pH, pH).unfold(2, pW, pW) 

    rows = int(rW / pW)
    cols = int(rH / pH)

    xyxy = []
    x1, y1, x2, y2 = 0, 0, 0, 0
    j = 0
    for _ in range(rows * cols):
        h, w = pH, pW
        if j % rows == 0:
            y1 = y2
            y2 = y2 + h
            j = 0
        x1 = j * w
        x2 = x1 + w
        j += 1
        xyxy.append(th.tensor([x1/rW, y1/rH, x2/rW, y2/rH]))

    img_patches = img_patches.reshape(C, -1, pH, pW).permute(1, 0, 2, 3)
    xyxy = th.stack(xyxy)
    
    return img_patches, xyxy

@th.no_grad()
def draw_bounding_boxes(
    image: th.Tensor,
    boxes: th.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 3,
    font: Optional[str] = None,
    font_size: int = 16,
    alpha: Optional[int] = 100,
) -> th.Tensor:
    r"""Draws bounding boxes on given image in CHW, uint8.
    If fill is True, Resulting Tensor should be saved as PNG image.
    The current drawing backend is PIL.

    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, Union[4, 6]) 
            containing bounding boxes in (xmin, ymin, xmax, ymax) format or (xmin, ymin, xmax, ymax, score, class). 
            Note `0 <= xmin < xmax < W` and `0 <= ymin < ymax < H`.
        labels (List[str]): containing the labels of bounding boxes or classes if extended boxes are used.
        colors (Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]): 
            List containing the colors or a single color for all of the bounding boxes. 
            The colors can be represented as `str` or `Tuple[int, int, int]`.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. 
            If the file is not found in this filename, the loader may
            also search in other directories, such as 
            `fonts/` directory on Windows or 
            `/Library/Fonts/`, `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """
    from ml.av.backend.pil import Image, ImageFont, ImageDraw
    if font is None:
        FONT_MONO = 'UbuntuMono-B.ttf'
        try:
            font = ImageFont.truetype(font=FONT_MONO, size=font_size)
        except OSError as e:
            font = ImageFont.load_default()
            logging.warning(f"{e}: font {FONT_MONO} unavailable")
    
    if th.is_tensor(boxes):
        if boxes.shape[1] >= 6:
            # xyxysc
            classes = labels
            labels = []
            cs = []
            for s, c in boxes[:, 4:4+2].tolist():
                labels.append(f"{classes[int(c)]}: {s * 100:.0f}%")
                cs.append(COLORS91[int(c)])
            colors = cs if colors is None else colors
    elif boxes:
        # [(tid, xyxysc)*]
        tids, boxes = list(zip(*boxes))
        boxes = th.stack(boxes)
        assert boxes.shape[1] >= 6
        if labels:
            classes = labels
            labels = [f"{classes[c.int()]}[{tid}]" for tid, c in zip(tids, boxes[:, 5])]
        else:
            labels = [f"[{int(c)}][{tid}]" for tid, c in zip(tids, boxes[:, 5])]
        colors = [rgb(tid, integral=True) for tid in tids]
        cv.drawBoxes(img, result[:, :4], labels=labels, scores=result[:, 4], colors=colors)

    if not isinstance(image, th.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != th.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")

    if image.size(0) == 1:
        image = th.tile(image, (3, 1, 1))
    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(th.int64).tolist()
    draw = ImageDraw.Draw(img_to_draw, "RGBA")
    for i, bbox in enumerate(img_boxes):
        bbox = bbox[:4]
        if colors is None:
            color = None
        elif isinstance(colors, list):
            color = colors[i]
        else:
            color = colors
        if fill:
            if color is None:
                fill_color = (255, 255, 255, alpha)
            elif isinstance(color, str):
                # This will automatically raise Error if rgb cannot be parsed.
                fill_color = ImageColor.getrgb(color) + (alpha,)
            elif isinstance(color, tuple):
                fill_color = color + (alpha,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if labels is not None:
            # TODO: fine-tune the position and format
            w, h = draw.textsize(labels[i], font=font)
            margin = width + 1
            if bbox[1] > h + margin:
                x1, y1 = bbox[0], bbox[1] - (h + margin)
                x2, y2 = x1 + w + 2 * margin, bbox[1] - width // 2
                draw.rectangle((x1, y1, x2, y2), width=0, outline=None, fill=(255//2, 255//2, 255//2, 100))
                draw.text((x1 + margin, y1), labels[i], fill=color, font=font)
            else:
                x1, y1 = bbox[0] + width, bbox[1] + width
                x2, y2 = x1 + w + 2 * margin, y1 + h + width
                draw.rectangle((x1, y1, x2, y2), width=0, outline=None, fill=(255//2, 255//2, 255//2, 100))
                draw.text((x1 + margin, y1), labels[i], fill=color, font=font)

    return th.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=th.uint8)
