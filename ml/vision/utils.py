import random
from colorsys import rgb_to_hsv

import torch as th

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
COLORS = lambda n_classes: [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(n_classes)]

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

def gen_patches(img, patch_size=(144, 320), resize=(720, 1280)):
    """
    Args:
        img: Tensor(C, H, W)
        patch_size: Tuple([H, W])
        resize: Tuple([H, W])
    Returns:
        img_patches: Tensor(P, C, H ,W)
        normalized cordinates: Tensor[P, (x1, y1, x2, y2)]
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

    img_patches = TF.resize(img, resize, constraint='longer').unfold(1, pH, pH).unfold(2, pW, pW) 

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

