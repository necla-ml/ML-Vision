from torchvision.ops._utils import _cat, convert_boxes_to_roi_format as boxes2rois, check_roi_boxes_shape
import torch
import numpy as np


## Detection manipulation

def dets_select(dets, inclusion, inverse=False):
    """
    Args:
        dets(xyxysc, List[xyxysc]): detection tensor(s)
        inclusion(List[int]): list of classes to include
        inverse(bool): invert the selection or not
    """
    res = []
    listed = isinstance(dets, list)
    if not listed:
        dets = [dets]
    for i, dets_i in enumerate(dets):
        selection = dets_i[:, -1] == -1
        for p in inclusion:
            selection |= dets_i[:, -1] == p
        res.append(~selection if inverse else selection)
    return res if listed else res[0]

## RoI conversions

def rois2boxes(rois, bs=None):
    r"""RoI tensor with 0th column specifying batch index to batch list of bounding box tensors.
    """
    bs = bs or int(rois[-1][0].item())
    return [rois[rois[:, 0] == b] for b in range(bs)]

'''
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def boxes2rois(boxes):
    r"""List of bounding box tensors to a ROI tensor with 0th column specifying batch index.

    NOTE:
        Even if no RoI is detected, an empty tensor must be given to decide the device and dimensions to be transparent for RoI pooling.
    """
    dev = boxes[0].device if boxes else None
    concat_boxes = _cat([b for b in boxes], dim=0)
    ids = _cat(
        [
            torch.full_like(b[:, :1], i) # if len(b) > 0 else torch.tensor([], device=dev)
            for i, b in enumerate(boxes)
        ],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois
'''

## Detection format conversions

def xyxys2xyxysc(dets):
    """Convert detection format List[(x1,y1,x2,y2,score)*] to (x1,y1,x2,y2,score,cls)
    Args:
        dets(List[Tensor[N, 5]]: list of detections by in class order from one frame
    Returns:
        xyxysc(Tensor[N, 6]): detections of one frame
    """
    xyxysc = []
    count = len(dets)
    if count <= 0:
        return torch.Tensor(0, 6)
    device = None
    for c, detsC in enumerate(dets):
        if detsC is not None:
            device = detsC.device
            C = torch.Tensor([c], device=detsC.device).expand(len(detsC), 1)
            detsC = torch.cat((detsC, C), dim=1)
            xyxysc.append(detsC)
    xyxysc = torch.cat(xyxysc)
    return xyxysc

def xyxysc2xyxys(dets, classes):
    """Convert detection format (x1,y1,x2,y2,score,cls) to List[(x1,y1,x2,y2,score)*]
    Args:
        dets(Tensor[N, 6]: detections from one frame
    Returns:
        xyxys(List[Tensor[N, 5]]: list of detections by in class order
    """
    xyxys = []
    count = isinstance(classes, int) and classes or len(classes)
    for c in range(count):
        selection = dets[:, -1] == c
        selection = dets[selection][:, :-1]
        xyxys.append(selection) # potentially empty
    return xyxys

## Plain box format conversions
# xyxy: detections
# xywh: CV2 rect
# xcycwh: deep sort

def xcycwh2xyxy(x, inplace=False):
    """Convert Nx4 boxes from center, width and height to top-left and bottom-right coordinates.
    Args:
        x(Tensor[N, 4]): N boxes in the format [xc, yc, w, h]
        inplace(bool): whether to modify the input
    Returns:
        boxes(Tensor[N, 4]): N boxes in the format of [x1, y1, x2, y2]
    """
    # Boolean mask causes a new tensor unless assignment inplace
    y = torch.zeros_like(x)
    even = (x[:, 2] % 2) == 0
    odd =  (x[:, 2] % 2) == 1
    y[:, 0][even] = x[:, 0][even] - x[:, 2][even] // 2 + 1
    y[:, 0][odd] = x[:, 0][odd] - x[:, 2][odd] // 2
    even = (x[:, 3] % 2) == 0
    odd =  (x[:, 3] % 2) == 1
    y[:, 1][even] = x[:, 1][even] - x[:, 3][even] // 2 + 1
    y[:, 1][odd] = x[:, 1][odd] - x[:, 3][odd] // 2
    y[:, 2] = x[:, 0] + x[:, 2] // 2
    y[:, 3] = x[:, 1] + x[:, 3] // 2
    if inplace:
        x.copy_(y)
        return x
    return y

def xcycwh2xywh(xcycwh, inplace=False):
    """Convert bbox from (xc,yc,w,h) to (x1,y1,w,h)
    Args:
        xcycwh(Tensor): bboxes in the format of [xc, yc, w, h]
        inplace(bool): whether to modify the input
    Returns:
        xywh(Tensor): bboxes in [x, y, w, h]
    """
    xywh = xcycwh if inplace else xcycwh.clone()
    even = (xcycwh[:, 2] % 2) == 0
    odd =  (xcycwh[:, 2] % 2) == 1
    xywh[:, 0][even] = xcycwh[:, 0][even] - xcycwh[:, 2][even] // 2 + 1
    xywh[:, 0][odd] = xcycwh[:, 0][odd] - xcycwh[:, 2][odd] // 2
    even = (xcycwh[:, 3] % 2) == 0
    odd = (xcycwh[:, 3] % 2) == 1
    xywh[:, 1][even] = xcycwh[:, 1][even] - xcycwh[:, 3][even] // 2 + 1
    xywh[:, 1][odd] = xcycwh[:, 1][odd] - xcycwh[:, 3][odd] // 2
    return xywh

def xyxy2xcycwh(xyxy, inplace=False):
    """Convert boxes in (x1,y1,x2,y2) to (xc,yc,w,h)
    Args:
        xyxy(Tensor[N, 4]): N boxes in (x1,y1,x2,y2)
        inplace(bool): whether to modify the input inplace as output or make the results a new copy
    Returns:
        xcycwh(Tensor[N, 4]): N boxes in (xc,yc,w,h)
    """
    xcycwh = torch.zeros_like(xyxy)
    xcycwh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) // 2  # xc
    xcycwh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) // 2  # yc
    xcycwh[:, 2] = xyxy[:, 2] - xyxy[:, 0] + 1     # width
    xcycwh[:, 3] = xyxy[:, 3] - xyxy[:, 1] + 1     # height
    if inplace:
        xyxy.copy_(xcycwh)
        return xyxy
    return xcycwh

def xyxy2xywh(xyxy, inplace=False):
    """Convert bbox from (x1,y1,x2,y2) to (x1,y1,w,h).
    """
    if inplace:
        xywh = xyxy
    else:
        xywh = xyxy.clone()
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0] + 1
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1] + 1
    return xywh

def xywh2xyxy(xywh, inplace=False):
    """Convert bbox from (x1,y1,w,h) to (x1,y1,x2,y2).
    """
    if inplace:
        xyxy = xywh
    else:
        xyxy = xywh.clone()
    if xywh.dim() == 1:
        xywh = xywh.unsqueeze(0)
        xyxy.unsqueeze_(0)
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] - 1
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] - 1
    return xyxy.squeeze_()