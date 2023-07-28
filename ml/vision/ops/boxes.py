import torch as th

def enlarge_boxes(boxes, enlarge_ratio=None, inplace=False):
    '''
    Args:
        boxes: in xyxy
        inplace(bool): whether to overwrite the input boxes
        enlarge_ratio(float, Tuple[float, float]): ratios in width and height to pad
    '''
    xyxy = boxes if inplace else boxes.clone()
    if enlarge_ratio is None:
        # XXX enlarge to square in the longest side to resize for backbone feature extraction
        widths = xyxy[:, 2] - xyxy[:, 0] + 1
        heights = xyxy[:, 3] - xyxy[:, 1] + 1
        wider = widths >= heights
        offsetH = (widths[wider] - heights[wider]) / 2
        xyxy[wider, 1] -= offsetH
        xyxy[wider, 3] += offsetH
        higher = ~wider
        offsetW = (heights[higher] - widths[higher]) / 2
        xyxy[higher, 0] -= offsetW
        xyxy[higher, 2] += offsetW
    else:
        # XXX enlarge to specified enlarge_ratio
        enlarge_ratio = (enlarge_ratio, enlarge_ratio) if isinstance(enlarge_ratio, float) else enlarge_ratio
        widths = boxes[:, 2] - boxes[:, 0] + 1
        heights = boxes[:, 3] - boxes[:, 1] + 1
        paddingW = widths * enlarge_ratio[0]
        paddingH = heights * enlarge_ratio[1]
        xyxy[:, 0] -= paddingW
        xyxy[:, 1] -= paddingH
        xyxy[:, 2] += paddingW
        xyxy[:, 3] += paddingH
    return xyxy
    
    
def box_intersect(boxes1, boxes2):
    """
    Return intersection of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        intersection (Tensor[N, M]): the NxM matrix containing the pairwise intersection areas
    """
    lt = th.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = th.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    return wh[:, :, 0] * wh[:, :, 1]  # [N,M]