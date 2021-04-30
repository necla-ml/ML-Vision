from torchvision.ops.boxes import *
import torch as th

def pad_boxes(boxes, padding=None, inplace=False):
    '''
    Args:
        boxes: in xyxy
        inplace(bool): whether to overwrite the input boxes
        padding(float, Tuple[float, float]): ratios in width and height to pad
    '''
    xyxy = boxes if inplace else boxes.clone()
    if padding is None:
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
        '''
        print('original ppls_f[:, :4]: ', ppls_f[:, :4].tolist())
        print('crop ppls_xyxy: ', ppls_xyxy.tolist())
        print('clip ppls_xyxy: ', ppls_xyxy_clip.tolist())
        print('widths:', widths.tolist())
        print('heights:', heights.tolist())
        print('widers:', wider.tolist())
        print('higher:', higher.tolist())
        print('offsetW:', offsetW.tolist())
        print('offsetH:', offsetH.tolist())
        '''
    else:
        # XXX enlarge to specified padding
        # entering/exiting may prefer large windows but not otherwise
        padding = (padding, padding) if isinstance(padding, int) else padding
        widths = boxes[:, 2] - boxes[:, 0] + 1
        heights = boxes[:, 3] - boxes[:, 1] + 1
        paddingW = widths * padding[0]
        paddingH = heights * padding[1]
        xyxy[:, 0] -= paddingW
        xyxy[:, 1] -= paddingH
        xyxy[:, 2] += paddingW
        xyxy[:, 3] += paddingH
    return xyxy
    
def clip_boxes_to_image(boxes, size):
    # type: (Tensor, Tuple[int, int]) -> Tensor
    """
    Clip boxes so that they lie inside an image of size `size`.
    Arguments:
        boxes (Tensor[N, 4]): boxes in (x1, y1, x2, y2) format
        size (Tuple[height, width]): size of the image
    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    if torchvision._is_tracing():
        boxes_x = th.max(boxes_x, th.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_x = th.min(boxes_x, th.tensor(width - 1, dtype=boxes.dtype, device=boxes.device))
        boxes_y = th.max(boxes_y, th.tensor(0, dtype=boxes.dtype, device=boxes.device))
        boxes_y = th.min(boxes_y, th.tensor(height - 1, dtype=boxes.dtype, device=boxes.device))
    else:
        boxes_x = boxes_x.clamp(min=0, max=width - 1)
        boxes_y = boxes_y.clamp(min=0, max=height - 1)

    clipped_boxes = th.stack((boxes_x, boxes_y), dim=dim)
    return clipped_boxes.reshape(boxes.shape)

def box_intersect(boxes1: Tensor, boxes2: Tensor) -> Tensor:
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