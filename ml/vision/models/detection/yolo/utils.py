import torch

from ml import logging
from ml.av import io
from ....transforms import functional as TF
from ....ops import *
from ....utils import letterbox

def parse(cfg):
    import re
    with open(cfg, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    lines = [x for x in lines if x and not x.startswith('#')]
    mdefs = []
    for line in lines:
        # print(f"line: {line}")
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].strip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0  # default to 0 for None
        else:
            key, val = line.split("=")
            key = key.strip()
            val = val.strip()
            if key == 'anchors':  # return nparray
                mdefs[-1][key] = torch.tensor([float(x) for x in re.split(r",\s*", val)]).view(-1, 2)
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                mdefs[-1][key] = [int(x) for x in re.split(r",\s*", val)]
            else:
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val

    # Check all fields are supported
    supported = {'type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'max_delta'}
    unsupported = set()

    f = set()
    for m in mdefs[1:]:
        [f.add(k) for k in m]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not u, f"Unsupported fields {u} in {cfg}. See https://github.com/ultralytics/yolov3/issues/631"
    return mdefs

def preprocess(image, size=640, **kwargs):
    """Sequential preprocessing of input images for YOLO
    Args:
        image(str | list[str] | ndarray | list[ndarray] | list[Tensors]): 
            image filename(s) or list[Tensor(RGB[CHW])] | CV BGR image(s)
    Returns:
        images(Tensor[BCHW]):
    """
    import numpy as np
    if isinstance(image, (str, np.ndarray)):
        images = [image]
    else:
        images = image

    if isinstance(images[0], str):
        images = [io.load(image) for image in images]

    # minimal only when all shapes are the same as in a batch
    shapes = [img.shape for img in images]
    minimal = all(map(lambda s: s == images[0].shape, shapes))

    resized = []
    metas = []
    # resize w/ optional padding to a mulitple of 32
    if TF.is_tensor(images[0]):
        for img in images:
            img, meta = letterbox(img, new_shape=size, auto=minimal)
            resized.append(img)
            metas.append(meta)
    else:
        for img in images:
            img, meta = letterbox(img, new_shape=size, auto=minimal)
            resized.append(torch.from_numpy(img).flip(-1).permute(2, 0, 1))
            metas.append(meta)
    
    resized = torch.stack(resized).to(dtype=torch.get_default_dtype()).div(255)
    return resized, metas

def batched_nms(predictions, 
                conf_thres=0.3, iou_thres=0.6, 
                agnostic=False, merge=True, 
                multi_label=False, classes=None):
    """Perform NMS on inference results
    Args:
        prediction(B, AG, 4+1+80): center, width and height refinement, plus anchor and class scores 
                                   per anchor and grid combination
        conf_thres(float): anchor confidence threshold
        iou_thres(float): NMS IoU threshold
        agnostic(bool): class agnostic NMS or per class NMS
        merge(bool): weighted merge by IoU for best mAP
        multi_label(bool): whether to select mulitple class labels above the threshold or just the max
        classes(list | tuple): class ids of interest to retain
    Returns:
        output(List[Tensor[B, N, 6]]): list of detections per image in (x1, y1, x2, y2, conf, cls)
    """
    min_wh, max_wh = 2, 4096                                        # minimum and maximum box width and height
    B, _, nc = predictions.shape
    nc -= 5
    multi_label &= nc > 1                                           # multiple labels per box if nc > 1 too
    output = [None] * B
    
    for b, x in enumerate(predictions):                             # image index and inference
        x = x[x[:, 4] > conf_thres]                                 # Threshold anchors by confidence
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)] # width/height constraints
        if x.numel() == 0:
            continue

        # Compute resulting class scores = anchor * class
        x[..., 5:] *= x[..., 4:5]

        # xcycwh to xyxy
        boxes = xcycwh2xyxy(x[:, :4].round())

        # Single or multi-label boxes
        if multi_label:
            # Combinations of repeated boxes with different classes: [x1, y1, x2, y2, conf, class]
            keep, cls = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((boxes[keep], x[keep, 5 + cls].unsqueeze(1), cls.float().unsqueeze(1)), 1)
        else:  
            # Best class only: [x1, y1, x2, y2, conf, class]
            conf, cls = x[:, 5:].max(1)
            x = torch.cat((boxes, conf.unsqueeze(1), cls.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter out boxes not in any specified classses
        if classes:
            x = x[(cls.view(-1, 1) == torch.tensor(classes, device=cls.device)).any(1)]

        if x.numel() == 0:
            continue

        # Batched NMS
        scores = x[:, 4]
        boxes = x[:, :4]
        cls = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        boxes = boxes + cls.view(-1, 1) * boxes.max()  # boxes (offset by class), scores
        keep = nms(boxes, scores, iou_thres)
        if merge and (1 < x.shape[0] < 3e3):
            # Weighted NMS box merge by IoU * scoress
            try:
                iou = box_iou(boxes[keep], boxes) > iou_thres   # Filtered IoU
                weights = iou * scores[None]                    # weighted IoU by class scores
                x[keep, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            except Exception as e:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                logging.error(f"Failed to merge NMS boxes by weighted IoU: {e}")
                print(x, x.shape, keep, keep.shape)
        output[b] = x[keep].to(predictions.dtype)

    return output

def postprocess(predictions, metas, 
                conf_thres=0.3, iou_thres=0.6, 
                agnostic=False, merge=True, 
                multi_label=False, classes=None):
    """Post-process to restore predictions on pre-processed images back.
    Args:
        predictions(Tensor[B,K,4+1+80]): batch output predictions from YOLO
    Returns:
        dets(xyxysc): 
    """
    dets = [None] * len(predictions)
    predictions = batched_nms(predictions, conf_thres, iou_thres, agnostic, merge, multi_label, classes)
    for b, (pred, meta) in enumerate(zip(predictions, metas)):
        if pred is None or len(pred) == 0:
            continue
        # Shift back
        top, left = meta['offset']
        pred[:, [0, 2]] -= left
        pred[:, [1, 3]] -= top
        # Scale back
        rH, rW = meta['ratio']
        pred[:, [0, 2]] /= rW
        pred[:, [1, 3]] /= rH
        # Clip boxes
        pred[:, :4] = clip_boxes_to_image(pred[:, :4].round(), meta['shape'])
        dets[b] = pred
    return dets