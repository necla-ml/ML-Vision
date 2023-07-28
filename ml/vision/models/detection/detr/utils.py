import torch
import numpy as np

def preprocess(images, transform=None):
    """
    Apply transform to batch of image

    Parameters:
        images: Tensor[N, C, H, W] or List[Tensor[C, H, W]]
        transform: Torch transform 
    Returns:
        batch: Transformed batch of image tensors
        sizes: Original image sizes before transform
    """
    if isinstance(images, torch.Tensor):
        batch = transform(images)
        sizes = [batch[0].shape[1:]] * len(batch)
    else:
        batch = torch.stack([transform(im) for im in images])
        sizes = [im.shape[1:] for im in images]

    return batch.float(), sizes

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(b, size):
    img_h, img_w = size
    b = box_cxcywh_to_xyxy(b)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b

def postprocess(outputs, decoder_out, sizes, conf=0.7):
    """
    Postprocess DETR output logits

    Parameters:
        outputs: Output from DETR
        sizes: Original image sizes 
        conf: Inference confidence threshold
    Returns:
        batched_output: List([N, 6])
    """

    out_preds = outputs['pred_logits'].softmax(-1)
    out_bbox = outputs['pred_boxes']
    probas = out_preds[:, :, :-1]
    keep = probas.max(-1).values > conf

    device = probas.device
    dtype = probas.dtype
  
    dets = [None] * len(out_preds)
    feats = [None] * len(out_preds)
    for i, out_pred in enumerate(probas):
        # empty dets and feats 
        if not torch.any(keep[i]):
            dets[i] = torch.empty(0, 6, dtype=dtype, device=device)
            feats[i] = torch.empty(0, decoder_out[i].size(-1), dtype=dtype, device=device)
            continue

        # convert boxes from [0; 1] to image scales
        boxes = rescale_bboxes(out_bbox[i][keep[i]], sizes[i])
        scores, classes = out_pred[keep[i]].max(dim=-1)

        # feats and dets
        feats[i] = decoder_out[i][keep[i]]
        dets[i] = torch.cat((boxes, scores.unsqueeze(1), classes.unsqueeze(1)), dim=1)

    return dets, feats

