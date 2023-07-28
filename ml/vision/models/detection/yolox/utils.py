import torch
import torch.nn.functional as F

import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def yolox_resize(img, input_size=(640, 640), pad_value=114, interpolation=InterpolationMode.BILINEAR):
    """
    Args:
        img: Tensor(C, H, W) | Tensor(B, C, H, W)
        input_size: Tuple(H, W)
    Returns:
        resized_padded_img, ratio
    """
    img_h, img_w = img.shape[-2:]

    r = min(input_size[0] / img_h, input_size[1] / img_w)
    size = (int(img_h * r), int(img_w * r))

    resized_img = TF.resize(img, size=size, interpolation=interpolation, antialias=True)
    r_shape = resized_img.shape[-2:]
    pad_h = input_size[0] - r_shape[0]
    pad_w = input_size[1] - r_shape[1]
    padded_img = F.pad(resized_img, pad=(0, pad_w, 0, pad_h), value=pad_value)

    return padded_img.float(), r


def preprocess(img, input_size=(640, 640), pad_value=114):
    '''
    Args:
        img: List[Tensor(C, H, W)] | Tensor(B, C, H, W)
        input_size: Tuple(H, W)
    Returns:
        resized_padded_img, ratio
    '''
    if isinstance(img, torch.Tensor):
        return yolox_resize(img, input_size, pad_value)
    else:
        padded_img, r  = zip(*[yolox_resize(im) for im in img])
        padded_img = torch.stack(padded_img)
        return padded_img, r
    

def postprocess(prediction, ratio=1, num_classes=80, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    """
    Returns:
        scaled output: Tensor(N, xyxy + s + c)
            - The xyxy bounding boxes are scaled according to the input ratio
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    if isinstance(ratio, float):
        ratio = [ratio] * len(prediction)
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            output[i] = torch.empty(0, 4+1+1, device=detections.device, dtype=detections.dtype)
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        # xyxy + s + c
        # scale: xyxy / ratio
        detections = torch.cat([detections[:, :4] / ratio[i], detections[:, -3:-2] * detections[:, -2:-1], detections[:, -1:]], dim=1)
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output