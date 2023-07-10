import torch
from torch.nn import functional as F

from torchvision.ops import roi_align

from ... import nn
from .utils import rois2boxes

class MultiScaleFusionRoIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale=1.0, sampling_ratio=-1):
        """Multi-scale fusion RoIAlign pooling fuses multi-scale feature maps for consistent RoI algin.
        Args:
            featmap_names (List[str]): names of feature maps used for the pooling.
            output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = tuple(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, x, boxes, metas):
        """
        Args:
            x (List[Tensor]): list of batch multi-scale feature maps from small scale to largest.
            boxes(List[Tensor[N, 4]] or Tensor[K, 5]): list of boxes to pool w.r.t. original image sizes.
            metas(List[dict]): the preprocessing params w.r.t. original shape, resize ratio, padding offsets
        Returns:
            aligned(List[Tensor[K, C, OH, OW]]): list of pooled RoI features w.r.t. boxes
        """
        size = x[0].shape[2:]
        resampled = [x[0]]
        for i, feats in enumerate(x[1:], 1):
            interpolated = F.interpolate(feats, scale_factor=2 ** i, mode='bilinear', align_corners=False)
            resampled.append(interpolated)
        batch = torch.cat(resampled, 1)


        # XXX pooling w.r.t. the resized/padded image sizes
        if torch.is_tensor(boxes):
            rois = rois2boxes(rois.clone(), len(metas))
        else:
            rois = [dets_f[:,:4].clone() for dets_f in boxes]
        
        rois_rp = []
        scale = None
        for dets_rp, meta in zip(rois, metas):
            rH, rW = meta['ratio']
            top, left = meta['offset']
            dets_rp[:, [0, 2]] = dets_rp[:, [0, 2]] * rW + left
            dets_rp[:, [1, 3]] = dets_rp[:, [1, 3]] * rH + top
            rois_rp.append(dets_rp)
            if scale is None:
                shape = list(meta['shape'])
                shape[0] = int(shape[0] * rH + 2 * top)
                shape[1] = int(shape[1] * rW + 2 * left)
                scale = (size[1]/shape[1], size[0]/shape[0])

        # FIXME roi_align() is very slow when output_size is small
        if self.output_size[0] == 1:
            aligned = roi_align(batch, rois_rp, (7, 7), spatial_scale=scale, aligned=True)
            if len(aligned) > 0:
                aligned = F.max_pool2d(aligned, 7)
            else:
                aligned = aligned[:, :, 0, 0, None, None]
        else:
            aligned = roi_align(batch, rois_rp, self.output_size, spatial_scale=scale, aligned=True)
        offset = 0
        alignedL = []
        for dets in rois:
            alignedL.append(aligned[offset:offset+len(dets)])
            offset += len(dets)

        return alignedL