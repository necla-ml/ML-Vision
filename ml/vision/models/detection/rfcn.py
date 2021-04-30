# Copyright (c) 2017-present, NEC Laboratories America, Inc. ("NECLA"). 
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import time
from pathlib import Path
import numpy as np
from ml import cv, hub

"""
def soft_nms_wrapper(thresh, max_dets=-1):
    def _soft_nms(dets):
        return soft_nms(dets, thresh, max_dets)
    return _soft_nms
"""

def attempt_download(params, epoch=8, model_dir=None, force=False):
    if model_dir is None:
        hub_dir = hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
    
    try:
        os.makedirs(model_dir)
    except OSError as e:
        import errno
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise
    chkpts = {
        'rfcn_dcn_coco': '0B6T5quL13CdHZ3ZrRVNjcnFmZk0',
        'rfcn_coco': None,
    }
    id = chkpts[params]
    path = f"{model_dir}/{params}-{epoch:04d}.params"
    if hub.download_gdrive(id, path, force=force) != 0:
        raise IOError(f"Failed to download to {path}")
    return path

class RFCN(object):
    def __init__(self, scales=(800, 1200), batch_size=1, dcn=True, softNMS=True, pretrained=True, model_dir=None, force_reload=False, gpu=0):
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
        os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
        import mxnet as mx
        from dcn.nms import gpu_nms_wrapper, cpu_softnms_wrapper, soft_nms
        from dcn.utils.load_model import load_param
        from rfcn.config.config import config, update_config
        from rfcn.core.tester import Predictor
        from rfcn.symbols import resnet_v1_101_rfcn
        from rfcn.symbols import resnet_v1_101_rfcn_dcn
        from rfcn.symbols import resnet_v1_101_rfcn_dcn_rpn
        from rfcn import CFGS

        cfg    = CFGS / f"rfcn_coco_demo{softNMS and '_softNMS' or ''}.yaml"
        update_config(cfg)
        config.TEST.BATCH_IMAGES = batch_size
        config.symbol = 'resnet_v1_101_rfcn_dcn_rpn' if dcn else 'resnet_v1_101_rfcn'
        config.SCALES[0] = scales

        params, arg_params, aux_params = None, None, None
        if pretrained:
            epoch = softNMS and 8 or 0
            params = dcn and 'rfcn_dcn_coco' or 'rfcn_coco'
            path = attempt_download(params, epoch=epoch, model_dir=model_dir, force=force_reload)
            arg_params, aux_params = load_param(f"{Path(path).parent / params}", epoch, process=True)
        instance = eval(config.symbol + '.' + config.symbol)()
        sym = instance.get_symbol(config, is_train=False)

        # Build the model predictor
        # BATCH_IMAGES Per GPU context
        self.config = config
        gpus = [gpu] if type(gpu) == int else gpu
        data_names  = ['data', 'im_info']
        label_names = []
        data        = self.preprocess()
        data        = [[mx.nd.array(data[i][name]) for name in data_names] for i in range(len(data))]                   # [[data, im_info], ...]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 800, 1200))]]
        provide_data  = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in range(len(data))]                 # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        
        self.predictor = Predictor(sym, data_names, label_names,
                              context=[mx.gpu(gpu) for gpu in gpus], max_data_shapes=max_data_shape,
                              provide_data=provide_data, provide_label=provide_label,
                              arg_params=arg_params, aux_params=aux_params)
        self.data_names = data_names
        self.nms = gpu_nms_wrapper(config.TEST.NMS, 0)  # 0.2 for SNMS and 0.3 otherwise
        if softNMS:
            self.snms = cpu_softnms_wrapper()

    def preprocess(self, frames=None, interpolation=cv.INTER_LINEAR):
        if frames is None:
            # Fake frames to warmup
            frames = [np.ones((480, 640, 3)) * 114 for _ in range(self.config.TEST.BATCH_IMAGES)]
        else:
            if isinstance(frames, (str, np.ndarray)):
                frames = [frames]
            if isinstance(frames[0], str):
                frames = [cv.imread(frame) for frame in frames]

        # resize to a predefined scale (800, 1200) for SoftNMS with aspect ratio preserved
        # transform from BGR HxWxC to RGB CxHxW with normalization
        data = []
        config = self.config
        from dcn.utils.image import resize, transform
        for im in frames:
            # config.SCALES = [(800, 1200)]
            # data.shape = (B, C, H, W)
            # im_info = [[800, 1067, 1.666]]
            target_size = config.SCALES[0][0]
            max_size = config.SCALES[0][1]
            im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE, interpolation=interpolation)
            im_tensor = transform(im, config.network.PIXEL_MEANS)
            im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)
            data.append({'data': im_tensor, 'im_info': im_info})
        return data

    def rpn(self, frames):
        r"""Return RoIs from RPN only.        
        """
        import mxnet as mx
        data = [[mx.nd.array(frames[i][name]) for name in self.data_names] for i in range(len(frames))]                 # [[data, im_info], ...]
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 800, 1200))]]
        provide_data = [[(k, v.shape) for k, v in zip(self.data_names, data[i])] for i in range(len(data))]             # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=None, 
                                     provide_data=provide_data, 
                                     provide_label=provide_label)
        output_all      = self.predictor.predict(data_batch)
        data_dict_all   = [dict(zip(self.data_names, idata)) for idata in data_batch.data]
        scales          = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]
        rois_output_all = []
        rois_scores_all = []
        for output, data_dict, scale in zip(output_all, data_dict_all, scales):
            if config.TEST.HAS_RPN: # from updated yaml
                # ROIs from RPN
                rois = output['rois_output'].asnumpy()[:, 1:]
            else:
                # ROIs from RFCN? 
                rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
            rois_output_all.append(rois / scale)
            rois_scores_all.append(output['rois_score'].asnumpy())
        return rois_output_all, rois_scores_all

    def _detect(self, frames):
        from dcn.bbox.bbox_transform import bbox_pred, clip_boxes
        import mxnet as mx
        config = self.config
        max_data_shape = [[('data', (1, 3, max([v[0] for v in config.SCALES]), max([v[1] for v in config.SCALES])))]]   # [[('data', (1, 3, 800, 1200))]]
        data = [[mx.nd.array(frames[i][name]) for name in self.data_names] for i in range(len(frames))]                 # [[data, im_info], ...]
        provide_data = [[(k, v.shape) for k, v in zip(self.data_names, data[i])] for i in range(len(data))]             # [[('data', (1, 3, h, w)), ('im_info', (h, w, scale))], ...]]
        provide_label = [None for i in range(len(data))]                                                                # [None, ...]
        data_batch = mx.io.DataBatch(data=data, label=[], pad=0, index=None, 
                                     provide_data=provide_data, 
                                     provide_label=provide_label)

        output_all      = self.predictor.predict(data_batch)
        data_dict_all   = [dict(zip(self.data_names, idata)) for idata in data_batch.data]
        scales          = [data_batch.data[i][1].asnumpy()[0, 2] for i in range(len(data_batch.data))]
        scores_all      = []
        pred_boxes_all  = []
        rois_output_all = []
        rois_scores_all = []
        features_all    = []
        for output, data_dict, scale in zip(output_all, data_dict_all, scales):
            # conv_feat(res4b22_relu), relu1(res5c_relu)
            # rois_output, rois_score
            # bbox_pred_reshape_output, cls_prob_reshape_output
            im_shape = data_dict['data'].shape
            if config.TEST.HAS_RPN: # from updated cfg
                # ROIs from RPN
                rois = output['rois_output'].asnumpy()[:, 1:]
            else:
                # ROIs from RFCN? 
                rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]

            # post processing of bboxes from rois
            bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
            pred_boxes = bbox_pred(rois, bbox_deltas)
            pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])
            pred_boxes = pred_boxes / scale            
            pred_boxes_all.append(pred_boxes)
            
            scores = output['cls_prob_reshape_output'].asnumpy()[0]
            scores_all.append(scores)

            # batch of one frame
            rois_output_all.append(rois / scale)
            rois_scores_all.append(output['rois_score'].asnumpy())
            features_all.append(output['res4b22_relu_output'].asnumpy())
            # features_all.append(output['res5c_relu_output'].asnumpy())
        return pred_boxes_all, scores_all, rois_output_all, rois_scores_all, features_all

    def detect(self, frames):
        outputs = [[item[0] for item in self._detect([frame])] for frame in frames]
        pred_boxes_all, scores_all, rois_output_all, rois_scores_all, features_all = list(zip(*outputs))
        return pred_boxes_all, scores_all, rois_output_all, rois_scores_all, features_all

    def postprocess(self, pred_boxes_all, scores_all):
        r"""Class-agnostic NMS for the final predictions with FG removed.
        """
        # TODO potentially slow => parallelization
        # RPN_POST_NMS_TOP_N: 300
        dets = []
        for scores, boxes in zip(scores_all, pred_boxes_all):
            boxes = boxes.astype('f')
            scores = scores.astype('f')
            dets_nms = []
            for c in range(1, scores.shape[1]): # Starting with person FG objects
                cls_scores  = scores[:, c, np.newaxis]
                cls_boxes   = boxes[:, 4:8] if self.config.CLASS_AGNOSTIC else boxes[:, c*4: (c+1)*4]
                cls_dets    = np.hstack((cls_boxes, cls_scores))
                keep        = self.nms(cls_dets)
                cls_dets    = cls_dets[keep, :]
                cls_dets    = cls_dets[cls_dets[:, -1] > 0.7, :]
                dets_nms.append(cls_dets)
        
            # per class bboxes (#bboxes, 4+1)
            dets.append(dets_nms)
        return dets