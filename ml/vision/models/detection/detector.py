import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pathlib import Path

import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection import maskrcnn_resnet50_fpn

import torch as th
import numpy as np

from .... import nn, random, logging
from ...ops import MultiScaleFusionRoIAlign, roi_align
from ...datasets import coco

COLORS91 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(coco.COCO91_CLASSES))]
COLORS80 = [COLORS91[coco.COCO80_TO_91[i]] for i in range(len(coco.COCO80_CLASSES))]

## Model factory methods to create detectors with consistent APIs

def rfcn(pooling=0, scales=(800, 1200), batch_size=1, dcn=True, softNMS=True, pretrained=True, model_dir=None, force_reload=False, gpu=0):
    return RFCNDetector(pooling=pooling, scales=scales, batch_size=batch_size, dcn=dcn, softNMS=softNMS, 
                        model_dir=model_dir, force_reload=force_reload, pretrained=pretrained, gpu=gpu)

def yolo4(pooling=False, fuse=True, **kwargs):
    from .yolo import yolo4
    cfg = kwargs.get('cfg', 'yolov4.cfg')
    weights = kwargs.get('weights', 'yolov4.weights')
    fuse = kwargs.get('fuse', True)
    m = yolo4(fuse=fuse)
    return YOLODetector(m, pooling=pooling)

def yolo5(chkpt, pretrained=False, channels=3, pooling=False, fuse=True, model_dir=None, force_reload=False, **kwargs):
    '''
    Kwargs:
        classes(List[str]): labels in class order
        person(List[int]): one or more person classes
        exclusion(List[int]): classes to ignore
    '''
    from .yolo5 import yolo5
    classes = kwargs.pop('classes', len(coco.COCO80_CLASSES))
    m = yolo5(chkpt, pretrained, channels=channels, classes=classes, fuse=fuse, model_dir=model_dir, force_reload=force_reload, **kwargs)
    return YOLODetector(m, pooling=pooling, classes=classes)

def yolo5l(pretrained=False, channels=3, pooling=False, fuse=True, model_dir=None, force_reload=False, **kwargs):
    from .yolo5 import yolo5l
    classes = kwargs.pop('classes', len(coco.COCO80_CLASSES))
    m = yolo5l(pretrained, channels=channels, classes=classes, fuse=fuse, model_dir=model_dir, force_reload=force_reload, **kwargs)
    return YOLODetector(m, pooling=pooling)

def yolo5x(pretrained=False, channels=3, pooling=False, fuse=True, model_dir=None, force_reload=False, **kwargs):
    from .yolo5 import yolo5x
    classes = kwargs.pop('classes', len(coco.COCO80_CLASSES))
    m = yolo5x(pretrained, channels=channels, classes=classes, fuse=fuse, model_dir=model_dir, force_reload=force_reload, **kwargs)
    return YOLODetector(m, pooling=pooling)

def detr(pretrained=False, pooling=False, deformable=False, backbone='resnet50', num_classes=91, model_dir=None, force_reload=False, **kwargs):
    from .detr.model import detr
    model = detr(pretrained, deformable=deformable, backbone=backbone, num_classes=num_classes, model_dir=model_dir, force_reload=force_reload, **kwargs)
    return DETRDetector(model, pooling=pooling, **kwargs)

def mask_rcnn(pretrained=False, num_classes=1+90, representation=1024, backbone=None, with_mask=True, **kwargs):
    if backbone is None:
        model = maskrcnn_resnet50_fpn(pretrained, pretrained_backbone=not pretrained, progress=True, **kwargs)
    else:
        model = maskrcnn_resnet50_fpn(pretrained, pretrained_backbone=False, progress=True, **kwargs)
        model.backbone = backbone

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    out_features = model.roi_heads.box_predictor.cls_score.out_features
    if representation != in_features:
        logging.info(f"Replaced box_head with representation size of {representation}")
        out_channels = model.backbone.out_channels
        resolution = model.roi_heads.box_roi_pool.output_size[0]
        model.roi_heads.box_head = TwoMLPHead(out_channels * resolution ** 2, representation)

    if representation != in_features or num_classes != out_features:
        logging.info(f"Replaced box_predictor with (representation, num_classes) = ({representation}, {num_classes})")
        model.roi_heads.box_predictor = FastRCNNPredictor(representation, num_classes)
        
    if not with_mask:
        model.roi_heads.mask_roi_pool = None
        model.roi_heads.mask_head = None
        model.roi_heads.mask_predictor = None
    
    return THDetector(model)

def mmdet_load(cfg, chkpt=None, with_mask=False, **kwargs):
    r"""Load an mmdet detection model from cfg and checkpoint.

    Args:
        cfg(str): config filename in configs/
        chkpt(str): optional path to checkpoint
        with_mask(bool): whether to perform semantic segmentation

    Kwargs:
        device(str): default to 'cuda:0' if not specified

    mmdet:
        htc:
            cfg: 'htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py'
            chkpt: 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth'
        cascade_rcnn:
            cfg: 'cascade_rcnn_r101_fpn_1x.py'
            chkpt: 'cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth'
    """

    from mmdet.apis import init_detector
    import mmdet
    cfg = Path(mmdet.__file__).parents[1] / 'configs' / cfg
    chkpt = chkpt and str(chkpt) or chkpt
    model = init_detector(str(cfg), chkpt, device=kwargs.get('device', 'cuda:0'))
    if not with_mask:
        model.mask_roi_extractor = None
        model.mask_head = None
        model.semantic_roi_extractor = None
        model.semantic_head = None

    return MMDetector(model)

## ML Detector APIs

class Detector(nn.Module):
    def __init__(self, model, classes=coco.COCO80_CLASSES, rewrites=None, **kwargs):
        '''
        Args:
            model(nn.Module): pre-defined model to make inferences
            classes(List[str]): list of class labels
            rewrites(dict): target to rewrite one or more source labels into an aggregated one
        '''
        super(Detector, self).__init__()
        self.module = model
        self.classes = classes

    def __getattr__(self, name):
        try:
            return super(Detector, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    @property
    def __class__(self):
        return self.module.__class__

    @property
    def with_rpn(self):
        return False

    @property
    def with_mask(self):
        return False
    
    @property
    def with_keypts(self):
        return False

    @abstractmethod
    def backbone(self, images, **kwargs):
        pass

    @abstractmethod
    def rpn(self, images, **kwargs):
        pass

    @abstractmethod
    def detect(self, images, **kwargs):
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

class RFCNDetector(Detector):
    """Pretrained Deformable RFCN for inference only with options to pull out object features for tracking.
    Limitations:
    - Only batch size of 1 is supported for inference
    - Pretrained requires dcn and softNMS to be True by default
    - No training support
    """
    def __init__(self, pooling=0, scales=(800, 1200), batch_size=1, dcn=True, softNMS=True, pretrained=True, model_dir=None, force_reload=False, gpu=0):
        from .rfcn import RFCN
        m = RFCN(scales=scales, batch_size=batch_size, dcn=dcn, softNMS=softNMS, pretrained=pretrained, model_dir=model_dir, force_reload=force_reload, gpu=gpu)
        super().__init__(m)
        self.pooling = pooling and int(pooling) or False
    
    @property
    def with_rpn(self):
        return True

    def detect(self, images, **kwargs):
        """Perform object detection.
        Args:
            images(str | list[str] | Tensor[B,C,H,W]): filename, list of filenames or an image tensor batch
        Returns:
            detection(list[Tensor[N, 6]]): list of object detection tensors in [x1, y1, x2, y2, score, class] per image
            pooled(list[Tensor[B, 256 | 512 | 1024, GH, GW]], optional): pooled features at three different scales
        """
        mode = self.training
        self.eval()
        model = self.module
        batch = self.preprocess(images)
        pred_boxes_all, scores_all, rois_output_all, rois_scores_all, features_all = model.detect(batch)
        dets = self.postprocess(pred_boxes_all, scores_all)
        
        # [(K, 4+1) * 80] -> [(K, 4+1+1)]
        boxes_all = []
        for i, dets_f in enumerate(dets):
            boxes = []
            for c, dets_c in enumerate(dets_f):
                dets_c = th.from_numpy(dets_c)
                classes = th.Tensor([c] * len(dets_c)).view(-1, 1)
                boxes_c = th.cat([dets_c, classes], dim=1)
                boxes.append(boxes_c)
            boxes = th.cat(boxes)
            indices = th.argsort(boxes[:, -1], dim=0, descending=True)
            boxes_all.append(boxes[indices])
        dets = boxes_all
        
        # [(K, 4+1)*]
        rois = []
        for i, (rois_boxes, rois_scores) in enumerate(zip(rois_output_all, rois_scores_all)):
            rois_boxes = th.from_numpy(rois_boxes)
            rois_scores = th.from_numpy(rois_scores).view(-1, 1)
            rois.append(th.cat([rois_boxes, rois_scores], dim=1))

        return_rpn = kwargs.get('return_rpn', False)
        if self.pooling:
            features = th.cat([th.from_numpy(features) for features in features_all])
            with th.no_grad():
                fh, fw = features.shape[-2:]
                h, w, ratio = batch[0]['im_info'][0]
                h /= ratio
                w /= ratio
                spatial_scale = (fw / w.item(), fh / h.item())
                dev = th.cuda.default_stream().device if th.cuda.is_available() else 'cpu'
                aligned = roi_align(features.to(dev), [d[:, :4].to(dev) for d in dets], output_size=(self.pooling, self.pooling), spatial_scale=spatial_scale)
                aligned = aligned.cpu()
            offset = 0
            alignedL = []
            for d in dets:
                alignedL.append(aligned[offset:offset+len(d)])
                offset += len(d)
            self.train(mode)
            return (rois, dets, alignedL) if return_rpn else (dets, alignedL)
        else:
            self.train(mode)
            return (rois, dets) if return_rpn else dets

class YOLODetector(Detector):
    def __init__(self, model, pooling=0, **kwargs):
        super(YOLODetector, self).__init__(model, **kwargs)
        self.engine = self.pooler = None
        if pooling:
            self.pooler = MultiScaleFusionRoIAlign(isinstance(pooling, bool) and 1 or pooling)
            logging.info(f"Multi-scale pooling size={self.pooler.output_size}")

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)[0]
        features = self.module.features
        return outputs, features

    def deploy(self, name='yolo5x', batch_size=10, spec=(3, 640, 640), fp16=True, backend='trt', reload=False, **kwargs):
        r"""Deploy optimized runtime backend.
        Args:
            batch_size(int): max batch size
            spec(Tuple[int]): preprocessed frame shape which must be fixed through the batch
            amp(bool): mixed precision with FP16
            kwargs:
                dynamix_axes: dynamic axes for each input ==> {'input_0': {0: 'batch_size', 2: 'height'}}
                min_shapes: min input shapes ==> [(3, 320, 640)]
                max_shapes: max input shapes ==> [(3, 640, 640)]
        """
        from ml import deploy
        module = self.module
        # avoids warning for dynamic ifs
        module.model[-1].onnx_dynamic = True
        int8 = kwargs.get('int8', False)
        strict = kwargs.get('strict', False)
        if int8:
            from ml import hub
            from ml.vision.datasets.coco import download

            def preprocessor(size=(384, 640)):
                from PIL import Image
                from torchvision import transforms
                trans = transforms.Compose([transforms.Resize(size),
                                            transforms.ToTensor()])

                H, W = size
                def preprocess(image_path, *shape):
                    r'''Preprocessing for TensorRT calibration
                    Args:
                        image_path(str): path to image
                        channels(int):
                    '''
                    image = Image.open(image_path)
                    logging.debug(f"image.size={image.size}, mode={image.mode}")
                    image = image.convert('RGB')
                    C = len(image.mode)
                    im = trans(image)
                    assert im.shape == (C, H, W)
                    return im

                return preprocess

            int8_calib_max = kwargs.get('int8_calib_max', 5000)
            int8_calib_batch_size = kwargs.get('int8_calib_batch_size', max(batch_size, 64)) 
            cache = f'{name}-COCO2017-val-{int8_calib_max}-{int8_calib_batch_size}.cache'
            cache_path = Path(os.path.join(hub.get_dir(), cache))
            kwargs['int8_calib_cache'] = str(cache_path)
            kwargs['int8_calib_data'] = download(split='val2017', reload=False)
            kwargs['int8_calib_preprocess_func'] = preprocessor()
            kwargs['int8_calib_max'] = int8_calib_max
            kwargs['int8_calib_batch_size'] = int8_calib_batch_size

        device = next(self.module.parameters()).device
        # FIXME: cuda + onnx_dynamic: causes the onnx export to fail: https://github.com/ultralytics/yolov5/issues/5439
        self.to('cpu') 
        self.engine = deploy.build(f"{name}-bs{batch_size}_{spec[-2]}x{spec[-1]}{fp16 and '_fp16' or ''}{int8 and '_int8' or ''}{strict and '_strict' or ''}",
                                   self,
                                   [spec],
                                   backend=backend, 
                                   reload=reload,
                                   batch_size=batch_size,
                                   fp16=fp16,
                                   strict_type_constraints=strict,
                                   **kwargs)
        self.to(device)
        # TODO: avoid storing dummy modules to keep track of module device
        self.dummy = module.model[-1]
        del self.module

    def detect(self, images, **kwargs):
        """Perform object detection.
        Args:
            images(str | List[str] | ndarray[HWC] | List[ndarray[HWC]]): filename, list of filenames or an image batch
        Returns:
            detection(List[Tensor[N, 6]]): list of object detection tensors in [x1, y1, x2, y2, score, class] per image
            pooled(list[Tensor[B, 256 | 512 | 1024, GH, GW]], optional): pooled features at three different scales
        """
        dev = next(self.parameters()).device

        from ml.vision.models.detection import yolo
        # mosaic = kwargs.get('mosaic', False)
        batch_preprocess = kwargs.get('batch_preprocess', False)
        size = kwargs.get('size', 640)
        cfg = dict(
            conf_thres = kwargs.get('cls_thres', 0.4),
            iou_thres = kwargs.get('nms_thres', 0.5),
            agnostic = kwargs.get('agnostic', False),
            merge = kwargs.get('merge', True),
        )
        batch, metas = batch_preprocess and yolo.batched_preprocess(images.to(dev), size=size) or yolo.preprocess(images, size=size)
        batch = batch.to(dev)
        with th.no_grad():
            if self.engine is None:
                predictions, features = self(batch)
            else:
                predictions, *features = self.engine.predict(batch)

        dets = yolo.postprocess(predictions, metas, **cfg)
        dtype = th.float32
        for dets_f in dets:
            if dets_f is not None:
                dtype = dets_f.dtype
                break

        dets = list(map(lambda det: th.empty(0, 6, dtype=dtype, device=dev) if det is None else det, dets))
        if self.pooler is None:
            return dets
        else:
            with th.no_grad():
                features = [feats.to(dets[0]) for feats in features]
                pooled = self.pooler(features, dets, metas)
            return dets, pooled

class DETRDetector(Detector):
    def __init__(self, model, pooling=False, **kwargs):
        super(DETRDetector, self).__init__(model, **kwargs)
        self.engine = None

        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])
        resize = kwargs.get('resize', (800, 800))
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(resize),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

        self.pooling = pooling

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        return outputs

    def deploy(self):
        r"""Deploy optimized runtime backend.
        """
        raise NotImplementedError('Deployment for DETR is not supported yet')
    
    def detect(self, images, **kwargs):
        """Perform object detection.       
        """
        param = next(self.parameters())

        from ml.vision.models.detection import detr
        conf_thres = kwargs.get('cls_thres', 0.5)
        
        batch, sizes = detr.preprocess(images, transform=self.transform)
        with th.no_grad():
            if self.engine is None:
                outputs, recordings = self(batch.to(param.device))
            else:
                raise NotImplementedError('DETR engine is not supported yet')

        decoder_out = recordings[0][-1][0][-1] # transformer decoder memory last layer
        dets, feats = detr.postprocess(outputs, decoder_out, sizes, conf=conf_thres)

        if self.pooling:
            return dets, feats
        else:
            return dets
    
class THDetector(Detector):
    def __init__(self, model):
        super(THDetector, self).__init__(model)

    @property
    def with_rpn(self):
        # self.model.roi_heads.[box_roi_pool | box_head | box_predictor]
        heads = self.module.roi_heads
        return hasattr(heads, 'box_head') and heads.box_head is not None

    @property
    def with_mask(self):
        return self.module.roi_heads.has_mask

    @property
    def with_keypts(self):
        return self.module.roi_heads.has_keypoint

    def backbone(self, images, **kwargs):
        r"""Returns backbone features and transformed input image list.

        Args:
            images(tensor | List[tensor | str]): a batch tensor of images, a list of image tensors, or image filenames
        
        Returns:
            images(ImageList): a transformed image list with scaled/padded image batch and shape meta
            features(tensor): backbone features in a batch
        """

        mode = self.training
        self.eval()
        model = self.module
        dev = next(model.parameters()).device

        if th.is_tensor(images):
            if images.dim() == 3:
                images = images.unsqueeze(0)
        elif not isinstance(images, list):
            images = [images]

        from ml import cv
        images = [
            image.to(dev) if th.is_tensor(image) else cv.toTorch(cv.imread(image), device=dev) 
            for image in images
        ]

        original_image_sizes = [img.shape[-2:] for img in images]
        with th.no_grad():
            images, _ = model.transform(images, targets=None)
            self.train(mode)
            return model.backbone(images.tensors), images, original_image_sizes
    
    def rpn(self, images, **kwargs):
        r"""Returns RPN proposals as well as backbone features and transformed input image list.

        Args:
            images(tensor): a batch tensor of images

        Kwargs:
            pooling(bool): whether to compute pooled and transformed RoI features and representations
            targets(dict): target descriptor of keys
                boxes(float): list of RoI box tensor of shape(N, 4)
                labels(int64):
                keypoints(float):
        """
        features, images, original_image_sizes = self.backbone(images, **kwargs)

        # Layered outputs or a last layer single batch tensor
        if isinstance(features, th.Tensor):
            features = OrderedDict([(0, features)])
        
        targets = kwargs.get('targets')
        if targets is not None:
            for t in targets:
                assert t['boxes'].dtype.is_floating_point, 'target boxes must be of float type'
                assert t['labels'].dtype == th.int64, 'target labels must be of int64 type'
                if self.has_keypoint:
                    assert t["keypoints"].dtype == th.float32, 'target keypoints must be of float type'

        pooling = kwargs.pop('pooling', False)
        mode = self.training
        self.eval()
        model = self.module
        with th.no_grad():
            proposals, _ = model.rpn(images, features, **kwargs)
            if self.training:
                proposals, matched_idxs, labels, regression_targets = model.select_training_samples(proposals, targets)
                
            if pooling:
                roi_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
                box_features = model.roi_heads.box_head(roi_features)
                self.train(mode)
                return (proposals, box_features, roi_features), (features, images, original_image_sizes)
            else:
                self.train(mode)
                return proposals, (features, images, original_image_sizes)

    def detect(self, images, **kwargs):
        r"""Returns detections as well as RPN proposals and backbone features.

        Args:
            images(tensor): a batch tensor of images

        Kwargs:
            score_thr(float): threshold to filter out low scored objects
            pooling(bool): whether to compute pooled and transformed RoI features and representations
            targets(dict): target descriptor of keys
                boxes(float):
                labels(int64):
                keypoints(float):
        Returns:
            results(list[tensor]): a list of sorted detection tensors per image tensor([[x1, y1, x2, y2, score, cls]*]+)

        Note:
            - clipped to image
            - bg removed
            - empty or too small filtering
            - scoring threshold: 0.05
            - per class NMS threshold: 0.5
        """
        mode = self.training
        self.eval()
        model = self.module
        (proposals, box_features, roi_featurs), (features, images, original_image_sizes) = self.rpn(images, pooling=True)
        with th.no_grad():
            class_logits, box_regression = model.roi_heads.box_predictor(box_features)
            boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, images.image_sizes)
            score_thr = kwargs.pop('score_thr', 0.3)
            num_images = len(boxes)
            results = []
            for i in range(num_images):
                selection = scores[i] > score_thr
                res = dict(
                    boxes=boxes[i][selection],
                    scores=scores[i][selection],
                    labels=labels[i][selection],
                )
                #print(f"images[{i}]: scores[{len(scores[i])}] {scores[i]}")
                results.append(res)

            if kwargs.get('pooling'):
                det_roi_features = model.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
                det_box_features = model.roi_heads.box_head(det_roi_features)
                results = model.transform.postprocess(results, images.image_sizes, original_image_sizes)
                results = [th.cat([res['boxes'], res['scores'].view(-1,1), res['labels'].view(-1,1).float()], dim=1) for res in results]
                self.train(mode)
                return ((results, det_box_features, det_roi_features), 
                        (proposals, box_features, roi_featurs), 
                        (features, images))
            else:
                results = model.transform.postprocess(results, images.image_sizes, original_image_sizes)
                results = [th.cat([res['boxes'], res['scores'].view(-1,1), res['labels'].view(-1,1).float()], dim=1) for res in results]
                self.train(mode)
                return results

    def forward(self, images, targets=None):
        return self.module(images, targets)

    def show_result(self,
                    img,
                    result,
                    classes=coco.COCO91_CLASSES,
                    score_thr=0.3,
                    wait_time=0,
                    out_file=None):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            wait_time (int): Value of waitKey param.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.
        """
        import mmcv
        img = mmcv.imread(img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        # draw bounding boxes
        bboxes = bbox_result[:, :-1].cpu().numpy()
        labels = bbox_result[:, -1].cpu().int().numpy()
        #print(bbox_result.shape, bboxes.shape, labels.shape)
        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            class_names=self.CLASSES if classes is None else classes,
            score_thr=score_thr,
            show=out_file is None,
            wait_time=wait_time,
            out_file=out_file)

class MMDetector(Detector):
    def __init__(self, model):
        super(MMDetector, self).__init__(model)

    @property
    def __class__(self):
        return self.module.__class__

    @property
    def with_rpn(self):
        return hasattr(self.module, 'rpn_head') and self.module.rpn_head is not None

    @property
    def with_mask(self):
        return self.module.with_mask
    
    @property
    def with_keypts(self):
        raise NotImplementedError

    def backbone(self, images, **kwargs):
        r"""Returns list of backbone features and transformed images as well as meta info.
        """
        from mmdet.apis.inference import inference_detector, LoadImage
        from mmdet.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        model = self.module
        cfg = model.cfg
        device = next(model.parameters()).device
        test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)    
        results = []
        for img in images:
            data = dict(img=img)
            data = test_pipeline(data)
            data = scatter(collate([data], samples_per_gpu=1), [device])[0]
            img = data['img'][0]
            img_meta = data['img_meta'][0]
            data['img'] = img
            data['img_meta'] = img_meta
            data['feats'] = model.extract_feat(img)
            results.append(data)
            #print(img.shape, img_meta)
        
        #return model.backbone(images.tensors), images, original_image_sizes
        return results
        
    def rpn(self, images, **kwargs):
        r"""Returns a list of RPN proposals and RoI pooled features if necessary.
        """
        results = self.backbone(images, **kwargs)
        model = self.module
        pooling = kwargs.pop('pooling', False)
        for res in results:
            x = res['feats']
            img_meta = res['img_meta']
            proposals = model.simple_test_rpn(x, img_meta, model.test_cfg.rpn)[0]
            res['proposals'] = proposals
            #print(f"proposals: {len(proposals)}, {proposals[0]}")
            if pooling:
                from mmdet.core import bbox2roi
                rois = bbox2roi([proposals])
                stages = self.num_stages if hasattr(model, 'num_stages') else 2
                if stages <= 2:
                    # TODO
                    roi_feats = model.bbox_roi_extractor(x[:len(model.bbox_roi_extractor.featmap_strides)], rois)
                    if model.with_shared_head:
                        roi_feats = model.shared_head(roi_feats)

                    box_head = model.box_head
                    if box_head.with_avg_pool:
                        box_feats = box_head.avg_pool(roi_feats)                    
                    box_feats = box_feats.view(box_feats.shape[0], -1)
                    res['roi_feats'] = roi_feats
                    res['box_feats'] = box_feats
                    #print(roi_feats.shape, box_feats.shape)
                else:
                    ms_scores = []
                    ms_bbox_result = {}
                    for i in range(stages):
                        bbox_roi_extractor = model.bbox_roi_extractor[i]
                        bbox_head = model.bbox_head[i]
                        roi_feats = bbox_roi_extractor(x[:len(bbox_roi_extractor.featmap_strides)], rois)
                        if model.with_shared_head:
                            roi_feats = model.shared_head(roi_feats)
                        
                        cls_score, bbox_pred = bbox_head(roi_feats)
                        ms_scores.append(cls_score)
                        #print(f"[stage{i}] {rois.shape}, {roi_feats.shape}, {cls_score.shape}, {bbox_pred.shape}")
                        if i < stages - 1:
                            bbox_label = cls_score.argmax(dim=1)
                            rois = bbox_head.regress_by_class(rois, bbox_label, bbox_pred, img_meta[0])

                    cls_score = sum(ms_scores) / stages
                    res['proposals'] = th.cat([rois[:, 1:], cls_score.max(dim=1, keepdim=True)[0]], dim=1)
                    res['bbox_pred'] = bbox_pred
                    res['roi_feats'] = roi_feats
                    print(bbox_pred.shape, bbox_pred[0])
                    #res['box_feats'] = box_feats
                    #proposals = res['proposals']
                    #print(f"refined proposals: {len(proposals)}, {proposals[0]}")
        return results

    def detect(self, images, **kwargs):
        r"""Detect objects in one or more images.
        Args:
            images(str/ndarray | List[str/ndarray]): filename or a list of filenames

        Kwargs:
            score_thr(float): threshold to filter out low scored objects

        Returns:
            results: a list of sorted per image detection [[x1, y1, x2, y2, score, cls]*]
        """
        from mmdet.apis import inference_detector
        if not isinstance(images, list):
            images = [images]

        results = []
        score_thr = kwargs.pop('score_thr', 0.3)
        for i, img in enumerate(images):
            res = inference_detector(self.module, img)
            dets = []
            # det: (x1, y1, x2, y2, score, class)
            for c, det in enumerate(res): # no bg, 0:80
                det = th.from_numpy(det[det[:, -1] > score_thr])
                #print(f"{len(det)} detections of class {c}")
                if len(det) > 0:
                    cls = th.Tensor([c] * len(det)).view(-1, 1)
                    det = th.cat([det, cls], dim=1)
                    #print(det)
                    dets.append(det)

            if dets:
                dets = th.cat(dets)
                sorted = th.argsort(dets[:, 4+1], descending=True)
                dets = dets[sorted]
                results.append(dets)
            else:
                results.append(th.zeros(0, 5+1))
            #print(f"[{i}] {len(dets)} detections")

        return results
    
    def forward(self, images, targets=None):
        raise NotImplementedError
    
    def show_result(self,
                    img,
                    result,
                    classes=coco.COCO80_CLASSES,
                    score_thr=0.3,
                    wait_time=0,
                    out_file=None):
        """Visualize the detection results on the image.

        Args:
            img (str or np.ndarray): Image filename or loaded image.
            result (tuple[list] or list): The detection result, can be either
                (bbox, segm) or just bbox.
            class_names (list[str] or tuple[str]): A list of class names.
            score_thr (float): The threshold to visualize the bboxes and masks.
            wait_time (int): Value of waitKey param.
            out_file (str, optional): If specified, the visualization result will
                be written to the out file instead of shown in a window.
        """
        import mmcv
        img = mmcv.imread(img)
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        
        if False:
            # TODO: Show mask result
            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img[mask] = img[mask] * 0.5 + color_mask * 0.5

        # draw bounding boxes
        bboxes = bbox_result[:, :-1].numpy()
        labels = bbox_result[:, -1].int().numpy()
        #print(bbox_result.shape, bboxes.shape, labels.shape)
        mmcv.imshow_det_bboxes(
            img.copy(),
            bboxes,
            labels,
            class_names=self.CLASSES if classes is None else classes,
            score_thr=score_thr,
            show=out_file is None,
            wait_time=wait_time,
            out_file=out_file)