import os
from abc import abstractmethod
from pathlib import Path

import torch as th
import torchvision.transforms as T

from ml import nn, random, logging
from ...datasets import coco

COLORS91 = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(coco.COCO91_CLASSES))]
COLORS80 = [COLORS91[coco.COCO80_TO_91[i]] for i in range(len(coco.COCO80_CLASSES))]

## Model factory methods to create detectors with consistent APIs
"""
DETR
"""
def detr(pretrained=False, pooling=False, deformable=False, backbone='resnet50', num_classes=91, model_dir=None, force_reload=False, **kwargs):
    from .detr.model import detr
    model = detr(pretrained, deformable=deformable, backbone=backbone, num_classes=num_classes, model_dir=model_dir, force_reload=force_reload, **kwargs)
    return DETRDetector(model, pooling=pooling, **kwargs)

"""
YOLOX
"""
def yolox_x(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_x', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)

def yolox_l(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_l', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)

def yolox_m(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_m', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)

def yolox_s(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_s', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)

def yolox_nano(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_nano', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)

def yolox_tiny(pretrained=True, pooling=False, num_classes=80, device='cpu', force_reload=False, unload_after=True, **kwargs):
    from .yolox.model import yolox
    model, exp = yolox(arch='yolox_tiny', pretrained=pretrained, num_classes=num_classes, device=device, force_reload=force_reload, unload_after=unload_after)
    return YOLOXDetector(model, pooling=pooling, size=exp.test_size, pad_value=114, num_classes=num_classes, **kwargs)
    

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


class DETRDetector(Detector):
    def __init__(self, model, pooling=False, **kwargs):
        super(DETRDetector, self).__init__(model, **kwargs)
        self.engine = None

        mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        std = kwargs.get('std', [0.229, 0.224, 0.225])
        resize = kwargs.get('resize', (800, 800))
        self.transform = T.Compose([
            T.Resize(resize, antialias=True),
            T.Lambda(lambda x: x.float().div(255.0)),
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
        with th.inference_mode():
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
        

class YOLOXDetector(Detector):
    def __init__(self, model, pooling=False, **kwargs):
        super().__init__(model, **kwargs)
        self.engine = None

        self.input_size = kwargs.get('size', (640, 640))
        self.pad_vaue = kwargs.get('pad_value', 114)
        self.num_classes = kwargs.get('num_classes', 80)

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        return outputs

    def deploy(self, name='yolox', batch_size=10, spec=(3, 640, 640), fp16=True, backend='trt', reload=False, **kwargs):
        r"""Deploy optimized runtime backend.
        """
        from ml import deploy
        module = self.module

        int8 = kwargs.get('int8', False)
        strict = kwargs.get('strict', False)
        if int8:
            from ml import hub
            from ml.vision.datasets.coco import download

            def preprocessor(size=(640, 640)):
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
            kwargs['int8_calib_preprocess_func'] = preprocessor(spec[1:])
            kwargs['int8_calib_max'] = int8_calib_max
            kwargs['int8_calib_batch_size'] = int8_calib_batch_size

        device = next(self.module.parameters()).device
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
        self.module = module.head.obj_preds[-1]
        # del self.module
    
    def detect(self, images, **kwargs):
        """Perform object detection.       
        """
        device = next(self.parameters()).device

        from ml.vision.models.detection import yolox
        nms_thresh = kwargs.get('nms_thresh', 0.65)
        cls_thresh = kwargs.get('cls_thresh', 0.45)
        batch_preprocess = kwargs.get('batch_preprocess', False)

        images = images.to(device) if batch_preprocess else images
        batch, ratio = yolox.preprocess(images, input_size=self.input_size, pad_value=self.pad_vaue)
        with th.inference_mode():
            if self.engine is None:
                predictions, feats = self(batch.to(device))
            else:
                predictions, *feats = self.engine.predict(batch.to(device))
                

            dets = yolox.postprocess(predictions, ratio, num_classes=self.num_classes, nms_thre=nms_thresh, conf_thre=cls_thresh)

        return dets