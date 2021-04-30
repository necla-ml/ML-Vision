from pathlib import Path
import torch

from ..... import nn, io, logging
from .....nn import yolo, functional as F
from .....utils import Config
from . import utils

def create(cfg="yolov4.cfg"):
    cfg = cfg.endswith('.cfg') and cfg or f"{cfg}.cfg"
    url = f"https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/{cfg}"
    path = Path(f"/tmp/{cfg}")
    assert io.download(url, path), f"Failed to download {url}"
    mdefs = utils.parse(path)
    net = mdefs.pop(0)
    assert net['type'] == 'net'
    logging.info(f"Parsed {len(mdefs)} module definitions")

    out_channels = [net['channels']]
    width, hight = net['width'], net['height']
    modules = nn.ModuleList()
    routed = set()
    yolo_index = -1
    for i, mdef in enumerate(mdefs):
        mtype = mdef['type']
        if mtype == 'convolutional':
            filters = mdef['filters']
            module = yolo.Conv.create(mdef, out_channels[-1])
            if mdef['batch_normalize'] == 0: # output to head
                routed.add(i)
        elif mtype == 'maxpool':
            module = yolo.MaxPool2d.create(mdef)
        elif mtype == 'upsample':
            module = yolo.Upsample.create(mdef)
        elif mtype == 'route':
            route = yolo.Route.create(mdef)
            filters = sum([out_channels[l + 1 if l > 0 else l] for l in route.layers])
            routed = routed.union(i + l if l < 0 else l for l in route.layers)
            module = route
        elif mtype == 'shortcut':
            shortcut = yolo.Shortcut.create(mdef)
            filters = out_channels[-1]
            routed = routed.union(i + l if l < 0 else l for l in shortcut.layers)
            module = shortcut
        elif mtype == 'yolo':
            yolo_index += 1
            module = nn.YOLOHead.create(mdef, yolo_index)
            modules[-2].pooled = True
            '''
            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -1
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[:modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                bias[:, 4] += -4.5  # obj
                bias[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                logging.warning('WARNING: smart bias initialization failure.')
            '''
        else:
            logging.warning(f'Unrecognized module type: {mtype}')
        modules.append(module)
        out_channels.append(filters)
    routes = [False] * (i + 1)
    for i in routed:
        routes[i] = True
    return net, modules, routes

def load(cfg="yolov4.cfg", target='yolov4.weights', force=False):
    net, modules, routed = create(cfg)
    import numpy as np
    url = f"https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/{target}"
    path = Path(f"/tmp/{target}")
    assert io.download(url, path, force=force), f"Failed to download {url}"
    with open(path, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        version = np.fromfile(f, dtype=np.int32, count=3)  # major, minor, revision
        seen = np.fromfile(f, dtype=np.int64, count=1)     # number of images seen during training
        weights = torch.from_numpy(np.fromfile(f, dtype=np.float32))
        logging.info(f"{target}-v{'.'.join(map(str, version))} of {weights.numel()} floats trained for {seen.item()} images")
    ptr = 0
    for i, module in enumerate(modules):
        if isinstance(module, nn.Conv):
            conv = module[0]
            if module.with_bn:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()
                bn.bias.data.copy_(weights[ptr:ptr+nb].view_as(bn.bias))
                ptr += nb
                bn.weight.data.copy_(weights[ptr:ptr+nb].view_as(bn.weight))
                ptr += nb
                bn.running_mean.data.copy_(weights[ptr:ptr+nb].view_as(bn.running_mean))
                ptr += nb
                bn.running_var.data.copy_(weights[ptr:ptr + nb].view_as(bn.running_var))
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv.bias.data.copy_(weights[ptr:ptr+nb].view_as(conv.bias))
                ptr += nb
            # Load conv. weights
            logging.debug(f"conv {list(conv.weight.shape)} {conv.weight.numel()} {ptr}/{weights.numel()}")
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(weights[ptr:ptr+nw].view_as(conv.weight))
            ptr += nw
        else:
            # Shortcut, Route, Head
            # MiWRC?
            pass
    assert ptr == weights.numel()
    logging.info(f"Loaded darknet weights from {url}")
    return dict(
        version=version,
        seen=seen,
        cfg=net,
        modules=modules,
        routed=routed,
    )

def yolo4(**kwargs):
#def yolo4(cfg='yolov4.cfg', weights='yolov4.weights', fuse=True):
    """
    Args:
        cfg(str): configuration name to download if unavailable
        weights(str): pretrained weights name to download if unavailable
        device(str or torch.device): target device to host the model
        fuse(bool): whether to fuse convolutional and BN
        pooling(int): output RoI feature pooing size if greater than 0
    """
    model = YOLO.create(cfg='yolov4.cfg', weights='yolov4.weights')
    kwargs.get('fuse', True) and model.fuse()
    return model

class YOLO(nn.Module):
    @classmethod
    def create(cls, cfg="yolov4.cfg", weights='yolov4.weights', debug=False):
        chkpt = load(cfg, weights)
        ver = chkpt['version']
        seen = chkpt['seen']
        return cls(chkpt['cfg'], chkpt['modules'], chkpt['routed'], version=ver, seen=seen, debug=debug)
    
    def __init__(self, cfg, modules, routed, version=None, seen=None, debug=False):
        super(YOLO, self).__init__()
        self.cfg = Config(cfg)
        self.routed = routed
        self.stages = modules
        self.features = []
        self.logits = []
        self.debug = debug
        if self.debug:
            logging.info(f"{self.__class__.__name__}:")
            print(f"{self.cfg}")
            for i, m in enumerate(self.stages):
                print(f"[{i}] {routed[i] and '<routed> ' or ''}{m}")

    def fuse(self):
        # Fuse stages with Conv2d and BatchNorm2d
        from ml.nn import utils
        fused = 0
        for stage in self.stages:
            if isinstance(stage, nn.Conv) and stage.with_bn:
                conv, bn = stage[0], stage[1]
                leftover = stage[2]
                stage[0] = utils.fuse_conv_bn(conv, bn)
                for i in range(2, len(stage)):
                    stage[i-1] = stage[i]
                del stage[-1]
                stage.with_bn = False
                fused += 1
        logging.info(f"Fused {fused} convolutional stages with 2D BN")

    def forward(self, x, augment=False, verbose=False):
        """
        Returns:
            predictions(Tensor[B,K,4+1+80]): predictions in xyxysc from different anchors
        """
        return self.forward_once(x)
        '''
        if not augment:
            return self.forward_once(x)
        else:  
            # Augment images (inference and test only) 
            # https://github.com/ultralytics/yolov3/issues/931
            img_size = x.shape[-2:]  # height, width
            s = [0.83, 0.67]  # scales
            y = []
            for i, xi in enumerate((x,
                                    torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                    torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                    )):
                # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                y.append(self.forward_once(xi)[0])

            y[1][..., :4] /= s[0]  # scale
            y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
            y[2][..., :4] /= s[1]  # scale
            # for i, yi in enumerate(y):  # coco small, medium, large = < 32**2 < 96**2 <
            #     area = yi[..., 2:4].prod(2)[:, :, None]
            #     if i == 1:
            #         yi *= (area < 96. ** 2).float()
            #     elif i == 2:
            #         yi *= (area > 32. ** 2).float()
            #     y[i] = yi
            y = torch.cat(y, 1)
            return y, None
        '''

    def forward_once(self, x, augment=False, verbose=False):
        bs = x.shape[0]
        height, width = x.shape[-2:]
        predictions = []    # head predictions
        outputs = []        # per layer output
        self.features.clear()
        self.logits.clear()
        if verbose:
            logging.info(f'x.shape: {list(x.shape)}')
            msg = ''
        '''
        # Augment images (inference and test only)
        if augment:  # https://github.com/ultralytics/yolov3/issues/931
            nb = x.shape[0]     # batch size
            s = [0.83, 0.67]    # scales
            x = torch.cat((x,
                           torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                           torch_utils.scale_img(x, s[1]),  # scale
                           ), 0)
        '''
        for i, module in enumerate(self.stages):
            name = module.__class__.__name__
            if name in ['Shortcut', 'Route']:
                if verbose:
                    srcs = [i - 1] + module.layers  # layers
                    shapes = [list(x.shape)] + [list(out[j].shape) for j in module.layers]
                    msg = ' >> ' + ' + '.join([f'layer[{src}] {list(shape)}' for src, shape in zip(srcs, shapes)])
                x = module(x, outputs)
                if self.debug:
                    logging.info(f"[{i}] {name} output shape={tuple(x.shape)}, sum={x.sum():.3f}")
            elif name == 'YOLOHead':
                self.features.append(outputs[-2])
                self.logits.append(x)
                predictions.append(module(x))
            else:  
                # 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
                if self.debug:
                    logging.info(f"[{i}] {name} output shape={tuple(x.shape)}, sum={x.sum():.3f}")

            outputs.append(x if self.routed[i] or hasattr(module, "pooled") else None)
            if verbose:
                logging.info(f'{nane}[{i}/{len(self.stages)}] {list(x.shape)} {msg}')
                msg = ''

        if self.training:
            return predictions, self.logits
        else:
            y = predictions
            y = torch.cat(y, 1)  # cat yolo outputs
            if augment:  # de-augment results
                y = torch.split(y, bs, dim=0)
                y[1][..., :4] /= s[0]               # scale
                y[1][..., 0] = width - y[1][..., 0] # flip lr
                y[2][..., :4] /= s[1]               # scale
                y = torch.cat(y, 1)                 # (B, AG, 85)
            return y
        
'''
# [YOLOv4-pytorch](https://github.com/romulus0914/YOLOv4-PyTorch)
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class SpatialPyramidPooling(nn.Module):
    """### SPP ###
    """
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools]
        features = torch.cat([x]+features, dim=1)
        return features

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()
        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)

class PANet(nn.Module):
    def __init__(self, feature_channels):
        super(PANet, self).__init__()
        self.feature_transform3 = Conv(feature_channels[0], feature_channels[0]//2, 1)
        self.feature_transform4 = Conv(feature_channels[1], feature_channels[1]//2, 1)
        self.resample5_4 = Upsample(feature_channels[2]//2, feature_channels[1]//2)
        self.resample4_3 = Upsample(feature_channels[1]//2, feature_channels[0]//2)
        self.resample3_4 = Downsample(feature_channels[0]//2, feature_channels[1]//2)
        self.resample4_5 = Downsample(feature_channels[1]//2, feature_channels[2]//2)
        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2]*2, feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
            Conv(feature_channels[0]//2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0]//2, 1),
        )

        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
            Conv(feature_channels[1]//2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1]//2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1),
            Conv(feature_channels[2]//2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2]//2, 1)
        )

    def forward(self, features):
        features = [self.feature_transform3(features[0]), self.feature_transform4(features[1]), features[2]]
        downstream_feature5 = self.downstream_conv5(features[2])
        downstream_feature4 = self.downstream_conv4(torch.cat([features[1], self.resample5_4(downstream_feature5)], dim=1))
        downstream_feature3 = self.downstream_conv3(torch.cat([features[0], self.resample4_3(downstream_feature4)], dim=1))
        upstream_feature4 = self.upstream_conv4(torch.cat([self.resample3_4(downstream_feature3), downstream_feature4], dim=1))
        upstream_feature5 = self.upstream_conv5(torch.cat([self.resample4_5(upstream_feature4), downstream_feature5], dim=1))
        return [downstream_feature3, upstream_feature4, upstream_feature5]

class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels=255):
        super(PredictNet, self).__init__()
        self.predict_conv = nn.ModuleList([
            nn.Sequential(
                Conv(feature_channels[i]//2, feature_channels[i], 3),
                nn.Conv2d(feature_channels[i], target_channels, 1)
            ) for i in range(len(feature_channels))
        ])

    def forward(self, features):
        predicts = [predict_conv(feature) for predict_conv, feature in zip(self.predict_conv, features)]
        return predicts

class YOLOv4(nn.Module):
    def __init__(self):
        super(YOLOv4, self).__init__()

        # CSPDarknet53 backbone
        self.backbone, feature_channels = csp_darknet53()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
            Conv(feature_channels[-1]//2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1]//2, 1),
        )

        # Spatial Pyramid Pooling
        self.spp = SpatialPyramidPooling()

        # Path Aggregation Net
        self.panet = PANet(feature_channels)

        # predict
        self.predict_net = PredictNet(feature_channels)

    def forward(self, x):
        features = self.backbone(x)
        features[-1] = self.head_conv(features[-1])
        features[-1] = self.spp(features[-1])
        features = self.panet(features)
        predictions = self.predict_net(features)
        return predictions

if __name__ == '__main__':
    model = YOLOv4()
    x = torch.randn(1, 3, 256, 256)
    predictions = model(x)
'''