from functools import partial

import torch
from torch import nn

from torchvision.models.resnet import _resnet
from torchvision.models import get_model_weights
from torchvision.models.resnet import Bottleneck
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.models.detection.backbone_utils import BackboneWithFPN


class Backbone(nn.Module):
    r"""Select a subset of feature maps from a model to make a backbone with or without FPN.

    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.

    This adaptation allows not to use FPN and the last feature maps would be reshaped in two dimensions.

    Args:
        backbone (nn.Module): the full-fleged model to adapt
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
        fpn (bool): whether to add FPN or not.

    Attributes:
        out_channels (int): the number of channels in the FPN
    """

    def __init__(self, model, return_layers, in_channels_list=[], out_channels=None, fpn=False, classifier=False):
        super(Backbone, self).__init__()
        self.backbone = BackboneWithFPN(model, return_layers, in_channels_list, out_channels)
        self.classifier = None
        if classifier and hasattr(model, 'fc'):
            # XXX hardcoded for resnet family models
            self.classifier = model.fc
        if not fpn:
            del self.backbone._modules['fpn']

    def __getattr__(self, name):
        try:
            return super(Backbone, self).__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)

    def freeze(self, exceptions=None):
        exceptions = [] if exceptions is None else exceptions
        for name, parameter in self.backbone.body.named_parameters():
            if all(map(lambda k: k not in name, exceptions)):
                parameter.requires_grad = False
        if self.classifier is not None:
            for parameter in self.classifier.parameters():
                parameter.requires_grad = False

    def forward(self, x):
        # torchvision>=0.5
        if 'fpn' in self.backbone._modules:
            x = self.backbone.body(x)
            return self.backbone.fpn(x)
        else:
            # XXX explict shape necessarily for trt
            outputs = self.backbone.body(x)
            last = len(outputs) - 1
            avgpool = outputs[last]
            outputs[last] = avgpool.flatten(1)
            # logging.info(f"output shapes={[(k, tuple(v.shape)) for k, v in outputs.items()]}")
            return tuple(outputs.values())

    def load_state_dict(self, state_dict, strict=True):
        return self.backbone.body.load_state_dict(state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Save some checkpoint
        return self.backbone.body.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)


def resnet50(fpn=False, frozen=True, exceptions=['layer2', 'layer3', 'layer4'], classifier=False, progress=True, **kwargs):
    r"""ResNet50 model from
    [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    The keyward argument `norm_layer` defaults to BatchNorm2d.
    Alternative options include `FixedBatchNorm2d` and `FrozenBatchNorm2d`.
    The former acts always as in the evaluation mode without taking gradients.
    The later acts as transparent without changing input and tracking statistics.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        fpn (bool): Whether to include FPN or not
        exceptions (List[str]): keywords of layers not to freeze
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = f"resnet50"
    kwargs['norm_layer'] = kwargs.get('norm_layer', partial(FrozenBatchNorm2d, eps=1e-5) if frozen else None)
    if fpn:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    else:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'avgpool': 4}
    in_channels_stage2 = 256
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = kwargs.get('out_channels', 256)
    weights = get_model_weights(arch).DEFAULT
    model = _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
    backbone = Backbone(model, return_layers, in_channels_list, out_channels, fpn=fpn, classifier=classifier)
    backbone.freeze(exceptions)
    return backbone


def resnet101(fpn=False, frozen=True, exceptions=['layer2', 'layer3', 'layer4'], classifier=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    The keyward argument `norm_layer` defaults to BatchNorm2d.
    Alternative options include `FixedBatchNorm2d` and `FrozenBatchNorm2d`.
    The former acts always as in the evaluation mode without taking gradients.
    The later acts as transparent without changing input and tracking statistics.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        fpn (bool): Whether to include FPN or not
        exceptions (List[str]): keywords of layers not to freeze
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    arch = f"resnet101"
    kwargs['norm_layer'] = kwargs.get('norm_layer', partial(FrozenBatchNorm2d, eps=1e-5) if frozen else None)
    if fpn:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    else:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'avgpool': 4}
    in_channels_stage2 = 256
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = kwargs.get('out_channels', 256)
    weights = get_model_weights(arch).DEFAULT
    model = _resnet(Bottleneck, [3, 4, 23, 3], weights, progress, **kwargs)
    backbone = Backbone(model, return_layers, in_channels_list, out_channels, fpn=fpn, classifier=classifier)
    backbone.freeze(exceptions)
    return backbone


def resnext101_wsl(fpn=False, frozen=True, exceptions=['layer2', 'layer3', 'layer4'], classifier=False, progress=True, **kwargs):
    kwargs['groups'] = gs = kwargs.get('groups', 32)
    kwargs['width_per_group'] = gw = kwargs.get('width_per_group', 8)
    kwargs['norm_layer'] = kwargs.get('norm_layer', partial(FrozenBatchNorm2d, eps=1e-5) if frozen else None)

    if fpn:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}
    else:
        return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'avgpool': 4}
    in_channels_stage2 = 256
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = kwargs.get('out_channels', 256)

    arch = f"resnext101_{gs}x{gw}d_wsl"
    WSL = torch.hub.load('facebookresearch/WSL-Images', arch, **kwargs)
    backbone = Backbone(WSL, return_layers, in_channels_list, out_channels, fpn=fpn, classifier=classifier)

    backbone.freeze(exceptions)
    return backbone