import os 
import warnings

import torch
from ml import nn

from .models import (
    backbone,
    detection,
)

from . import version
from . import transforms
from . import utils
from . import io

# load and register _C extensions
from .extension import _HAS_OPS

# Check if mlvision is being imported within the root folder
if not _HAS_OPS and os.path.dirname(os.path.realpath(__file__)) == os.path.join(
    os.path.realpath(os.getcwd()), "vision"
):
    message = (
        "You are importing ml-vision within its own root folder ({}). "
        "This is not expected to work and may give errors. Please exit the "
        "torchvision project source and relaunch your python interpreter."
    )
    warnings.warn(message.format(os.getcwd()))

AE   = 'AE'
AE3  = '3AE'
AE4  = '4AE'
UAE  = 'UAE'
UAE4 = '4UAE'
DAE  = 'DAE'
DAE3 = '3DAE'

__all__ = [
    'Backbone',
]

def build(arch, pretrained=False, *args, **kwargs):
    num_classes = kwargs.pop('num_classes', None)
    
    model = None
    if arch == 'resnet50':
        from torchvision import models
        model = models.__dict__[arch](pretrained)
        if num_classes is not None and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnet101':
        from torchvision import models
        model = models.__dict__[arch](pretrained)
        layers = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        ]

        for i in range(3): # 56x56, 28x28, 14x14, 7x7
            name = 'layer%d' % (i + 1)
            layers.append(getattr(model, name))
        
        model.features = torch.nn.Sequential(*layers)
        model = models.__dict__[arch](pretrained)
        if num_classes is not None and model.fc.out_features != num_classes:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == 'resnext101': # 32x4d | 64x4d
        import pretrainedmodels
        cardinality = kwargs.get('cardinality', None)
        width = kwargs.get('width', None)
        arch = f"{arch}_{cardinality}x{width}d"
        model = pretrainedmodels.__dict__[arch](num_classes=num_classes or 1000, pretrained=pretrained and 'imagenet' or None)

    """
    elif arch == 'desenet121':
        if pretrained:
            print(f"=> using pre-trained model '{arch}'")
        else:
            print(f"=> creating model '{arch}'")
        model = models.__dict__[arch](pretrained)

        # XXX replace the last FC layer if necessary
        # model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.arch == 'alexnet':
        if args.pretrained == 1:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        # replace the last FC layer
        model._modules['classifier']._modules['6'] = nn.Linear(model.classifier[6].in_features, args.num_classes)
    elif args.arch == 'squeezenet1_0':
        if args.pretrained == 1:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(args.arch))
            model = models.__dict__[args.arch]()

        # replace the last FC layer
        model._modules['classifier']._modules['1'] = nn.Conv2d(512, args.num_classes, (1,1), stride=1)
    """
    return model
