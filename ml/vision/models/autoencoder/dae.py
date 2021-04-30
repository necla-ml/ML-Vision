import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict

# DAE: dense autoencoder

__all__ = ['create', 'DAE', ]
#__all__ = ['dae', 'DAE', 'daefc', 'DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

class _DenseLayer(nn.Sequential):
    r"""
    - 2D size remains with potentially dilated conv2d
    - input feature depth -> bottleneck (bn_size * k) -> k -> cat(x, k)
    - 
    """
    def __init__(self, num_input_features, bn_size, growth_rate, dilation, dp):
        super(_DenseLayer, self).__init__()
        bottleneck = bn_size * growth_rate
        kernel_size = 3
        padding = (kernel_size - 1) // 2 * dilation
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features, eps=EPS)),
        self.add_module('relu.1', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bottleneck, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bottleneck, eps=EPS)),
        self.add_module('relu.2', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bottleneck, growth_rate, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False))
        self.dp = dp

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.dp > 0:
            # new_features = F.dropout(new_features, p=self.dp, training=self.training)
            new_features = F.dropout2d(new_features, p=self.dp, training=self.training, inplace=True)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, dilation, dp):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, bn_size, growth_rate, dilation, dp)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, pooling=False):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features, eps=EPS))
        self.add_module('relu', nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if pooling:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

# Stacked CTX

class DAE(AE):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    #EPS=1e-04 # TF: 1e-03 => safe but higher loss, PTH: 1e-05 => fail at the end
    def __init__(self, blk_cfg=(6, 12, 24, 16), bn_sz=4, growth_rate=32//8*3,
                 patch_sz=None, center_sz=None, rec_sz=None, nc=None, nf=None, nz=None, bn_eps=None, lrelu=None, dp=None, 
                 variational=None, gmm=None, 
                 cfg=Config()):
        r"""
        nc=3
        nf=64//2
        growth_rate=32//8*3
        """

        super(DAE, self).__init__(patch_sz, center_sz, rec_sz, nc, nf, nz, bn_eps, lrelu, dp, variational, gmm, cfg)
        self.blk_cfg = blk_cfg
        self.bn_sz = bn_sz
        self.growth_rate = growth_rate
        self.build()

    def _build(self):
        ctx_sz = (self.patch_sz - self.center_sz) // 2
        ctx_nc = (self.patch_sz // ctx_sz + self.center_sz // ctx_sz) * 2
        nc = ctx_nc * self.nc #  9
        nf = ctx_nc * self.nf # 12 => 3 x self.nf

        # First convolution
        if True:
            # No subsampling
            self.main = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(nc, nf, kernel_size=3, stride=1, padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(nf, eps=self.bn_eps)),
                ('relu0', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(nc, nf, kernel_size=7, stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(nf, eps=self.bn_eps)),
                ('relu0', nn.LeakyReLU(negative_slope=0.01, inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = nf
        for i, num_layers in enumerate(self.blk_cfg):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=self.bn_sz, growth_rate=self.growth_rate, dilation=2**i, dp=self.dp)
            self.main.add_module('daeblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i == len(self.blk_cfg) - 1:
                # Reconstruction
                trans = _Transition(num_features, 3)
                self.main.add_module('transition%d' % (i + 1), trans)
                num_features = 3
            else:
                trans = _Transition(num_features, num_features // 2)
                self.main.add_module('transition%d' % (i + 1), trans)
                num_features //= 2


        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features, eps=EPS))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)

    #def forward(self, x):
        #features = self.features(x)
        #out = F.relu(features, inplace=True)
        #out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        #out = self.classifier(out)
        #out = self.features(x)
        #return out
