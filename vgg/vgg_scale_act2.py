'''VGG11/13/16/19 in Pytorch.'''
"""
    VGG网络权值和激活值一起量化为8bit小数
    激活值的量化加入缩放因子

"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from quantize.quantize_module_scale import SQWALinear, SQWALinear_B, SQWAConv2D
from quantize.quantize_method import quantize_weights_bias_gemm
from quantize.quantize_weight import quantize_weight_gemm_S
from quantize.quantize_activation import quantize_activations_gemm_A, quantize_activations_gemm
from quantize.quantize_bias import quantize_bias_gemm
from scale.scale_act_fun import ReLU6S
__all__ = [
    'VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
]

class VGG(nn.Module):

    def __init__(self, features, act_scale, bias_scale, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Linear(512,512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512,512),
            # nn.ReLU(True),
            # nn.Linear(512, num_classes),
            # QWALinear(512, num_classes),
            SQWALinear_B(512, num_classes, act_scale=act_scale[-1], bias_scale=bias_scale[-1]),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, act_scale, bias_scale):
    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = SQWAConv2D(in_channels, v, act_scale=act_scale[i], bias_scale=bias_scale[i],
                                kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU6(inplace=True)]
            in_channels = v
            i = i + 1
    return nn.Sequential(*layers)


act_scale = {
    'A': [4, 4, 1/4, 1/2, 1, 1, 1, 1/2, 4],
    'B': [4, 1, 1, 1, 1, 1, 1, 1, 0.5, 1],
    'D': [4, 2, 1/2, 1/2, 2, 1/2, 1, 2, 1, 1/2, 1, 1, 1, 2],
    'E': [4, 2, 0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1],
}

bias_scale = {
    'A': [4, 16, 4, 2, 2, 2, 2, 1, 4],
    'B': [4, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 4],
    'D': [4, 8, 4, 2, 4, 2, 2, 4, 4, 2, 2, 2, 2, 4],
    'E': [4, 2, 0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 4],
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A'], act_scale['A'], bias_scale['A']),
                act_scale['A'], bias_scale['A'], **kwargs)
    return model

def vgg13( **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B'], act_scale['B'], bias_scale['B']),
                act_scale['B'], bias_scale['B'], **kwargs)

    return model

def vgg16( **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D'], act_scale['D'], bias_scale['D']),
                act_scale['D'], bias_scale['D'], **kwargs)

    return model

def vgg19( **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], act_scale['E'], bias_scale['E']),
                act_scale['E'], bias_scale['E'], **kwargs)

    return model