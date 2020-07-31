'''VGG11/13/16/19 in Pytorch.'''
"""
    VGG网络只量化激活值为8bit小数
    激活值的量化加入缩放因子
    该网络准备利用训练好网络进行小规模训练
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from quantize.quantize_module_ import QWLinear, QALinear, QWALinear, Scalar
from quantize.quantize_module_scale import SQALinear, SQWALinear
from quantize.quantize_method import quantize_weights_bias_gemm
from quantize.quantize_weight import quantize_weight_gemm_S
from quantize.quantize_activation import quantize_activations_gemm_A, quantize_activations_gemm
from quantize.quantize_bias import quantize_bias_gemm
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


def conv3x3(in_planes, out_planes, bias):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=bias)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, act_scale):
        super(BasicBlock, self).__init__()
        self.conv = conv3x3(in_planes, planes, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU6(inplace=False)
        self.act_scale = act_scale

    def forward(self, x):
        x1 = quantize_activations_gemm_A(x, self.act_scale)
        out = self.conv(x1)
        out = out*self.act_scale
        out = self.bn(out)
        # out = quantize_activations_gemm_A(out, self.act_scale)
        # out = out*self.act_scale
        out = self.relu(out)

        return out


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True, batch_norm=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            # nn.Linear(512,512),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(512,512),
            # nn.ReLU(True),
            # nn.Linear(512, num_classes),
            QALinear(512, num_classes),
            # SQALinear(512, num_classes, 4),
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


def make_layers(cfg, scale, batch_norm=False):
    layers = []
    in_channels = 3
    i = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers.append(BasicBlock(in_planes=in_channels, planes=v,
                                     act_scale=scale[i]))
            in_channels = v
            i = i + 1
    return nn.Sequential(*layers)


scale = {
    'A': [4, 4, 4, 2, 4, 4, 4, 1],
    'B': [4, 1, 1, 1, 1, 1, 1, 1, 0.5, 1],
    'D': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 1],
    'E': [4, 2, 0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1],
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], scale['A']), **kwargs)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], scale['B']), **kwargs)

    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)

    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], scale['D']), **kwargs)

    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)

    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], scale['E']), **kwargs)

    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

    return model