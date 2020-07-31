# coding=utf-8
"""
1. 将卷积层, 除了第一层使用 QWConv2D(不量化输出, 不然性能下降10个百分点), 全部使用QWACvon2D
2. 线性层全部使用 QWALinear, 线性层所占的参数比例在 resnet18中占据 4.4%, resnet50中占据 8%, 不量化的话会有大约 0.4个百分点的性能提升
3. 在全连接层送入 softmax 之前, 加一个标量层, 做 softmax 的软化??
"""
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from quantize.quantize_module_ import QWConv2D, QWAConv2D, QWALinear, Scalar
from quantize.quantize_method import quantize_weights_bias_gemm
from scale.scale_quantize_method import quantize_weight_gemm, quantize_bias_gemm, quantize_activations_gemm


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QWAConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.relu = nn.ReLU6(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = True
            self.shortcut = nn.Sequential(
                QWAConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )
        else:
            self.downsample = False

        self.stride = stride

    def forward(self, x):
        # x = quantize_activations_gemm(x)
        residual = x

        out1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm(self.conv1.weight)
        # conv1_weight = quantize_weights_bias_gemm(self.conv1.weight)
        # conv1_scale = 1
        conv1_bias = quantize_bias_gemm(self.conv1.bias/conv1_scale)
        out = F.conv2d(x, conv1_weight, conv1_bias, stride=self.stride, padding=1)*conv1_scale
        out = self.relu(out)

        out = quantize_activations_gemm(out)

        out1 = self.conv2(out)
        conv2_weight, conv2_scale = quantize_weight_gemm(self.conv2.weight)
        # conv2_weight = quantize_weights_bias_gemm(self.conv2.weight)
        # conv2_scale = 1
        conv2_bias = quantize_bias_gemm(self.conv2.bias)/conv2_scale
        out = F.conv2d(out, conv2_weight, conv2_bias, stride=1, padding=1) * conv2_scale
        out = quantize_activations_gemm(out)

        out1 += self.shortcut(residual)
        if self.downsample:
            short_weight, short_scale = quantize_weight_gemm(self.shortcut[0].weight)
            short_bias = quantize_bias_gemm(self.shortcut[0].bias)/short_scale
            residual = F.conv2d(residual, short_weight, short_bias, stride=self.stride)*short_scale

        out += residual

        out = self.relu(out)
        out = quantize_activations_gemm(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QWAConv2D(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = QWAConv2D(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=True)
        self.conv3 = QWAConv2D(planes, planes * 4, kernel_size=1, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = True
            self.shortcut = nn.Sequential(
                QWAConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True)
            )
        else:
            self.downsample = False

        self.stride = stride

    def forward(self, x):
        x = quantize_activations_gemm(x)
        residual = x

        out1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm(self.conv1.weight)
        # conv1_weight = quantize_weights_bias_gemm(self.conv1.weight)
        # conv1_scale = 1
        conv1_bias = quantize_bias_gemm(self.conv1.bias)/conv1_scale
        out = F.conv2d(x, conv1_weight, conv1_bias) * conv1_scale
        out = self.relu(out)
        out = quantize_activations_gemm(out)

        out1 = self.conv2(out)
        conv2_weight, conv2_scale = quantize_weight_gemm(self.conv2.weight)
        # conv2_weight = quantize_weights_bias_gemm(self.conv2.weight)
        # conv2_scale = 1
        conv2_bias = quantize_bias_gemm(self.conv2.bias)/conv2_scale
        out = F.conv2d(out, conv2_weight, conv2_bias, stride=self.stride, padding=1) * conv2_scale
        out = self.relu(out)
        out = quantize_activations_gemm(out)

        out1 = self.conv3(out)
        conv3_weight, conv3_scale = quantize_weight_gemm(self.conv3.weight)
        # conv3_weight = quantize_weights_bias_gemm(self.conv3.weight)
        # conv3_scale = 1
        conv3_bias = quantize_bias_gemm(self.conv3.bias)/conv3_scale
        out = F.conv2d(out, conv3_weight, conv3_bias,padding=1) * conv3_scale
        out = quantize_activations_gemm(out)

        out1 += self.shortcut(residual)
        if self.downsample:
            short_weight, short_scale = quantize_weight_gemm(self.shortcut[0].weight)
            short_bias = quantize_bias_gemm(self.shortcut[0].bias) / short_scale
            residual = F.conv2d(residual, short_weight, short_bias, stride=self.stride) * short_scale

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, qblock, layers, num_classes=100):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = QWConv2D(3, 64, kernel_size=3, stride=1, padding=1,
                              bias=True)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(qblock, 64, layers[0])
        self.layer2 = self._make_layer(qblock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(qblock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(qblock, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = QWALinear(512 * qblock.expansion, num_classes)  # 修改
        # self.scalar = Scalar()  # 修改

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x =quantize_activations_gemm(x)

        x1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm(self.conv1.weight)
        # conv1_weight = quantize_weights_bias_gemm(self.conv1.weight)
        # conv1_scale = 1
        conv1_bias = quantize_bias_gemm(self.conv1.bias)/conv1_scale
        x = F.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1) * conv1_scale
        x = self.relu(x)
        x = quantize_activations_gemm(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        # x = quantize_activations_gemm(x)
        # x = self.scalar(x)  # 修改

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
