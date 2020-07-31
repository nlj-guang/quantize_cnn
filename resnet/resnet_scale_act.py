# coding=utf-8
"""
！！！！！未完成！！！！！
"""
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from quantize.quantize_module_ import QWConv2D, QWAConv2D, QWALinear, Scalar
from scale.scale_module_ import FoldData
from quantize.quantize_method import quantize_weights_bias_gemm
from quantize.quantize_weight import quantize_weight_gemm_S
from quantize.quantize_activation import quantize_activations_gemm_A, quantize_activations_gemm
from quantize.quantize_bias import quantize_bias_gemm
from scale.scale_act_fun import ReLU6S


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QWAConv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes,  act_scale, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.relu1 = ReLU6S(inplace=False, scale=act_scale[0])
        self.relu2 = ReLU6S(inplace=False, scale=act_scale[1])
        self.relu = nn.ReLU6(inplace=False)
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
        self.act_scale = act_scale
        self.bias_scale = act_scale


    def forward(self, x):
        x = quantize_activations_gemm_A(x, self.act_scale[0])
        residual = x

        out1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm_S(self.conv1.weight)
        conv1_bias = quantize_bias_gemm(self.conv1.bias)/(conv1_scale*self.bias_scale[0])
        out = F.conv2d(x, conv1_weight, conv1_bias, stride=self.stride, padding=1)*conv1_scale
        out = self.relu1(out)
        out = out*self.act_scale

        out = quantize_activations_gemm_A(out, self.act_scale[1])
        out2 = self.conv2(out)
        conv2_weight, conv2_scale = quantize_weight_gemm_S(self.conv2.weight)
        conv2_bias = quantize_bias_gemm(self.conv2.bias)/(conv2_scale*self.bias_scale[1])
        out = F.conv2d(out, conv2_weight, conv2_bias, stride=1, padding=1) * conv2_scale
        out = out*self.act_scale[1]

        out3 = self.shortcut(residual)
        if self.downsample:
            short_weight, short_scale = quantize_weight_gemm_S(self.shortcut[0].weight)
            short_bias = quantize_bias_gemm(self.shortcut[0].bias)/(short_scale*self.bias_scale[0])
            residual = F.conv2d(residual, short_weight, short_bias, stride=self.stride)*short_scale
            residual = residual*self.act_scale[0]

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, act_scale, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QWAConv2D(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = QWAConv2D(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=True)
        self.conv3 = QWAConv2D(planes, planes * 4, kernel_size=1, bias=True)
        self.relu1 = ReLU6S(inplace=False, scale=act_scale[0])
        self.relu2 = ReLU6S(inplace=False, scale=act_scale[1])
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
        self.act_scale = act_scale
        self.bias_scale = act_scale

    def forward(self, x):
        x = quantize_activations_gemm_A(x, self.act_scale[0])
        residual = x

        out1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm_S(self.conv1.weight)
        conv1_bias = quantize_bias_gemm(self.conv1.bias)/(conv1_scale*self.bias_scale[0])
        out = F.conv2d(x, conv1_weight, conv1_bias) * conv1_scale
        out = self.relu1(out)
        out = out*self.act_scale[0]

        out = quantize_activations_gemm_A(out, self.act_scale[1])
        out2 = self.conv2(out)
        conv2_weight, conv2_scale = quantize_weight_gemm_S(self.conv2.weight)
        conv2_bias = quantize_bias_gemm(self.conv2.bias)/(conv2_scale*self.bias_scale[1])
        out = F.conv2d(out, conv2_weight, conv2_bias, stride=self.stride, padding=1) * conv2_scale
        out = self.relu2(out)
        out = out*self.act_scale[1]

        out =quantize_activations_gemm_A(out, self.act_scale[2])
        out3 = self.conv3(out)
        conv3_weight, conv3_scale = quantize_weight_gemm_S(self.conv3.weight)
        conv3_bias = quantize_bias_gemm(self.conv3.bias)/(conv2_scale*self.bias_scale[2])
        out = F.conv2d(out, conv3_weight, conv3_bias,padding=1) * conv3_scale
        out = out * self.act_scale[2]

        out4 = self.shortcut(residual)
        if self.downsample:
            short_weight, short_scale = quantize_weight_gemm_S(self.shortcut[0].weight)
            short_bias = quantize_bias_gemm(self.shortcut[0].bias) / (short_scale*self.act_scale[0])
            residual = F.conv2d(residual, short_weight, short_bias, stride=self.stride) * short_scale
            residual = residual * self.act_scale[0]

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, qblock, layers, scale, num_classes=100):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = QWConv2D(3, 64, kernel_size=3, stride=1, padding=1,
                              bias=True)
        self.relu1 = ReLU6S(inplace=False, scale=scale[0])
        self.relu = nn.ReLU6(inplace=True)
        self.scale = scale[0]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(qblock, 64, layers[0], scale[1])
        self.layer2 = self._make_layer(qblock, 128, layers[1], scale[2], stride=2)
        self.layer3 = self._make_layer(qblock, 256, layers[2], scale[3], stride=2)
        self.layer4 = self._make_layer(qblock, 512, layers[3], scale[4], stride=2)
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

    def _make_layer(self, block, planes, blocks, scale, stride=1):

        strides = [stride] + [1] * (blocks - 1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, scale[i], stride))
            self.in_planes = planes * block.expansion
            i = i+1
        return nn.Sequential(*layers)

    def forward(self, x):

        x =quantize_activations_gemm_A(x, self.scale)
        x1 = self.conv1(x)
        conv1_weight, conv1_scale = quantize_weight_gemm_S(self.conv1.weight)
        conv1_bias = quantize_bias_gemm(self.conv1.bias)/conv1_scale
        x = F.conv2d(x, conv1_weight, conv1_bias, stride=1, padding=1) * conv1_scale
        x = self.relu1(x)
        x = x*self.scale

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

#　权值缩放系数
scale = {
    'A': [4, [[1,1], [1,1]], [[1,1], [1,1]], [[1,1], [1,1]], [[1,1], [1,1]]],
    'B': [4, [4, 1], [1, 1], [1, 1], [1, 1]],
    'C': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'D': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    'E': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
}


def resnet18( **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], scale= scale['A'],  **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], scale= scale['B'], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], scale= scale['C'], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], scale= scale['D'], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], scale= scale['E'], **kwargs)
    return model
