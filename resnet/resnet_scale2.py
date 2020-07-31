'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantize_module_bit import QWAConv2D_bit,QWALinear_bit

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return QWAConv2D_bit(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QWAConv2D_bit(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = (self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = QWAConv2D_bit(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = QWAConv2D_bit(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = QWAConv2D_bit(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QWAConv2D_bit(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = QWAConv2D_bit(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = QWALinear_bit(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes= num_classes)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes= num_classes)

def resnet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes= num_classes)

def resnet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes= num_classes)

def resnet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes= num_classes)


def test():
    net = resnet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
