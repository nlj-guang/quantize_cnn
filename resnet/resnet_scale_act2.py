'''ResNet in PyTorch.
仅用与cifar数据集， resnet18,resnet34，其他不能用
For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from quantize.quantize_module_scale import  SQWAConv2D, SQWALinear
from scale.scale_act_fun import ReLU6S

def conv3x3(in_planes, out_planes, act_scale, bias_scale, stride=1):
    """3x3 convolution with padding"""
    return SQWAConv2D(in_planes, out_planes,
                      act_scale=act_scale, bias_scale=bias_scale,
                      kernel_size=3, stride=stride,
                      padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, act_scale, bias_scale, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, act_scale[0], bias_scale[0], stride)
        self.conv2 = conv3x3(planes, planes, act_scale[1], bias_scale[1])
        self.relu1 = ReLU6S(inplace=True, scale=bias_scale[0])
        self.relu2 = ReLU6S(inplace=True, scale=bias_scale[1])
        self.relu = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SQWAConv2D(in_planes, self.expansion*planes, kernel_size=1,
                           act_scale=act_scale[0], bias_scale=bias_scale[0], stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = SQWAConv2D(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = SQWAConv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = SQWAConv2D(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                SQWAConv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, act_scale, bias_scale, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = SQWAConv2D(3, 64, act_scale = act_scale[0], bias_scale = bias_scale[0],
                                kernel_size=3, stride=1, padding=1, bias=True)
        # self.relu = ReLU6S(inplace=True, scale= bias_scale[0])
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0],
                                       act_scale=act_scale[1], bias_scale=bias_scale[1], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],
                                       act_scale=act_scale[2], bias_scale=bias_scale[2], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],
                                       act_scale=act_scale[3], bias_scale=bias_scale[3], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],
                                       act_scale=act_scale[4], bias_scale=bias_scale[4], stride=2)
        self.linear = SQWALinear(512*block.expansion, num_classes, scale=act_scale[5])

    def _make_layer(self, block, planes, num_blocks, act_scale, bias_scale, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        i = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                                act_scale = act_scale[i], bias_scale = bias_scale[i], stride = stride))
            i +=1
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

act_scale = {
    '18': [4, [[1,1], [1,1]], [[1,1], [1,1]],
           [[1,1], [1,1]], [[1,1], [1,1]], 1],
    '34': [4, [[1,1], [1,1], [1,1]], [[1,1], [1,1], [1,1], [1,1]],
           [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1]], [[1,1], [1,1], [1,1]], 1],
    '50': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    '101': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    '152': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
}
bias_scale = {
    '18': [4, [[4, 4], [4, 4]], [[4, 4], [4, 4]],
           [[4, 4], [4, 4]], [[4, 4], [4, 4]], 4],
    '34': [4, [[4, 4], [4, 4], [4, 4]], [[4, 4], [4, 4], [4, 4], [4, 4]],
           [[4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4] ], [[4, 4], [4, 4], [4, 4]], 4],
    '50': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    '101': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
    '152': [4, [4, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
}

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], act_scale=act_scale['18'], bias_scale=bias_scale['18'],
                  num_classes= num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], act_scale=act_scale['34'], bias_scale=bias_scale['34'],
                  num_classes= num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], act_scale=act_scale['50'], bias_scale=bias_scale['50'],
                  num_classes= num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], act_scale=act_scale['101'], bias_scale=bias_scale['101'],
                  num_classes= num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], act_scale=act_scale['152'], bias_scale=bias_scale['152'],
                  num_classes= num_classes)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
