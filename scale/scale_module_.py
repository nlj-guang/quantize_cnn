# coding=utf-8
import torch
import torch.nn as nn
from quantize.quantize_method import quantize_weights_bias_gemm
from scale.scale_quantize_method import quantize_activations_gemm, quantize_bias_gemm, quantize_weight_gemm
import torch.nn.functional as F

"卷积层加入了缩放因子"

class SQWConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SQWConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input, scale, bias):
        """
        关键在于使用函数 F.conv2d, 而不是使用模块 nn.ConV2d
        """
        qweight = quantize_weights_bias_gemm(self.weight)
        qbias = None
        return F.conv2d(input, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)*scale + bias


class SQWAConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SQWAConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input, scale, bias):
        qweight = quantize_weights_bias_gemm(self.weight)
        qbias = None
        qinput = quantize_activations_gemm(input)
        return F.conv2d(qinput, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)*scale + bias


def FoldData(real_conv, real_bn):
    delta = torch.sqrt(real_bn.running_var + real_bn.eps)
    FoldScale = torch.div(real_bn.weight, delta)
    FlodedWeight = nn.Parameter(torch.mul(real_conv.weight,
        torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(FoldScale, 1), 2), 3)))
    FoldedBias = 0-torch.div(torch.mul(real_bn.weight, real_bn.running_mean), delta)
    return FlodedWeight, FoldedBias