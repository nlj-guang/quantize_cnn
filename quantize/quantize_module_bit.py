# coding=utf-8

import torch
import torch.nn as nn
from scale.scale_quantize_method import quantize_activations_gemm, quantize_bias_gemm
from quantize.quantize_weight import quantize_weight_gemm_S, quantize_weight_gemm_C
import torch.nn.functional as F
"""
 weight 8bit +scale
 activation 12bit 
"""


class QWAConv2D_bit(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QWAConv2D_bit, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

    def forward(self, input):
        qweight, scale = quantize_weight_gemm_S(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/scale)
        else:
            qbias = None

        qinput = quantize_activations_gemm(input)
        out = F.conv2d(qinput, qweight, qbias, self.stride,
                    self.padding, self.dilation, self.groups)*scale

        return out

class QWALinear_bit(nn.Linear):
    """
    weight 有缩放
    """

    def __init__(self, in_features, out_features, bias=True):
        super(QWALinear_bit, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qweight, scale = quantize_weight_gemm_S(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/scale)
        else:
            qbias = None
        qinput = quantize_activations_gemm(input)
        out = F.linear(qinput, qweight, qbias)*scale
        # out = quantize_activations_gemm(out)
        return out

class QWALinear_B(nn.Linear):
    """
    weight 无缩放
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QWALinear_B, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qweight = quantize_weight_gemm_C(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias)
        else:
            qbias = None
        qinput = quantize_activations_gemm(input)
        out = F.linear(qinput, qweight, qbias)
        # out = quantize_activations_gemm(out)
        return out