# coding=utf-8
import torch
import torch.nn as nn
from quantize.quantize_weight import quantize_weight_gemm, quantize_weight_gemm_S, \
    quantize_weight_gemm_S_bit, quantize_weight_gemm_Log
from quantize.quantize_activation import quantize_activations_gemm_A, quantize_activations_gemm
from quantize.quantize_bias import quantize_bias_gemm
import torch.nn.functional as F


class SQWAConv2D(torch.nn.Conv2d):
    """

    权值和激活值的缩放都加载在里面
    """
    def __init__(self, n_channels, out_channels, kernel_size,
                 act_scale, bias_scale,
                 stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(SQWAConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        self.act_scale = act_scale
        self.bias_scale = bias_scale

    def forward(self, input):
        qweight, scale = quantize_weight_gemm_S(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/(scale*self.bias_scale))
        else:
            qbias = None

        qinput = quantize_activations_gemm_A(input, self.act_scale)
        return F.conv2d(qinput, qweight, qbias, self.stride,
                        self.padding, self.dilation, self.groups)*scale

class SQWALinear(nn.Linear):

    def __init__(self, in_features, out_features, scale, bias=True):
        super(SQWALinear, self).__init__(in_features, out_features, bias)
        self.scale = scale

    def forward(self, input):
        qinput = quantize_activations_gemm_A(input, scale=self.scale)
        qweight= quantize_weight_gemm(self.weight)
        # qweight, scale = quantize_weight_gemm_S_bit(self.weight)
        # qweight, scale = quantize_weight_gemm_Log(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/(self.scale))
        else:
            qbias = None

        return F.linear(qinput, qweight, qbias)

class SQWALinear_B(nn.Linear):

    def __init__(self, in_features, out_features, act_scale, bias_scale, bias=True):
        super(SQWALinear_B, self).__init__(in_features, out_features, bias)
        self.act_scale = act_scale
        self.bias_scale = bias_scale

    def forward(self, input):
        qinput = quantize_activations_gemm_A(input, scale=self.act_scale)
        qweight = quantize_weight_gemm(self.weight)
        # qweight, scale = quantize_weight_gemm_S_bit(self.weight)
        # qweight,scale = quantize_weight_gemm_Log(self.weight)
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/(self.bias_scale))
        else:
            qbias = None

        return F.linear(qinput, qweight, qbias)


class SQALinear(nn.Linear):

    def __init__(self, in_features, out_features, scale, bias=True):
        super(SQALinear, self).__init__(in_features, out_features, bias)
        self.scale = scale

    def forward(self, input):
        qinput = quantize_activations_gemm_A(input, self.scale)
        weight = self.weight
        if self.bias is not None:
            qbias = quantize_bias_gemm(self.bias/self.scale)
        else:
            qbias = None

        return F.linear(qinput, weight, qbias)

