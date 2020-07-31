# coding=utf-8
"""
"""
import torch
import math
import numpy as np
import torch.nn as nn

# 量化比特
QUANTIZE_BIT = 8
# 常用12bit/9bit，或16bit/9bit
QUANTIZE_BIT_ACT = 12
ACT_POINT = 9

def quantize_weight_gemm(weight):
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    i = weight
    # mean = torch.mean(i)
    # var = torch.var(i)
    # a = torch.abs(mean + 500*var)
    # b = torch.abs(mean - 500*var)
    a = torch.abs(torch.max(i))
    b = torch.abs(torch.min(i))
    if a > b:
        m = a
    else:
        m = b

    scale = math.pow(2, (math.log2(m).__round__()))
    q2scale = math.pow(2, (math.log2(n)).__round__())

    # mid = torch.clamp(i / scale, -1, 1)
    mid = i/scale
    # quantize_val = torch.round(mid * q2scale)
    # quantize_val = torch.clamp(torch.round(i*q2scale ), -n, n-1)
    quantize_val = torch.clamp(torch.round(mid * q2scale), -q2scale, q2scale - 1)
    return quantize_val/q2scale, scale


class QuantizeGEMM_Activate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):

        # QUANTIZE_BIT_ACT = 12
        n = math.pow(2.0, QUANTIZE_BIT_ACT - 1)
        q2scale = math.pow(2.0, ACT_POINT - 1)
        quantize_val = torch.clamp(torch.round(i*q2scale ), -n, n-1)

        return (quantize_val)/q2scale

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs

quantize_gemm_activate = QuantizeGEMM_Activate.apply
quantize_gemm_bias = QuantizeGEMM_Activate.apply

def quantize_activations_gemm(activation):
    return quantize_gemm_activate(activation)

def quantize_bias_gemm(bias):
    return quantize_gemm_bias(bias)

