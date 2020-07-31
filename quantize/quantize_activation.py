# coding=utf-8
"""
"""
import torch
import math
import numpy as np
import torch.nn as nn
from quantize.s_round import s_round
# 量化比特
QUANTIZE_BIT = 8


## 全部量化为8bit
def quantize_activations_gemm_A(activation, scale):

    i = activation/scale
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    q2scale = math.pow(2.0, QUANTIZE_BIT - 1)
    quantize_val = torch.clamp(torch.round(i * q2scale), -n, n - 1)
    # quantize_val = torch.clamp(s_round(i * q2scale), -n, n - 1)


    return quantize_val/q2scale

def quantize_activations_gemm_B(activation):
    """
    量化到８ｂｉｔ小数，
    未加缩放
    :param activation:
    :return:
    """
    i = activation
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    q2scale = n
    quantize_val = torch.clamp(torch.round(i * q2scale), -n, n - 1)

    return quantize_val/q2scale

def quantize_activations_gemm_C(activation):
    """
    模拟量化，量化到1/128的n倍
    :param activation:
    :return:
    """

    i = activation
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    q2scale = n
    quantize_val = torch.round(i * q2scale)

    return quantize_val/q2scale

#量化为其他bit数  eg 12/9
QUANTIZE_BIT_ACT = 12
ACT_POINT = 9

def quantize_activations_gemm(activation):

    n = math.pow(2.0, QUANTIZE_BIT_ACT - 1)
    q2scale = math.pow(2.0, ACT_POINT - 1)
    quantize_val = torch.clamp(torch.round(activation* q2scale), -n, n - 1)

    return quantize_val/q2scale
