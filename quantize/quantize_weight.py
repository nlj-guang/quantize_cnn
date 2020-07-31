# coding=utf-8
"""
"""
import torch
import math
from quantize.s_round import s_round

# 量化比特
QUANTIZE_BIT = 8

def quantize_weight_gemm(weight):
    """
    直接量化8bit小数
    :param weight:
    :return:
    """
    i = weight
    q2scale = math.pow(2.0, QUANTIZE_BIT - 1)
    quantize_val = torch.clamp(torch.round(i * q2scale), -q2scale, q2scale - 1)

    return (quantize_val) / q2scale

def quantize_weight_gemm_S(weight):
    """
    缩放量化，返回量化值和缩放值
    :param weight:
    :return:
    """
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    i = weight
    a = torch.abs(torch.max(i))
    b = torch.abs(torch.min(i))
    if a > b:
        m = a
    else:
        m = b

    scale = math.pow(2, (math.log2(m).__round__()))
    q2scale = math.pow(2, (math.log2(n)).__round__())
    mid = i/scale
    quantize_val = torch.clamp(torch.round(mid * q2scale), -q2scale, q2scale - 1)
    # quantize_val = torch.clamp(s_round(mid * q2scale), -q2scale, q2scale - 1)


    return quantize_val/q2scale, scale

def quantize_weight_gemm_S_bit(weight):
    """
    位数修改， 缩放量化，返回量化值和缩放值
    :param weight:
    :return:
    """
    bit = 4
    n = math.pow(2.0, bit - 1)
    i = weight
    a = torch.abs(torch.max(i))
    b = torch.abs(torch.min(i))
    if a > b:
        m = a
    else:
        m = b

    scale = math.pow(2, (math.log2(m).__round__()))
    q2scale = math.pow(2, (math.log2(n)).__round__())
    mid = i/scale
    quantize_val = torch.clamp(torch.round(mid * q2scale), -q2scale, q2scale - 1)

    return quantize_val/q2scale, scale


def quantize_weight_gemm_Log(weight):
    """
    缩放量化，返回量化值和缩放值
    :param weight:
    :return:
    """
    i = weight
    s = torch.sign(i)
    a = torch.abs(torch.max(i))
    b = torch.abs(torch.min(i))
    if a > b:
        m = a
    else:
        m = b
    scale = math.pow(2, (math.log2(m).__round__()))
    z = torch.clamp(torch.round((torch.log2(torch.abs(i/scale)))), -7, 0)
    quantize_val = s * torch.pow(2, z)

    return quantize_val, scale


def quantize_weight_gemm_C(weight):
    """
    直接缩放
    :param weight:
    :return:
    """
    i = weight
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    q2scale = n
    quantize_val = torch.clamp(torch.round(i * q2scale), -n, n - 1)

    return (quantize_val) / q2scale
