# coding=utf-8
"""
"""
import torch
import math
from quantize.s_round import s_round

# 量化比特
QUANTIZE_BIT = 24
POINT = 16

def quantize_bias_gemm(bias):
    n = math.pow(2.0, QUANTIZE_BIT - 1)
    q2scale = math.pow(2.0, POINT - 1)
    quantize_val = torch.clamp(torch.round(bias * q2scale), -n, n - 1)
    # quantize_val = torch.clamp(s_round(bias * q2scale), -n, n - 1)


    return (quantize_val) / q2scale
