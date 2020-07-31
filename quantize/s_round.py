import torch
import math
import random


def s_round(x):
    """
    随机舍入，最小的精度是1
    :param x:
    :return:
    """
    x_f = torch.floor(x)
    p = x-x_f
    b = torch.bernoulli_(p)
    x_f.__add__(b)
    return x_f

# test()