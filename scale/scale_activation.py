# coding=utf-8
"""
"""
import torch
import torch.nn as nn

class ReLU6S(nn.Hardtanh):
    r"""Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`

    Examples::
        >>> m = nn.ReLUS()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False, scale=1):
        super(ReLU6S, self).__init__(0, 6/scale, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str



