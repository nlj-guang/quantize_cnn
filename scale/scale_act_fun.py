# coding=utf-8
"""
"""
import torch
import torch.nn as nn

class ReLU6S(nn.Hardtanh):
    r"""Applies the element-wise function :math:`\text{ReLU6}(x) = \min(\max(0,x), 6)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

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



