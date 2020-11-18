import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):

    @staticmethod
    def forward(ctx, input, target_size, dimension=0, value=0, pad_start=False):
        ctx.target_size = target_size
        ctx.dimension = dimension
        ctx.value = value
        ctx.pad_start = pad_start

        ctx.num_pad = ctx.target_size - input.size(ctx.dimension)
        assert ctx.num_pad >= 0, 'target size has to be greater than input size'

        ctx.input_size = input.size()

        size = list(input.size())
        size[ctx.dimension] = ctx.target_size
        output = input.new(*tuple(size)).fill_(ctx.value)
        c_output = output

        # crop output
        if ctx.pad_start:
            c_output = c_output.narrow(ctx.dimension, ctx.num_pad, c_output.size(ctx.dimension) - ctx.num_pad)
        else:
            c_output = c_output.narrow(ctx.dimension, 0, c_output.size(ctx.dimension) - ctx.num_pad)

        c_output.copy_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new(*ctx.input_size).zero_()
        cg_output = grad_output

        # crop grad_output
        if ctx.pad_start:
            cg_output = cg_output.narrow(ctx.dimension, ctx.num_pad, cg_output.size(ctx.dimension) - ctx.num_pad)
        else:
            cg_output = cg_output.narrow(ctx.dimension, 0, cg_output.size(ctx.dimension) - ctx.num_pad)

        grad_input.copy_(cg_output)
        return grad_input, None, None, None, None


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False):
    return ConstantPad1d.apply(input, target_size, dimension, value, pad_start)
