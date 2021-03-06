from collections import namedtuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
import scipy.optimize as opt

__all__ = ['CPTConv2d']

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits']) # 由三个部分组成的表示

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)

def _deflatten_as(x, x_full):
    shape = list(x.shape) + [1] * (x_full.dim() - x.dim())
    return x.view(*shape)

def mse(x, alpha, sign, xmax):  
    alpha = torch.from_numpy(alpha).to(x.device)
    if sign:
        x_clip = (x / alpha).clamp(0, xmax)
    else:
        x_clip = (x / alpha).clamp(-xmax, xmax)
    x_q = x_clip.round()
    x_q = x_q * alpha
    return (((x_q - x) ** 2).sum() / x.numel()).cpu().item()

def get_alpha(x, sign, xmax):
    # method1
    # print('the x shape is : ' , x.shape)
    alpha = x.view(x.shape[0], -1).max(axis=1)[0].topk(10)[0][-1] / xmax

    mmse = lambda param: mse(x, param, sign=sign, xmax=xmax)
    res = opt.minimize(mmse, (alpha.detach().cpu().numpy()), method='powell',
                       options={'disp': False, 'ftol': 0.05, 'maxiter': 100, 'maxfev': 100})
    return torch.from_numpy(res.x).abs()


def calculate(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)

        # range_values = max_values - min_values # 得到最大最小值之间的范围结果，之后注册一个QParams的类
        # return QParams(range=range_values, zero_point=min_values,
        #                num_bits=num_bits)

def calculate_qparams(x, num_bits, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, reduce_type='mean', keepdim=False,
                      true_zero=False):
    with torch.no_grad():
        x_flat = x.flatten(*flatten_dims)
        if x_flat.dim() == 1: # 如果变成一个一维的数据结果，将数据摊开的维度数目
            min_values = _deflatten_as(x_flat.min(), x)
            max_values = _deflatten_as(x_flat.max(), x)
        else:
            min_values = _deflatten_as(x_flat.min(-1)[0], x)
            max_values = _deflatten_as(x_flat.max(-1)[0], x)

        if reduce_dim is not None: # 如果reduce_dim 不是None，那么进行相应的处理（缩减相应的维度），但是keepdim 
            if reduce_type == 'mean':
                min_values = min_values.mean(reduce_dim, keepdim=keepdim)
                max_values = max_values.mean(reduce_dim, keepdim=keepdim)
            else:
                min_values = min_values.min(reduce_dim, keepdim=keepdim)[0]
                max_values = max_values.max(reduce_dim, keepdim=keepdim)[0]

        range_values = max_values - min_values # 得到最大最小值之间的范围结果，之后注册一个QParams的类
        return QParams(range=range_values, zero_point=min_values,
                       num_bits=num_bits)


# 这个类定义了一个均匀量化的基本处理，包括了前向和反向的过程
class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN,
                reduce_dim=0, dequantize=True, signed=False, stochastic=False, inplace=False):

        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if qparams is None:
            assert num_bits is not None, "either provide qparams of num_bits to quantize"
            qparams = calculate_qparams(
                input, num_bits=num_bits, flatten_dims=flatten_dims, reduce_dim=reduce_dim) 
            # 如果没有直接给出qparams，那么就通过计算来得到相应的结果

        zero_point = qparams.zero_point
        num_bits = qparams.num_bits
        qmin = -(2. ** (num_bits - 1)) if signed else 0. # 是进行有符号还是无符号的量化
        qmax = qmin + 2. ** num_bits - 1.
        scale = qparams.range / (qmax - qmin) # 那么可以得到相应的scale，也就是直接通过range来得到

        min_scale = torch.tensor(1e-8).expand_as(scale).cuda() # 最小的scale 
        scale = torch.max(scale, min_scale) # 然后设置一个scale 的 比较

        with torch.no_grad():
            output.add_(qmin * scale - zero_point).div_(scale)
            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
            # quantize
            output.clamp_(qmin, qmax).round_()

            if dequantize:
                output.mul_(scale).add_(
                    zero_point - qmin * scale)  # dequantize
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output # STE 方法， 量化部分不改变实际的权重的梯度
        return grad_input, None, None, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @staticmethod
    def forward(ctx, input, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD,
                reduce_dim=0, dequantize=True, signed=False, stochastic=True):
        ctx.num_bits = num_bits
        ctx.qparams = qparams
        ctx.flatten_dims = flatten_dims
        ctx.stochastic = stochastic
        ctx.signed = signed
        ctx.dequantize = dequantize
        ctx.reduce_dim = reduce_dim
        ctx.inplace = False
        return input

    @staticmethod
    def backward(ctx, grad_output):
        qparams = ctx.qparams
        with torch.no_grad():
            if qparams is None:
                assert ctx.num_bits is not None, "either provide qparams of num_bits to quantize"
                qparams = calculate_qparams(
                    grad_output, num_bits=ctx.num_bits, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                    reduce_type='extreme')

            grad_input = quantize(grad_output, num_bits=None,
                                  qparams=qparams, flatten_dims=ctx.flatten_dims, reduce_dim=ctx.reduce_dim,
                                  dequantize=True, signed=ctx.signed, stochastic=ctx.stochastic, inplace=False)
        return grad_input, None, None, None, None, None, None, None

def quantize(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN, reduce_dim=0, dequantize=True, signed=False,
             stochastic=False, inplace=False):
    # 这里有两种量化方式，一种是通过qparams 来进行控制，而另一种是通过num_bits 来进行控制
    if qparams: # 当有相应的范围参数的时候
        if qparams.num_bits: # 如果设置了相应的num_bits
            return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic, inplace)
    elif num_bits:
        return UniformQuantize().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed, stochastic,
                                       inplace)

    return x


def quantize_grad(x, num_bits=None, qparams=None, flatten_dims=_DEFAULT_FLATTEN_GRAD, reduce_dim=0, dequantize=True,
                  signed=False, stochastic=True):
    if qparams:
        if qparams.num_bits:
            return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                               stochastic)
    elif num_bits:
        return UniformQuantizeGrad().apply(x, num_bits, qparams, flatten_dims, reduce_dim, dequantize, signed,
                                           stochastic)

    return x


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, shape_measure=(1,), flatten_dims=_DEFAULT_FLATTEN,
                 inplace=False, dequantize=True, stochastic=False, momentum=0.9, measure=False):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_zero_point', torch.zeros(*shape_measure))
        self.register_buffer('running_range', torch.zeros(*shape_measure))
        self.measure = measure
        if self.measure:
            self.register_buffer('num_measured', torch.zeros(1))
        self.flatten_dims = flatten_dims
        self.momentum = momentum
        self.dequantize = dequantize
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input, num_bits, qparams=None):

        if self.training or self.measure:
            if qparams is None:
                qparams = calculate_qparams(
                    input, num_bits=num_bits, flatten_dims=self.flatten_dims, reduce_dim=0, reduce_type='extreme')
            with torch.no_grad():
                if self.measure:
                    momentum = self.num_measured / (self.num_measured + 1)
                    self.num_measured += 1
                else:
                    momentum = self.momentum
                self.running_zero_point.mul_(momentum).add_(
                    qparams.zero_point * (1 - momentum))
                self.running_range.mul_(momentum).add_(
                    qparams.range * (1 - momentum))
        else:
            qparams = QParams(range=self.running_range,
                              zero_point=self.running_zero_point, num_bits=num_bits)
        if self.measure:
            return input
        else:
            q_input = quantize(input, qparams=qparams, dequantize=self.dequantize,
                               stochastic=self.stochastic, inplace=self.inplace)
            return q_input


class CPTConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CPTConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)

        self.quantize_input = QuantMeasure(shape_measure=(1, 1, 1, 1), flatten_dims=(1, -1))
        self.stride = stride

    def forward(self, input, actbits, wbits, gbits):
        if actbits == 0 and wbits==0:
            output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return output

        if self.bias is not None:
            qbias = quantize(
                self.bias, num_bits=self.num_bits,
                flatten_dims=(0, -1))
        else:
            qbias = None

        weight_qparams = calculate_qparams(self.weight, num_bits=wbits, flatten_dims=(1, -1),
                                           reduce_dim=None)
        qweight = quantize(self.weight, qparams=weight_qparams)

        qinput = self.quantize_input(input, actbits)
        output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
        output = quantize_grad(output, num_bits=gbits, flatten_dims=(1, -1))

        return output

# if __name__ == '__main__':
#     x = torch.rand(2, 3)
#     x_q = quantize(x, flatten_dims=(-1), num_bits=8, dequantize=True)
#     print(x)
#     print(x_q)

if __name__ == '__main__':
    x = torch.randn(32, 3, 16, 16)
    calculate(x, num_bits=8, flatten_dims=_DEFAULT_FLATTEN_GRAD)
