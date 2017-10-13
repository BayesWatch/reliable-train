'''Variational Dropout plugins for PyTorch

Based on the version of variational dropout developed in 
"Variational Dropout Sparsifies Deep Neural Networks"
https://arxiv.org/abs/1701.05369

Code ported from our Tensorflow replication:
https://github.com/BayesWatch/tf-variational-dropout
'''

import math

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch.autograd import Variable

from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

def paranoid_log(x, eps=1e-8):
    return torch.log(x+eps)

def clip(x):
    # yes, the value of 8 is arbitrary
    return torch.clamp(x, -8., 8.)

def get_log_alpha(log_sigma2, w):
    log_alpha = clip(log_sigma2 - paranoid_log(torch.pow(w,2)))
    return log_alpha


class Linear(nn.Module): 
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.log_sigma2 = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_sigma2.data.fill_(-10.) # arbitrary

    def forward(self, input):
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        thresh = 3 # arbitrary threshold
        if self.training:
            # when training, add noise for variational mc
            mu = F.linear(input, self.weight, None)
            si = torch.sqrt(F.linear(torch.pow(input,2),
                            torch.exp(log_alpha)*torch.pow(self.weight,2), None)+1e-8)
            noise = Variable(torch.randn(mu.size()))
            a = mu + si*noise
        else:
            # at test, don't add noise, but mask stuff that goes past the threshold
            mask = torch.lt(log_alpha, thresh).float()
            a = F.linear(input, mask*self.weight, None)
        return a + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        transposed = False
        output_padding = _pair(0)
        super(Conv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.log_sigma2 = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.log_sigma2 = Parameter(torch.Tensor(
                    out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.log_sigma2.data.fill_(-10.)

    def forward(self, input):
        log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        thresh = 3 # arbitrary threshold
        if self.training:
            mu = F.conv2d(input, self.weight, None, self.stride, self.padding,
                    self.dilation, self.groups)
            inp2 = torch.pow(input,2)
            w2 = torch.pow(self.weight,2)
            var = F.conv2d(inp2, w2*torch.exp(log_alpha), None, self.stride,
                    self.padding, self.dilation, self.groups) + 1e-8
            si = torch.sqrt(var)
            noise = Variable(torch.randn(mu.size()))
            a = mu + noise*si
        else:
            mask = torch.lt(log_alpha, thresh).float()
            a = F.conv2d(input, mask*self.weight, None, self.stride,
                    self.padding, self.dilation, self.groups)
        return a + self.bias.view(1, self.bias.size(0), 1, 1)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

def test():
    net = Linear(32,10)
    net.train()
    x = torch.randn(1,32)
    y = net(Variable(x))
    print(y.size())
    net.eval()
    y = net(Variable(x))
    print(y.size())
    net = Conv2d(3, 16, 3)
    net.train()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y.size())
    net.eval()
    y = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()
