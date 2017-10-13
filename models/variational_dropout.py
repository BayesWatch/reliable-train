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

def test():
    net = Linear(32,10)
    net.train()
    x = torch.randn(1,32)
    y = net(Variable(x))
    print(y.size())
    net.eval()
    x = torch.randn(1,32)
    y = net(Variable(x))
    print(y.size())


if __name__ == '__main__':
    test()
