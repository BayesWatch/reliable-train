'''
Modification of MobileNet to use a Butterfly 1x1 Convolution.
'''
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from pyinn.modules import Conv2dDepthwise

def is_power(x, power=2, eps=1e-8):
    x = int(x)
    while x > 1:
        if x/power - x/float(power) > eps:
            return False
        x = x/2
    return True

def log_2(x):
    return int(log(x)/log(2.))

class Block(nn.Module):
    expansion = 1
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        # must have power of twos in the numbers of filters
        assert is_power(in_planes) and is_power(out_planes)
        super(Block, self).__init__()
        self.conv1 = Conv2dDepthwise(in_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        n_steps = log_2(in_planes)
        if out_planes == in_planes//2:
            self.expansion = 0.5
        elif out_planes >= in_planes:
            self.expansion = out_planes/in_planes
            n_steps = log_2(self.expansion*in_planes)
        else:
            raise ValueError("Smaller sizes must be a division by two, not %f"%(out_planes/in_planes))
        butterfly_steps = []
        n_groups=1
        for step in range(n_steps):
            if out_planes == in_planes//2 and step == 1-n_steps:
                m = 1
            else:
                m = 2
            m2d = nn.Conv2d(n_groups, m*n_groups, (2,1), stride=(2,1), padding=(0,0), groups=n_groups, bias=False)
            butterfly_steps.append(m2d)
            bn = nn.BatchNorm2d(m*n_groups)
            butterfly_steps.append(bn)
            n_groups = n_groups*m
        self.butterfly = nn.Sequential(*butterfly_steps)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        n, c, h, w = out.size()
        out = out.view(n, 1, c, h*w)
        e = self.expansion
        out = out.repeat(1, 1, int(e) if e>1 else 1, 1) # bit of a hack
        out = F.relu(self.butterfly(out))
        out = out.view(n, int(e*c), h, w)
        return out


class ButterflyNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(ButterflyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ButterflyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, plane_multiplier=4):
        super(ButterflyResNet, self).__init__()
        self.in_planes = 4*2**plane_multiplier
        planes = 4*2**plane_multiplier

        self.conv1 = conv3x3(3,planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.layer1 = self._make_layer(block, planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, planes*8, num_blocks[3], stride=2)
        self.linear = nn.Linear(planes*8*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ButterflyResNet18():
    return ButterflyResNet(Block, [2,2,2,2])


def test():
    net = ButterflyResNet18()
    x = torch.randn(1,3,32,32)
    y = net(Variable(x))
    print(y.size())

if __name__ == '__main__':
    test()
