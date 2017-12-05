'''VGG11/13/16/19 in Pytorch.
Added in width multiplier as in https://arxiv.org/pdf/1704.04861.pdf'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, Conv2d=nn.Conv2d, width=1):
        super(VGG, self).__init__()
        self.width = width
        self.features, classifier_size = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(classifier_size, 10)
        self.Conv2d = Conv2d

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = math.ceil(self.width*x)
                layers += [self.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), in_channels

if __name__ == '__main__':
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    print(net(Variable(x)).size())
