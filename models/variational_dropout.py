'''Variational Dropout plugins for PyTorch

Based on the version of variational dropout developed in 
"Variational Dropout Sparsifies Deep Neural Networks"
https://arxiv.org/abs/1701.05369

Code ported from our Tensorflow replication:
https://github.com/BayesWatch/tf-variational-dropout
'''

import math
from functools import reduce

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
        nn.init.xavier_normal(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.)
        self.log_sigma2.data.fill_(-10.) # arbitrary

    def forward(self, input):
        self.log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        thresh = 3 # arbitrary threshold
        if self.training:
            # when training, add noise for variational mc
            mu = F.linear(input, self.weight, None)
            inp2, w2 = torch.pow(input, 2), torch.pow(self.weight, 2)
            si = torch.sqrt(F.linear(inp2,
                            torch.exp(self.log_alpha)*w2, None)+1e-8)
            noise = Variable(torch.randn(mu.size()))
            noise = noise.cuda() if si.is_cuda else noise
            a = mu + si*noise
        else:
            # at test, don't add noise, but mask stuff that goes past the threshold
            self.mask = torch.lt(self.log_alpha, thresh).float()
            a = F.linear(input, self.mask*self.weight, None)
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
        nn.init.xavier_normal(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.)
        self.log_sigma2.data.fill_(-10.)

    def forward(self, input):
        self.log_alpha = get_log_alpha(self.log_sigma2, self.weight)
        thresh = 3 # arbitrary threshold
        if self.training:
            mu = F.conv2d(input, self.weight, None, self.stride, self.padding,
                    self.dilation, self.groups)
            inp2 = torch.pow(input,2)
            w2 = torch.pow(self.weight,2)
            var = F.conv2d(inp2, w2*torch.exp(self.log_alpha), None, self.stride,
                    self.padding, self.dilation, self.groups) + 1e-8
            si = torch.sqrt(var)
            noise = Variable(torch.randn(mu.size()))
            noise = noise.cuda() if si.is_cuda else noise
            a = mu + noise*si
        else:
            self.mask = torch.lt(self.log_alpha, thresh).float()
            a = F.conv2d(input, self.mask*self.weight, None, self.stride,
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

def dkl_qp(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695; C = -k1
    mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) + C
    return -mdkl.sum()

def recent_sparsity(model):
    # on a model that has just performed a test pass
    # will calculate sparsity
    total, active = 0, 0
    for module in model.modules():
        if hasattr(module, 'mask'):
            active += torch.gt(module.mask, 0.5).float().sum().data.cpu().numpy()
            total += reduce(lambda a,b: a*b, module.mask.size())
    return float(active/total)

def variational_regulariser(model, N):
    log_alphas = []
    for module in model.modules():
        if hasattr(module, 'log_alpha'):
            log_alphas.append(module.log_alpha.view(-1))
    return (1./N)*dkl_qp(torch.cat(log_alphas))

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

    print("Layers not broken in an obvious way!")

    print("Training on MNIST to validate")
    from torchvision import datasets, transforms
    import torch.optim as optim

    cuda = torch.cuda.is_available() # use cuda if we have it

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=50, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=100, shuffle=True, **kwargs)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = Conv2d(1, 32, kernel_size=5, padding=2)
            self.conv2 = Conv2d(32, 64, kernel_size=5, padding=2)
            self.fc1 = Linear(7*7*64, 1024)
            self.fc2 = Linear(1024, 10)
            #for m in self.modules():
            #    print("initialising %s"%m)
            #    nn.init.xavier_normal(m.weight)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(-1, 7*7*64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x)

    class FCNet(nn.Module):
        def __init__(self):
            super(FCNet, self).__init__()
            self.fc1 = Linear(28*28, 100)
            self.fc2 = Linear(100, 10)
            #for m in self.modules():
            #    print("initialising %s"%m)
            #    nn.init.xavier_normal(m.weight)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x)

    model = FCNet()
    if cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), 1e-3, eps=1e-6)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target, size_average=True)
            #if batch_idx > 10 or epoch > 1:
            loss += variational_regulariser(model, len(train_loader))
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        sparsity = recent_sparsity(model)
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Sparsity {}\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), sparsity))

    for epoch in range(1, 100):
        train(epoch)
        test()

if __name__ == '__main__':
    test()
