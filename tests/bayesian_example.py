#!/usr/bin/env python
# -*- coding: utf-8 -*-
# copy of the example script supplied in the tutorial repo with the goal of
# making sure we still get the same results with the changes made to the layers
"""
Linear Bayesian Model


Karen Ullrich, Christos Louizos, Oct 2017

MIT License

Copyright (c) 2017 Karen Ullrich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# modified by Gavin Gray


# libraries
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
sys.path.append("..")
from models import BayesianLayers
from models.bayesian_utils import compute_compression_rate, \
        compute_reduced_weights, visualize_pixel_importance, \
        generate_gif, visualise_weights

N = 60000.  # number of data points in the training set


def main():
    # import data
    kwargs = {'num_workers': 1, 'pin_memory': True} if FLAGS.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),lambda x: 2 * (x - 0.5),
                       ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), lambda x: 2 * (x - 0.5),
        ])),
        batch_size=FLAGS.batchsize, shuffle=True, **kwargs)

    # for later analysis we take some sample digits
    mask = 255. * (np.ones((1, 28, 28)))
    examples = train_loader.sampler.data_source.train_data[0:5].numpy()
    images = np.vstack([mask, examples])

    # build a simple MLP
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # activation
            self.relu = nn.ReLU()
            # layers
            self.fc1 = BayesianLayers.LinearGroupNJ(28 * 28, 300, clip_var=0.04) 
            self.fc2 = BayesianLayers.LinearGroupNJ(300, 100)
            self.fc3 = BayesianLayers.LinearGroupNJ(100, 10)
            # layers including kl_divergence
            self.kl_list = [self.fc1, self.fc2, self.fc3]

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)

        def get_masks(self,thresholds):
            weight_masks = []
            mask = None
            for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
                # compute dropout mask
                if mask is None:
                    log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                    mask = log_alpha < threshold
                else:
                    mask = np.copy(next_mask)
                try:
                    log_alpha = layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                    next_mask = log_alpha < thresholds[i + 1]
                except:
                    # must be the last mask
                    next_mask = np.ones(10)

                weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
                weight_masks.append(weight_mask.astype(np.float))
            return weight_masks

        def kl_divergence(self):
            KLD = 0
            for layer in self.kl_list:
                KLD += layer.kl_divergence()
            return KLD

    # init model
    model = Net()
    if FLAGS.cuda:
        model.cuda()

    # init optimizer
    optimizer = optim.Adam(model.parameters())

    # we optimize the variational lower bound scaled by the number of data
    # points (so we can keep our intuitions about hyper-params such as the learning rate)
    discrimination_loss = nn.functional.cross_entropy

    def objective(output, target, kl_divergence):
        discrimination_error = discrimination_loss(output, target)
        variational_bound = discrimination_error + kl_divergence / N
        if FLAGS.cuda:
            variational_bound = variational_bound.cuda()
        return variational_bound

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = objective(output, target, model.kl_divergence())
            loss.backward()
            optimizer.step()
            # clip the variances after each step
            for layer in model.kl_list:
                layer.clip_variances()
        print('Epoch: {} \tTrain loss: {:.6f} \t'.format(
            epoch, loss.data[0]))

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if FLAGS.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += discrimination_loss(output, target, size_average=False).data[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        print('Test loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # train the model and save some visualisations on the way
    for epoch in range(1, FLAGS.epochs + 1):
        train(epoch)
        test()
        # visualizations
        weight_mus = [model.fc1.weight_mu, model.fc2.weight_mu]
        log_alphas = [model.fc1.get_log_dropout_rates(), model.fc2.get_log_dropout_rates(),
                      model.fc3.get_log_dropout_rates()]
        visualise_weights(weight_mus, log_alphas, epoch=epoch)
        log_alpha = model.fc1.get_log_dropout_rates().cpu().data.numpy()
        visualize_pixel_importance(images, log_alpha=log_alpha, epoch=str(epoch))

    generate_gif(save='pixel', epochs=FLAGS.epochs)
    generate_gif(save='weight0_e', epochs=FLAGS.epochs)
    generate_gif(save='weight1_e', epochs=FLAGS.epochs)

    # compute compression rate and new model accuracy
    layers = [model.fc1, model.fc2, model.fc3]
    thresholds = FLAGS.thresholds
    compute_compression_rate(layers, model.get_masks(thresholds))

    print("Test error after with reduced bit precision:")

    weights = compute_reduced_weights(layers, model.get_masks(thresholds))
    for layer, weight in zip(layers, weights):
        if FLAGS.cuda:
            layer.post_weight_mu.data = torch.Tensor(weight).cuda()
        else:
            layer.post_weight_mu.data = torch.Tensor(weight)
    for layer in layers: layer.deterministic = True
    test()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--thresholds', type=float, nargs='*', default=[-2.8, -3., -5.])

    FLAGS = parser.parse_args()
    FLAGS.cuda = torch.cuda.is_available()  # check if we can put the net on the GPU
    main()
