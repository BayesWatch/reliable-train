#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities

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

import os
import numpy as np
import imageio

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
cmap = sns.diverging_palette(240, 10, sep=100, as_cmap=True)

# -------------------------------------------------------
# VISUALISATION TOOLS
# -------------------------------------------------------


def visualize_pixel_importance(imgs, log_alpha, epoch="pixel_importance"):
    num_imgs = len(imgs)

    f, ax = plt.subplots(1, num_imgs)
    plt.title("Epoch:" + epoch)
    for i, img in enumerate(imgs):
        img = (img / 255.) - 0.5
        mask = log_alpha.reshape(img.shape)
        mask = 1 - np.clip(np.exp(mask), 0.0, 1)
        ax[i].imshow(img * mask, cmap=cmap, interpolation='none', vmin=-0.5, vmax=0.5)
        ax[i].grid("off")
        ax[i].set_yticks([])
        ax[i].set_xticks([])
    plt.savefig("./.pixel" + epoch + ".png", bbox_inches='tight')
    plt.close()


def visualise_weights(weight_mus, log_alphas, epoch):
    num_layers = len(weight_mus)

    for i in range(num_layers):
        f, ax = plt.subplots(1, 1)
        weight_mu = np.transpose(weight_mus[i].cpu().data.numpy())
        # alpha
        log_alpha_fc1 = log_alphas[i].unsqueeze(1).cpu().data.numpy()
        log_alpha_fc1 = log_alpha_fc1 < -3
        if weight_mu.ndim == 2:
            log_alpha_fc2 = log_alphas[i + 1].unsqueeze(0).cpu().data.numpy()
            log_alpha_fc2 = log_alpha_fc2 < -3
            mask = log_alpha_fc1 + log_alpha_fc2
        else:
            d = log_alpha_fc1.shape[0]
            weight_mu = weight_mu.squeeze()
            mask = np.ones_like(weight_mu)*log_alpha_fc1.reshape(1,d)
        # weight
        c = np.max(np.abs(weight_mu))
        s = ax.imshow(weight_mu * mask, cmap='seismic', interpolation='none', vmin=-c, vmax=c)
        ax.grid("off")
        ax.set_yticks([])
        ax.set_xticks([])
        s.set_clim([-c * 0.5, c * 0.5])
        f.colorbar(s)
        plt.title("Epoch:" + str(epoch))
        plt.savefig("./.weight" + str(i) + '_e' + str(epoch) + ".png", bbox_inches='tight')
        plt.close()


def generate_gif(save='tmp', epochs=10):
    images = []
    filenames = ["./." + save + "%d.png" % (epoch + 1) for epoch in np.arange(epochs)]
    for filename in filenames:
        images.append(imageio.imread(filename))
        os.remove(filename)
    imageio.mimsave('./figures/' + save + '.gif', images, duration=.5)

"""
Compression Tools


Karen Ullrich, Oct 2017

References:

    [1] Michael T. Heath. 1996. Scientific Computing: An Introductory Survey (2nd ed.). Eric M. Munson (Ed.). McGraw-Hill Higher Education. Chapter 1
"""

# -------------------------------------------------------
# General tools
# -------------------------------------------------------

def unit_round_off(t=23):
    """
    :param t:
        number significand bits
    :return:
        unit round off based on nearest interpolation, for reference see [1]
    """
    return 0.5 * 2. ** (1. - t)


SIGNIFICANT_BIT_PRECISION = [unit_round_off(t=i + 1) for i in range(23)]


def float_precision(x):

    out = np.sum([x < sbp for sbp in SIGNIFICANT_BIT_PRECISION])
    return out


def float_precisions(X, dist_fun, layer=1):

    X = X.flatten()
    out = [float_precision(2 * x) for x in X]
    out = np.ceil(dist_fun(out))
    return out


def special_round(input, significant_bit):
    delta = unit_round_off(t=significant_bit)
    rounded = np.floor(input / delta + 0.5)
    rounded = rounded * delta
    return rounded


def fast_infernce_weights(w, exponent_bit, significant_bit):

    return special_round(w, significant_bit)


def compress_matrix(x):

    if len(x.shape) != 2:
        A, B, C, D = x.shape
        x = x.reshape(A * B,  C * D)
        # remove non-necessary filters and rows
        x = x[:, (x > 1e-8).any(axis=0)]
        x = x[(x > 1e-8).any(axis=1), :]
    else:
        # remove unnecessary rows, columns
        x = x[(x > 1e-8).any(axis=1), :]
        x = x[:, (x > 1e-8).any(axis=0)]
    return x


def extract_pruned_params(layers, masks):

    post_weight_mus = []
    post_weight_vars = []

    for i, (layer, mask) in enumerate(zip(layers, masks)):
        # compute posteriors
        post_weight_mu, post_weight_var = layer.compute_posterior_params()
        post_weight_var = post_weight_var.cpu().data.numpy()
        post_weight_mu  = post_weight_mu.cpu().data.numpy()
        # apply mask to mus and variances
        post_weight_mu  = post_weight_mu * mask
        post_weight_var = post_weight_var * mask

        post_weight_mus.append(post_weight_mu)
        post_weight_vars.append(post_weight_var)

    return post_weight_mus, post_weight_vars


# -------------------------------------------------------
#  Compression rates (fast inference scenario)
# -------------------------------------------------------


def _compute_compression_rate(vars, in_precision=32., dist_fun=lambda x: np.max(x), overflow=10e38):
    # compute in  number of bits occupied by the original architecture
    sizes = [v.size for v in vars]
    nb_weights = float(np.sum(sizes))
    IN_BITS = in_precision * nb_weights
    # prune architecture
    vars = [compress_matrix(v) for v in vars]
    sizes = [v.size for v in vars]
    # compute
    significant_bits = [float_precisions(v, dist_fun, layer=k + 1) for k, v in enumerate(vars)]
    exponent_bit = np.ceil(np.log2(np.log2(overflow + 1.) + 1.))
    total_bits = [1. + exponent_bit + sb for sb in significant_bits]
    OUT_BITS = np.sum(np.asarray(sizes) * np.asarray(total_bits))
    return nb_weights / np.sum(sizes), IN_BITS / OUT_BITS, significant_bits, exponent_bit


def compute_compression_rate(layers, masks):
    # reduce architecture
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    # compute overflow level based on maximum weight
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    # compute compression rate
    CR_architecture, CR_fast_inference, _, _ = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)
    #print("Compressing the architecture will decrease the model by a factor of %.1f." % (CR_architecture))
    #print("Making use of weight uncertainty can reduce the model by a factor of %.1f." % (CR_fast_inference))
    return CR_architecture, CR_fast_inference


def compute_reduced_weights(layers, masks):
    weight_mus, weight_vars = extract_pruned_params(layers, masks)
    overflow = np.max([np.max(np.abs(w)) for w in weight_mus])
    _, _, significant_bits, exponent_bits = _compute_compression_rate(weight_vars, dist_fun=lambda x: np.mean(x), overflow=overflow)
    weights = [fast_infernce_weights(weight_mu, exponent_bits, significant_bit) for weight_mu, significant_bit in
               zip(weight_mus, significant_bits)]
    return weights
