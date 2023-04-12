import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.distributions.normal import Normal


class VIModel(nn.Module):
    """
    a mixin class to attach loss function to layer.
    This is useful when doing variational inference with deep model
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._internal_losses = []
        self.lossScale_factor = 1

    def add_loss(self, func):
        self._internal_losses.append(func)

    def eval_losses(self):
        t_loss = 0

        for l in self._internal_losses:
            t_loss += l(self)

        return t_loss

    def eval_all_losses(self):
        t_loss = 0

        for l in self._internal_losses:
            t_loss += l(self)

        return t_loss


class MeanFieldGaussianFeedForward(VIModel):
    """
    a feed forward neural network with a gaussian prior distribution and a gaussian posterior distribution
    """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 groups=1,
                 weight_prior_sigma=1.,
                 bias_prior_sigma=1.,
                 init_mean_zero=False,
                 init_bias_mean_zero=False,
                 init_prior_sigma_scale=0.01
                 ):
        super(MeanFieldGaussianFeedForward, self).__init__()

        self.samples = {'weights': None, 'bias': None, 'w_noise_state': None, 'b_noise_state': None}

        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weights_mean = Parameter(
            (0.0 if init_mean_zero else 1.0) * torch.rand(out_features, round(in_features / groups)) - 0.5)
        self.log_weights_sigma = Parameter(
            torch.log(
                init_prior_sigma_scale * weight_prior_sigma * torch.ones(out_features, round(in_features / groups))))
        self.noise_source_weights = Normal(torch.zeros(out_features, round(in_features / groups)),
                                           torch.ones(out_features, round(in_features / groups)))
        self.add_loss(lambda s: 0.5 * s.get_sampled_weights().pow(2).sum() / (weight_prior_sigma ** 2))
        self.add_loss(lambda s: -self.out_features / 2 * np.log(2 * np.pi) - 0.5 * s.samples['b_noise_state'].pow(
            2).sum() - self.log_weights_sigma.sum())

        if self.has_bias:
            self.bias_mean = Parameter((0.0 if init_bias_mean_zero else 1.0) * torch.rand(out_features) - 0.5)
            self.log_bias_sigma = Parameter(
                torch.log(init_prior_sigma_scale * bias_prior_sigma * torch.ones(out_features)))

            self.noise_source_bias = Normal(torch.zeros(out_features), torch.ones(out_features))

            self.add_loss(lambda s: 0.5 * s.get_sampled_bias().pow(2).sum() / (bias_prior_sigma ** 2))
            self.add_loss(lambda s: -self.out_features / 2 * np.log(2 * np.pi) - 0.5 * s.samples['b_noise_state'].pow(
                2).sum() - self.log_bias_sigma.sum())

    def sample_transform(self, stochastic=True):
        self.samples['w_noise_state'] = self.noise_source_weights.sample().to(device=self.weights_mean.device)
        self.samples['weights'] = self.weights_mean + (
            torch.exp(self.log_weights_sigma) * self.samples['w_noise_state'] if stochastic else 0)

        if self.has_bias:
            self.samples['b_noise_state'] = self.noise_source_weights.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (
                torch.exp(self.log_bias_sigma) * self.samples['b_noise_state'] if stochastic else 0)

    def get_sampled_weights(self):
        return self.samples['weights']

    def get_sampled_bias(self):
        return self.samples['bias']

    # noinspection PyTypeChecker
    def forward(self, x, stochastic=True):
        self.sample_transform(stochastic)
        return F.linear(x, weight=self.samples['weights'], bias=self.samples['bias'] if self.has_bias else None)


class MeanFieldGaussian2DConvolution(VIModel):
    """
    A Bayesian module that fit a posterior gaussian distribution on a 2D convolution module with normal prior.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 w_prior_sigma=1.,
                 b_prior_sigma=1.,
                 init_mean_zero=False,
                 init_bias_mean_zero=False,
                 init_prior_sigma_scale=0.01):
        super(MeanFieldGaussian2DConvolution, self).__init__()

        self.samples = {'weights': None, 'bias': None, 'w_noise_state': None, 'b_noise_state': None}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode

        self.weights_mean = Parameter((0.0 if init_mean_zero else 1.0)
                                      * torch.rand(out_channels, in_channels // groups, *self.kernel_size) - 0.5)
        self.log_weights_sigma = Parameter(torch.log(init_prior_sigma_scale * w_prior_sigma *
                                                     torch.ones(out_channels, round(in_channels // groups),
                                                                *self.kernel_size)))
        self.noise_source_weights = Normal(torch.zeros(out_channels, in_channels // groups, *self.kernel_size),
                                           torch.ones(out_channels, in_channels // groups, *self.kernel_size))

        self.add_loss(lambda s: 0.5 * s.get_sampled_weights().pow(2).sum() / (w_prior_sigma ** 2))
        self.add_loss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['w_noise_state'].pow(
            2).sum() - self.log_weights_sigma.sum())

        if self.has_bias:
            self.bias_mean = Parameter((0.0 if init_bias_mean_zero else 1.0) * torch.rand(out_channels) - 0.5)
            self.log_bias_sigma = Parameter(
                torch.log(init_prior_sigma_scale * b_prior_sigma * torch.ones(out_channels)))
            self.noise_source_bias = Normal(torch.zeros(out_channels), torch.ones(out_channels))

            self.add_loss(lambda s: 0.5 * s.get_sampled_bias().pow(2).sum() / (b_prior_sigma ** 2))
            self.add_loss(lambda s: -self.out_channels / 2 * np.log(2 * np.pi) - 0.5 * s.samples['b_noise_state'].pow(
                2).sum() - self.log_bias_sigma.sum())

    def sample_transform(self, stochastic=True):
        self.samples['w_noise_state'] = self.noise_source_weights.sample().to(device=self.weights_mean.device)
        self.sample['weights'] = self.weights_mean + (torch.exp(self.log_weights_sigma) * self.samples['w_noise_state']
                                                      if stochastic else 0)

        if self.has_bias:
            self.samples['b_noise_state'] = self.noise_source_bias.sample().to(device=self.bias_mean.device)
            self.samples['bias'] = self.bias_mean + (torch.exp(self.log_bias_sigma) * self.samples['b_noise_state']
                                                     if stochastic else 0)

    def get_sampled_weights(self):
        return self.sample['weights']

    def get_sampled_bias(self):
        return self.sample['bias']

    # noinspection PyTypeChecker
    def forward(self, x, stochastic=True):
        self.sample_transform(stochastic)

        if self.padding != 0 and self.padding != (0, 0):
            pad_kernel = (self.padding, self.padding, self.padding, self.padding) if isinstance(self.padding, int) else \
                (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            mx = F.pad(x, pad_kernel, mode=self.padding_mode, value=0)
        else:
            mx = x

        return F.conv2d(mx, weight=self.samples['weights'], bias=self.samples['bias'] if self.has_bias else None,
                        stride=self.stride, padding=0, dilation=self.dilation, groups=self.groups)


class BayesianMnistNet(VIModel):
    def __init__(self,
                 conv_w_prior_sigma=1.0,
                 conv_b_prior_sigma=5.0,
                 linear_w_prior_sigma=1.0,
                 linear_b_prior_sigma=5.0,
                 p_mc_dropout=0.5):
        super().__init__()

        self.p_mc_dropout = p_mc_dropout

        self.conv1 = MeanFieldGaussian2DConvolution(1, 16, padding=1, w_prior_sigma=conv_w_prior_sigma,
                                                    b_prior_sigma=conv_b_prior_sigma, kernel_size=5,
                                                    init_prior_sigma_scale=1e-7)

        self.conv2 = MeanFieldGaussian2DConvolution(16, 32, padding=1, w_prior_sigma=conv_w_prior_sigma,
                                                    b_prior_sigma=conv_b_prior_sigma, kernel_size=5,
                                                    init_prior_sigma_scale=1e-7)

        self.linear1 = MeanFieldGaussianFeedForward(512, 128, weight_prior_sigma=linear_w_prior_sigma,
                                                    bias_prior_sigma=linear_b_prior_sigma, init_prior_sigma_scale=1e-7)

        self.linear2 = MeanFieldGaussianFeedForward(128, 10, weight_prior_sigma=linear_w_prior_sigma,
                                                    bias_prior_sigma=linear_b_prior_sigma, init_prior_sigma_scale=1e-7)

    def forward(self, x, stochastic=True):
        x = F.relu(F.max_pool2d(self.conv1(x, stochastic=stochastic), 2))
        x = self.conv2(x, stochastic)
        if self.p_mc_dropout is not None:
            x = F.dropout2d(x, p=self.p_mc_dropout, training=stochastic)
        x = F.relu(F.max_pool2d(x, 2))
        x = x.view(-1, 512)
        x = F.relu(self.linear1(x, stochastic=stochastic))
        if self.p_mc_dropout is not None:
            x = F.dropout(x, p=self.p_mc_dropout, training=stochastic)
        x = self.linear2(x, stochastic=stochastic)
        return F.log_softmax(x, dim=1)
