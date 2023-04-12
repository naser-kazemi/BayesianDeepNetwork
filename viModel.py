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
                 weight_prior_mean=0,
                 weight_prior_sigma=1,
                 bias_prior_mean=0,
                 bias_prior_sigma=1,
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
        self.noise_source_weights = Normal()
