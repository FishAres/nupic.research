# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Layers with stochastic binary gates for each synapse.

The gate parameters p are learned using the gradient of the expected loss with
respect to those parameters. We use an estimator for this gradient that is based
on using the gradient of the loss wrt the gate's value z to compute a Taylor
approximation of the loss for different values of z, and using this
approximation we compute the expected loss. For Bernoulli random variables the
gradient of this loss simplifies, leading to estimator dE[L]/dp = dL/dz. When
gradient descent sets p to be outside range [0, 1], we allow it to continue
increasing or decreasing, making the gate's value more permanent, and we clamp
it to [0,1] when sampling the gate.
"""

import torch
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules import Module
from torch.nn.modules.utils import _pair as pair
from torch.nn.parameter import Parameter


def sample_weight(p1, p1_unclamped, weight):
    t = (torch.FloatTensor if not torch.cuda.is_available()
         else torch.cuda.FloatTensor)
    u = t(p1.size()).uniform_(0, 1)
    mask = p1 > u

    z = mask.float()

    with torch.no_grad():
        ret = weight * z

    def handle_gradient(grad):
        with torch.no_grad():
            p1_unclamped.backward(grad * weight)
            # Learn only on synapses that weren't gated.
            weight.backward(grad * z)

    ret.requires_grad_()
    ret.register_hook(handle_gradient)

    return ret


class BinaryGatedLinear(Module):
    """
    Linear layer with stochastic binary gates
    """
    def __init__(self, in_features, out_features, l0_strength=1.,
                 l2_strength=1., learn_weight=True, bias=True, droprate_init=0.5,
                 **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param l2_strength: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the gates will be initialized to
        :param l0_strength: Strength of the L0 penalty
        """
        super(BinaryGatedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.l0_strength = l0_strength
        self.l2_strength = l2_strength
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        exc_weight = torch.Tensor(out_features, in_features)
        inh_weight = torch.Tensor(out_features, in_features)
        if learn_weight:
            self.exc_weight = Parameter(exc_weight)
            self.inh_weight = Parameter(inh_weight)
        else:
            self.register_buffer("exc_weight", exc_weight)
            self.register_buffer("inh_weight", inh_weight)

        self.exc_p1 = Parameter(torch.Tensor(out_features, in_features))
        self.inh_p1 = Parameter(torch.Tensor(out_features, in_features))

        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.use_bias = False
        if bias:
            b = torch.Tensor(out_features)
            if learn_weight:
                self.bias = Parameter(b)
            else:
                self.register_buffer("bias", b)
            self.use_bias = True
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.exc_weight, mode="fan_out")
        init.kaiming_normal_(self.inh_weight, mode="fan_out")
        self.exc_weight.data.abs_()
        self.inh_weight.data.abs_()
        self.exc_p1.data.normal_(1 - self.droprate_init, 1e-2)
        self.inh_p1.data.normal_(1 - self.droprate_init, 1e-2)
        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.exc_weight.data.clamp_(min=0.)
        self.inh_weight.data.clamp_(min=0.)

    def get_gate_probabilities(self):
        with torch.no_grad():
            exc_p1 = torch.clamp(self.exc_p1, min=0., max=1.)
            inh_p1 = torch.clamp(self.inh_p1, min=0., max=1.)
        return exc_p1, inh_p1

    def weight_size(self):
        return self.exc_weight.size()

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        exc_p1, inh_p1 = self.get_gate_probabilities()

        exc_p1.requires_grad_()

        def backpropagate_exc(grad):
            self.exc_p1.backward(grad)
        exc_p1.register_hook(backpropagate_exc)

        inh_p1.requires_grad_()

        def backpropagate_inh(grad):
            self.inh_p1.backward(grad)
        inh_p1.register_hook(backpropagate_inh)

        exc_weight_decay_ungated = (
            .5 * self.l2_strength * self.exc_weight.pow(2))
        inh_weight_decay_ungated = (
            .5 * self.l2_strength * self.inh_weight.pow(2))
        exc_weight_l2_l0 = torch.sum(
            (exc_weight_decay_ungated + self.l0_strength) * exc_p1)
        inh_weight_l2_l0 = torch.sum(
            (inh_weight_decay_ungated + self.l0_strength) * inh_p1)
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -exc_weight_l2_l0 - inh_weight_l2_l0 - bias_l2

    def get_inference_mask(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        exc_count1 = exc_p1.sum(dim=1).round().int()
        inh_count1 = inh_p1.sum(dim=1).round().int()

        # pytorch doesn't offer topk with varying k values.
        exc_mask = torch.zeros_like(exc_p1)
        inh_mask = torch.zeros_like(inh_p1)
        for i in range(exc_count1.size()[0]):
            _, exc_indices = torch.topk(exc_p1[i], exc_count1[i].item())
            _, inh_indices = torch.topk(inh_p1[i], inh_count1[i].item())
            exc_mask[i].scatter_(-1, exc_indices, 1)
            inh_mask[i].scatter_(-1, inh_indices, 1)

        return exc_mask, inh_mask

    def sample_weight(self):
        if self.training:
            exc_p1, inh_p1 = self.get_gate_probabilities()
            return (sample_weight(exc_p1, self.exc_p1, self.exc_weight)
                    - sample_weight(inh_p1, self.inh_p1, self.inh_weight))
        else:
            exc_mask, inh_mask = self.get_inference_mask()
            return (exc_mask * self.exc_weight
                    - inh_mask * self.inh_weight)

    def forward(self, x):
        return F.linear(x, self.sample_weight(),
                        (self.bias if self.use_bias else None))

    def get_expected_nonzeros(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        # Flip two coins with probabilities pi_1 and pi_2. What is the
        # probability one of them is 1?
        #
        # 1 - (1 - pi_1)*(1 - pi_2)
        # = 1 - 1 + pi_1 + pi_2 - pi_1*pi_2
        # = pi_1 + pi_2 - pi_1*pi_2
        p1 = exc_p1 + inh_p1 - (exc_p1 * inh_p1)

        return p1.sum(dim=1).detach()

    def get_inference_nonzeros(self):
        exc_mask, inh_mask = self.get_inference_mask()

        return torch.sum(exc_mask.int() | inh_mask.int(), dim=1)

    def count_inference_flops(self):
        # For each unit, multiply with its n inputs then do n - 1 additions.
        # To capture the -1, subtract it, but only in cases where there is at
        # least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies = torch.sum(nz_by_unit)
        adds = multiplies - torch.sum(nz_by_unit > 0)
        return multiplies.item(), adds.item()


class BinaryGatedConv2d(Module):
    """
    Convolutional layer with binary stochastic gates
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, learn_weight=True, bias=True,
                 droprate_init=0.5, l2_strength=1., l0_strength=1.,
                 **kwargs):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: Size of the kernel
        :param stride: Stride for the convolution
        :param padding: Padding for the convolution
        :param dilation: Dilation factor for the convolution
        :param groups: How many groups we will assume in the convolution
        :param bias: Whether we will use a bias
        :param droprate_init: Dropout rate that the gates will be initialized to
        :param l2_strength: Strength of the L2 penalty
        :param l0_strength: Strength of the L0 penalty
        """
        super(BinaryGatedConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.output_padding = pair(0)
        self.groups = groups
        self.l2_strength = l2_strength
        self.l0_strength = l0_strength
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.floatTensor = (torch.FloatTensor if not torch.cuda.is_available()
                            else torch.cuda.FloatTensor)
        self.use_bias = False
        exc_weight = torch.Tensor(out_channels, in_channels // groups,
                                  *self.kernel_size)
        inh_weight = torch.Tensor(out_channels, in_channels // groups,
                                  *self.kernel_size)
        if learn_weight:
            self.exc_weight = Parameter(exc_weight)
            self.inh_weight = Parameter(inh_weight)
        else:
            self.register_buffer("exc_weight", exc_weight)
            self.register_buffer("inh_weight", inh_weight)
        self.exc_p1 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                             *self.kernel_size))
        self.inh_p1 = Parameter(torch.Tensor(out_channels, in_channels // groups,
                                             *self.kernel_size))
        self.dim_z = out_channels
        self.input_shape = None

        if bias:
            b = torch.Tensor(out_channels)
            if learn_weight:
                self.bias = Parameter(b)
            else:
                self.register_buffer("bias", b)
            self.use_bias = True

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.exc_weight, mode="fan_out")
        init.kaiming_normal_(self.inh_weight, mode="fan_out")
        self.exc_weight.data.abs_()
        self.inh_weight.data.abs_()
        self.exc_p1.data.normal_(1 - self.droprate_init, 1e-2)
        self.inh_p1.data.normal_(1 - self.droprate_init, 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.exc_weight.data.clamp_(min=0.)
        self.inh_weight.data.clamp_(min=0.)

    def weight_size(self):
        return self.exc_weight.size()

    def regularization(self):
        """
        Expected L0 norm under the stochastic gates, takes into account and
        re-weights also a potential L2 penalty
        """
        exc_p1, inh_p1 = self.get_gate_probabilities()

        exc_p1.requires_grad_()

        def backpropagate_exc(grad):
            self.exc_p1.backward(grad)
        exc_p1.register_hook(backpropagate_exc)

        inh_p1.requires_grad_()

        def backpropagate_inh(grad):
            self.inh_p1.backward(grad)
        inh_p1.register_hook(backpropagate_inh)

        exc_weight_decay_ungated = (
            .5 * self.l2_strength * self.exc_weight.pow(2))
        inh_weight_decay_ungated = (
            .5 * self.l2_strength * self.inh_weight.pow(2))
        exc_weight_l2_l0 = torch.sum(
            (exc_weight_decay_ungated + self.l0_strength) * exc_p1)
        inh_weight_l2_l0 = torch.sum(
            (inh_weight_decay_ungated + self.l0_strength) * inh_p1)
        bias_l2 = (0 if not self.use_bias
                   else torch.sum(.5 * self.l2_strength * self.bias.pow(2)))
        return -exc_weight_l2_l0 - inh_weight_l2_l0 - bias_l2

    def get_gate_probabilities(self):
        with torch.no_grad():
            exc_p1 = torch.clamp(self.exc_p1, min=0., max=1.)
            inh_p1 = torch.clamp(self.inh_p1, min=0., max=1.)
        return exc_p1, inh_p1

    def get_inference_mask(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        exc_count1 = exc_p1.sum(
            dim=tuple(range(1, len(exc_p1.shape)))
        ).round().int()
        inh_count1 = inh_p1.sum(
            dim=tuple(range(1, len(inh_p1.shape)))
        ).round().int()

        # pytorch doesn't offer topk with varying k values.
        exc_mask = torch.zeros_like(exc_p1)
        inh_mask = torch.zeros_like(inh_p1)
        for i in range(exc_count1.size()[0]):
            _, exc_indices = torch.topk(exc_p1[i].flatten(),
                                        exc_count1[i].item())
            _, inh_indices = torch.topk(inh_p1[i].flatten(),
                                        inh_count1[i].item())
            exc_mask[i].flatten().scatter_(-1, exc_indices, 1)
            inh_mask[i].flatten().scatter_(-1, inh_indices, 1)

        return exc_mask, inh_mask

    def sample_weight(self):
        if self.training:
            exc_p1, inh_p1 = self.get_gate_probabilities()
            return (sample_weight(exc_p1, self.exc_p1, self.exc_weight)
                    - sample_weight(inh_p1, self.inh_p1, self.inh_weight))
        else:
            exc_mask, inh_mask = self.get_inference_mask()
            return (exc_mask * self.exc_weight
                    - inh_mask * self.inh_weight)

    def forward(self, x):
        if self.input_shape is None:
            self.input_shape = x.size()
        return F.conv2d(x, self.sample_weight(),
                        (self.bias if self.use_bias else None),
                        self.stride, self.padding, self.dilation, self.groups)

    def get_expected_nonzeros(self):
        exc_p1, inh_p1 = self.get_gate_probabilities()

        # Flip two coins with probabilities pi_1 and pi_2. What is the
        # probability one of them is 1?
        #
        # 1 - (1 - pi_1)*(1 - pi_2)
        # = 1 - 1 + pi_1 + pi_2 - pi_1*pi_2
        # = pi_1 + pi_2 - pi_1*pi_2
        p1 = exc_p1 + inh_p1 - (exc_p1 * inh_p1)

        return p1.sum(dim=tuple(range(1, len(p1.shape)))).detach()

    def get_inference_nonzeros(self):
        exc_mask, inh_mask = self.get_inference_mask()
        return torch.sum(exc_mask.int() | inh_mask.int(),
                         dim=tuple(range(1, len(exc_mask.shape))))

    def count_inference_flops(self):
        # For each unit, multiply with n inputs then do n - 1 additions.
        # Only subtract 1 in cases where is at least one weight.
        nz_by_unit = self.get_inference_nonzeros()
        multiplies_per_instance = torch.sum(nz_by_unit)
        adds_per_instance = multiplies_per_instance - torch.sum(nz_by_unit > 0)

        # for rows
        instances = (
            (self.input_shape[-2] - self.kernel_size[0]
             + 2 * self.padding[0]) / self.stride[0]) + 1
        # multiplying with cols
        instances *= (
            (self.input_shape[-1] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]) + 1

        multiplies = multiplies_per_instance * instances
        adds = adds_per_instance * instances

        return multiplies.item(), adds.item()