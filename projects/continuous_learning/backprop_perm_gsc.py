# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

from torch import nn

from nupic.research.frameworks.backprop_structure.modules.binary_layers import (
    BinaryGatedConv2d,
    BinaryGatedLinear,
)
from nupic.torch.modules import Flatten, KWinners, KWinners2d


class ToyNet(nn.Module):
    def __init__(self,
                 input_size=(1, 32, 32),
                 n_classes=11,
                 cnn_channels=(64, 64),
                 linear_n=(1000,),
                 cnn_droprate_init=(0.5, 0.5),
                 linear_droprate_init=(0.5, 0.5),
                 l0_strength=(0.5, 0.5),
                 l2_strength=(0.5, 0.5),
                 cnn_pct_on=(0.1, 0.1),
                 linear_pct_on=(0.1,),
                 boost_strength=(1., 1.),
                 boost_strength_factor=(0.9, 0.9),
                 duty_cycle_period=(1000, 1000),
                 batch_norm=True,
                 ):
        super(ToyNet, self).__init__()

        self.cnn_channels = cnn_channels
        self.linear_n = linear_n
        self.cnn_pct_on = cnn_pct_on
        self.linear_pct_on = linear_pct_on
        self.boost_strength = boost_strength
        self.boost_strength_factor = boost_strength_factor
        self.duty_cycle_period = duty_cycle_period
        self.batch_norm = batch_norm

        self.cnn1 = BinaryGatedConv2d(in_channels=input_size[0],
                                      out_channels=cnn_channels[0],
                                      kernel_size=5,
                                      droprate_init=cnn_droprate_init[0],
                                      l0_strength=l0_strength[0],
                                      l2_strength=l2_strength[0],
                                      )

        self.bn1 = nn.BatchNorm2d(cnn_channels[0], affine=False)

        self.mp1 = nn.MaxPool2d(2)

        self.k1 = KWinners2d(channels=cnn_channels[0],
                             percent_on=cnn_pct_on[0],
                             boost_strength=boost_strength[0],
                             boost_strength_factor=boost_strength_factor[0],
                             duty_cycle_period=duty_cycle_period[0],)

        self.flatten = Flatten()

        self.linear1 = BinaryGatedLinear(in_features=self.conv_out(cnn_channels[0]),
                                         out_features=linear_n[0],
                                         droprate_init=linear_droprate_init[0],
                                         l0_strength=l0_strength[1],
                                         l2_strength=l2_strength[1],
                                         )

        self.bn2 = nn.BatchNorm1d(linear_n[0], affine=False)

        self.linear1_k = KWinners(n=linear_n[0],
                                  percent_on=linear_pct_on[0],
                                  # -- NOTE -- replace this if you add 2nd conv layer
                                  boost_strength=boost_strength[1],
                                  boost_strength_factor=boost_strength_factor[1],
                                  duty_cycle_period=duty_cycle_period[1],
                                  )

        self.output = nn.Linear(linear_n[0], n_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Conv component
        x = self.cnn1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.mp1(x)
        x = self.k1(x)
        # flatten
        x = self.flatten(x)
        # Linear component
        x = self.linear1(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.linear1_k(x)

        # output
        x = self.output(x)
        x = self.log_softmax(x)

        return x

    def conv_out(self, cnn_channels):
        if cnn_channels == 64:
            out_size = 12544
        elif cnn_channels == 256:
            out_size = 50176
        return out_size
