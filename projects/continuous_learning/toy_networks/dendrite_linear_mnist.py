import torch
import torch.nn.functional as F
from torch import nn

from nupic.research.frameworks.continuous_learning.dendrite_layers import DendriteLayer
from nupic.torch.modules import Flatten, KWinners, SparseWeights


def hard_sigmoid(x):
    return 1 / (1 + torch.exp(-3 * (x - 1)))


class ToyNetwork(nn.Module):
    def __init__(self,
                 input_dim=28 * 28,
                 num_classes=10,
                 dpc=(2, 2, 2, 2),
                 linear_n=(256, 128, 64,),
                 cat_ws=(0.25, 0.25, 0.25, 0.25),
                 dend_ws=(0.25, 0.25, 0.25, 0.25),
                 output_w_sparsity=0.4,
                 act_fun_type="sigmoid",
                 duty_cycle_period=3000,
                 pct_on=(0.2, 0.2, 0.2, 0.2,),):
        super(ToyNetwork, self).__init__()
        self.num_classes = num_classes
        self.flatten = Flatten()
        self.d1 = DendriteLayer(in_dim=input_dim,
                                out_dim=linear_n[0],
                                dendrites_per_neuron=dpc[0],
                                act_fun_type=act_fun_type,
                                weight_sparsity=dend_ws[0],
                                )
        self.bn2 = nn.BatchNorm1d(linear_n[0])
        self.k1 = KWinners(n=linear_n[0], percent_on=pct_on[0], k_inference_factor=1.,
                           boost_strength=0.,
                           boost_strength_factor=0.,
                           duty_cycle_period=duty_cycle_period,
                           )

        self.d2 = DendriteLayer(in_dim=linear_n[0],
                                out_dim=linear_n[1],
                                dendrites_per_neuron=dpc[1],
                                act_fun_type=act_fun_type,
                                weight_sparsity=dend_ws[1],
                                )
        self.bn3 = nn.BatchNorm1d(linear_n[1])

        self.k2 = KWinners(n=linear_n[1], percent_on=pct_on[1], k_inference_factor=1.,
                           boost_strength=0.,
                           boost_strength_factor=0.,
                           duty_cycle_period=duty_cycle_period,
                           )

        self.d3 = DendriteLayer(in_dim=linear_n[1],
                                out_dim=linear_n[2],
                                dendrites_per_neuron=dpc[2],
                                act_fun_type=act_fun_type,
                                weight_sparsity=dend_ws[2],
                                )

        self.bn4 = nn.BatchNorm1d(linear_n[2])

        self.k3 = KWinners(n=linear_n[2], percent_on=pct_on[2], k_inference_factor=1.,
                           boost_strength=0.,
                           boost_strength_factor=0.,
                           duty_cycle_period=duty_cycle_period,
                           )

        self.dend_output = DendriteLayer(
            in_dim=linear_n[2],
            out_dim=num_classes,
            dendrites_per_neuron=dpc[3],
            act_fun_type=act_fun_type,
            weight_sparsity=dend_ws[3],
        )

        self.kout = KWinners(n=num_classes, percent_on=pct_on[3], k_inference_factor=1.,
                             boost_strength=0.,
                             boost_strength_factor=0.,
                             duty_cycle_period=duty_cycle_period,
                             )

        self.log_softmax = F.log_softmax

        self.cat_layer_0 = SparseWeights(
            nn.Linear(num_classes, linear_n[0] * dpc[0]), weight_sparsity=cat_ws[0],)
        self.cat_layer_1 = SparseWeights(
            nn.Linear(num_classes, linear_n[1] * dpc[1]), weight_sparsity=cat_ws[1],)
        self.cat_layer_2 = SparseWeights(
            nn.Linear(num_classes, linear_n[2] * dpc[2]), weight_sparsity=cat_ws[2],)

        self.cat_layer_output = SparseWeights(
            nn.Linear(num_classes, num_classes * dpc[3]), weight_sparsity=cat_ws[3],)

        self.cat_output = hard_sigmoid

    def forward(self, x, y=None):
        x = self.flatten(x)
        if y is not None:
            yhat = torch.eye(self.num_classes)[y].cuda()

            ypred0 = self.cat_output(self.cat_layer_0(yhat))
            x = self.d1(x, ypred0)
            x = self.bn2(x)
            x = self.k1(x)

            y_pred = self.cat_output(self.cat_layer_1(yhat))
            x = self.d2(x, y_pred)
            x = self.bn3(x)
            x = self.k2(x)

            y_pred1 = self.cat_output(self.cat_layer_2(yhat))
            x = self.d3(x, y_pred1)
            x = self.bn4(x)
            x = self.k3(x)

            y_pred_out = self.cat_output(self.cat_layer_output(yhat))
            x = self.dend_output(x, y_pred_out)
            x = self.kout(x)
            x = self.log_softmax(x, dim=1)

        else:
            x = self.d1(x)
            x = self.bn2(x)
            x = self.d2(x)

            x = self.bn3(x)
            x = self.k2(x)

            x = self.d3(x)
            x = self.bn4(x)
            x = self.k3(x)

            x = self.dend_output(x)
            x = self.kout(x)
            x = self.log_softmax(x, dim=1)
        return x
