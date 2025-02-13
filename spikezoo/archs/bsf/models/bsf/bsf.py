import torch
import torch.nn as nn
import torch.nn.functional as F
from .rep import MODF
from .align import Multi_Granularity_Align


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()

    ####################################################################################
    ## Tools functions for neural networks
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]

    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


from typing import Literal


def split_and_b_cat(x, spike_dim: Literal[41, 61] = 61):
    if spike_dim == 61:
        win_r = 10
        win_step = 10
    elif spike_dim == 41:
        win_r = 6
        win_step = 7
    x0 = x[:, 0 : 2 * win_r + 1, :, :].clone()
    x1 = x[:, win_step : win_step + 2 * win_r + 1, :, :].clone()
    x2 = x[:, 2 * win_step : 2 * win_step + 2 * win_r + 1, :, :].clone()
    x3 = x[:, 3 * win_step : 3 * win_step + 2 * win_r + 1, :, :].clone()
    x4 = x[:, 4 * win_step : 4 * win_step + 2 * win_r + 1, :, :].clone()
    return torch.cat([x0, x1, x2, x3, x4], dim=0)


class Encoder(nn.Module):
    def __init__(self, base_dim=64, layers=4, act=nn.ReLU()):
        super().__init__()
        self.conv_list = nn.ModuleList()
        for ii in range(layers):
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
                    act,
                    nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
                )
            )
        self.act = act

    def forward(self, x):
        for conv in self.conv_list:
            x = self.act(conv(x) + x)
        return x


##########################################################################
class BSF(BasicModel):
    def __init__(self, spike_dim=61, act=nn.ReLU()):
        super().__init__()
        self.spike_dim = spike_dim
        self.offset_groups = 4
        self.corr_max_disp = 3
        if spike_dim == 61:
            self.rep = MODF(in_dim=21,base_dim=64, act=act)
        elif spike_dim == 41:
            self.rep = MODF(in_dim=13,base_dim=64, act=act)
        self.encoder = Encoder(base_dim=64, layers=4, act=act)

        self.align = Multi_Granularity_Align(base_dim=64, groups=self.offset_groups, act=act, sc=3)

        self.recons = nn.Sequential(
            nn.Conv2d(64 * 5, 64 * 3, kernel_size=3, padding=1),
            act,
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            act,
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, input_dict):
        dsft_dict = input_dict["dsft_dict"]
        dsft11 = dsft_dict["dsft11"]
        dsft12 = dsft_dict["dsft12"]
        dsft21 = dsft_dict["dsft21"]
        dsft22 = dsft_dict["dsft22"]

        dsft_b_cat = {
            "dsft11": split_and_b_cat(dsft11, self.spike_dim),
            "dsft12": split_and_b_cat(dsft12, self.spike_dim),
            "dsft21": split_and_b_cat(dsft21, self.spike_dim),
            "dsft22": split_and_b_cat(dsft22, self.spike_dim),
        }

        feat_b_cat = self.rep(dsft_b_cat)
        feat_b_cat = self.encoder(feat_b_cat)
        feat_list = feat_b_cat.chunk(5, dim=0)
        feat_list_align = self.align(feat_list=feat_list)
        out = self.recons(torch.cat(feat_list_align, dim=1))

        return out
