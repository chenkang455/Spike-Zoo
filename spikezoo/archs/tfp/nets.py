import torch.nn as nn
import torch


class TFPModel(nn.Module):
    def __init__(self, model_win_length = 41):
        self.window = model_win_length
        super(TFPModel, self).__init__()

    def forward(self, spike):
        mid = spike.shape[1] // 2
        spike = spike[:, mid - self.window // 2 : mid + self.window // 2 + 1]
        return torch.mean(spike, dim=1, keepdim=True)
