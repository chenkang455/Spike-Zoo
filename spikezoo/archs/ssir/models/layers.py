import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from spikingjelly.clock_driven import layer, neuron, surrogate

backend = 'torch'
# backend = 'cupy'

def get_neuron_code(neuron_type):
    if neuron_type == 'IF':
        neuron_code = 'neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)'
    elif neuron_type == 'LIF':
        neuron_code = 'neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)'
    elif neuron_type == 'PLIF':
        neuron_code = 'neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)'

    return neuron_code


class ConvLayerSNN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, neuron_type='PLIF'):
        super().__init__()
        
        neuron_code = get_neuron_code(neuron_type)

        self.layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_ch),
        eval(neuron_code)
    )

    def forward(self, x):
        return self.layer(x)


class BottleneckBlockSNN(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, padding=1, neuron_type='PLIF'):
        super().__init__()

        neuron_code = get_neuron_code(neuron_type)

        mid_ch = out_ch // expansion
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.sn1 = eval(neuron_code)

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=kernel_size, padding=padding, stride=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.sn2 = eval(neuron_code)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.sn3 = eval(neuron_code)

    def forward(self, x):
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        out = out + x
        return out


class DeConvLayerSNN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, neuron_type='PLIF'):
        super().__init__()

        neuron_code = get_neuron_code(neuron_type)

        self.layer = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, padding=1, stride=2, bias=False),
        nn.BatchNorm2d(out_ch),
        eval(neuron_code)
    )

    def forward(self, x):
        return self.layer(x)


class PredHead2(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=3, padding=4, stride=1, bias=False),
        nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=0, stride=1, bias=False),
        nn.BatchNorm2d(32)
        )
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=0, stride=1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.conv4(x3)
        return out

