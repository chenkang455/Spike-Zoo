import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *

class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    ####################################################################################
    ## Tools functions for neural networks
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

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


########################################################################
class SSIR(BasicModel):
    def __init__(self):
        super().__init__()
        base_ch = 128

        self.static_conv = ConvLayerSNN(in_ch=41, out_ch=base_ch, stride=1)

        self.enc1 = ConvLayerSNN(in_ch=base_ch , out_ch=base_ch , stride=2)
        self.eres1 = BottleneckBlockSNN(in_ch=base_ch, out_ch=base_ch)

        self.dec3 = DeConvLayerSNN(in_ch=base_ch, out_ch=base_ch//2)

        self.pred3 = PredHead2(in_ch=base_ch//2 , out_ch=1)

    def forward(self, x):
        # x: B x C x H x W
        x0 = self.static_conv(x)

        x1 = self.eres1(self.enc1(x0))

        x7 = self.dec3(x1)
        out3 = self.pred3(x7)

        if self.training:
            return out3
        else:
            return out3
