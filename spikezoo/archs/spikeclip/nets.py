import torch.nn as nn
import torch

def conv_layer(inDim, outDim, ks, s, p, norm_layer='none'):
    ## convolutional layer
    conv = nn.Conv2d(inDim, outDim, kernel_size=ks, stride=s, padding=p)
    relu = nn.ReLU(True)
    assert norm_layer in ('batch', 'instance', 'none')
    if norm_layer == 'none':
        seq = nn.Sequential(*[conv, relu])
    else:
        if (norm_layer == 'instance'):
            norm = nn.InstanceNorm2d(outDim, affine=False, track_running_stats=False) # instance norm
        else:
            momentum = 0.1
            norm = nn.BatchNorm2d(outDim, momentum = momentum, affine=True, track_running_stats=True)
        seq = nn.Sequential(*[conv, norm, relu])
    return seq

def LRN(inDim=50, outDim=1, norm='none'):  
    convBlock1 = conv_layer(inDim,64,3,1,1)
    convBlock2 = conv_layer(64,128,3,1,1,norm)
    convBlock3 = conv_layer(128,64,3,1,1,norm)
    convBlock4 = conv_layer(64,16,3,1,1,norm)
    conv = nn.Conv2d(16, outDim, 3, 1, 1) 
    seq = nn.Sequential(*[convBlock1, convBlock2, convBlock3, convBlock4, conv])
    return seq

    
from thop import profile
if __name__ == "__main__":
    net = LRN()
    total = sum(p.numel() for p in net.parameters())
    spike = torch.zeros((1,50,250,400))
    flops, _ = profile((net), inputs=(spike,))
    re_msg = (
        "Total params: %.4fM" % (total / 1e6),
        "FLOPs=" + str(flops / 1e9) + '{}'.format("G"),
    )    
    print(re_msg)