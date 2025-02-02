import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel 


def tensor2npy(tensor,normalize = False):
    """Convert the 0-1 torch float tensor to the 0-255 uint numpy array"""
    if tensor.dim() == 4:
        tensor = tensor[0,0]
    tensor = tensor.clip(0, 1).detach().cpu().numpy()
    if normalize == True:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    tensor = 255 * tensor
    return tensor.astype(np.uint8)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count
        
def load_network(load_path, network, strict=False):
    # network multi-gpu training 
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    
    # load .pt or .pth
    if load_path.endswith('.pt') == True:
        load_net = torch.load(load_path)
        if isinstance(load_net, nn.DataParallel) or isinstance(load_net, DistributedDataParallel):
            load_net = load_net.module
        
        if isinstance(load_net,nn.Module):
            load_state = load_net.state_dict() 
        else:
            load_state = load_net
    elif load_path.endswith('.pth') == True:
        load_state = torch.load(load_path)

    # clean multi-gpu state
    load_state_clean = OrderedDict()  
    for k, v in load_state.items():
        if k.startswith('module.'):
            load_state_clean[k[7:]] = v
        else:
            load_state_clean[k] = v
            
    # load the model_weight
    if 'model_state_dict' in load_state_clean.keys():
        network.load_state_dict(load_state_clean['model_state_dict'], strict=strict)
    elif 'model' in load_state_clean.keys():
        network.load_state_dict(load_state_clean['model'], strict=strict)
    else:
        network.load_state_dict(load_state_clean, strict=strict)
    return network


