import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DeformConv(nn.Module):
    '''
    Deformable Convolution Implementation 
    '''
    def __init__(self, in_channel, out_channel, k_size, pad):
        super(DeformConv, self).__init__()
        self.s = k_size
        self.pad = pad
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=self.s, stride=1, padding=1) 
 
        self.conv_offset = nn.Conv2d(in_channel, 2*self.s*self.s, kernel_size=self.s, stride=1, padding=self.s//2)
        init_offset = torch.Tensor(np.zeros([2*self.s*self.s, in_channel, self.s, self.s]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset) 
 
        self.conv_mask = nn.Conv2d(in_channel, self.s*self.s, kernel_size=self.s, stride=1, padding=self.s//2)
        init_mask = torch.Tensor(np.zeros([self.s*self.s, in_channel, self.s, self.s])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask) 
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) 
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            mask=mask, padding=(self.pad, self.pad))

        return out


class Denoise(nn.Module):
    """
    Denoise Module
    """
    def __init__(self, T):
        super(Denoise, self).__init__()
        self.T = T
        self.dt = 3
        self.size = 7
        
        # Initialize convolution layers dynamically
        self.localconvs = nn.ModuleList()
        for i in range(T):
            # For the first and last layer, use dt-1 channels; for others, use dt channels
            channels = self.dt - 1 if i == 0 or i == T-1 else self.dt
            conv = nn.Conv2d(channels, 1, self.size, padding=self.size // 2)
            self.localconvs.append(conv)

    def forward(self, x):
        outputs = []
        # Apply each convolution to the appropriate slice of x
        for i, conv in enumerate(self.localconvs):
            if i == 0:
                slice = x[:, 0:self.dt-1, ...]
            elif i == self.T-1:
                slice = x[:, -self.dt+1:, ...]
            else:
                slice = x[:, i-1:i+self.dt-1, ...]
            outputs.append(conv(slice))
        # Concatenate all outputs along the channel dimension
        x = torch.cat(outputs, axis=1)
        return x


class MotionEnhance(nn.Module):
    '''
    Motion Enhancement Module 
    '''
    def __init__(self, T):
        super(MotionEnhance, self).__init__()
        k_size = 5
        atten_size = 7
        self.deform_conv = DeformConv(T, 64, k_size, k_size//2)
        self.conv = nn.Conv2d(2, 1, kernel_size=atten_size, padding=atten_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        x = self.deform_conv(x)
        max_result,_ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        attn = torch.cat([max_result, avg_result], 1)
        attn = self.conv(attn)
        map = self.sigmoid(attn)

        return x * map + x


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class DMERNet(nn.Module):
    '''
    DMER Net for Recognition Task
    Based on Binarized Neural Network
    '''
    def __init__(self, block, layers, num_classes=10, T=7):
        super(DMERNet, self).__init__()
        self.inplanes = 64
        self.T = T
        self.key = 1
        if self.key == 0:
            self.module = HardBinaryConv(self.T, 64, kernel_size=7, stride=2, padding=3)
        elif self.key == 1:
            self.module = nn.Sequential(
                Denoise(self.T), 
                MotionEnhance(self.T))

        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.module(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dmer_net18(**kwargs):
    """Constructs a DMER-Net-18 model. """
    model = DMERNet(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def dmer_net34(**kwargs):
    """Constructs a DMER-Net-34 model. """
    model = DMERNet(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model


class RPSNet(nn.Module):
    '''
    Rock-Paper-Scissors Network
    '''
    def __init__(self):
        super(RPSNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=2, padding=2)
        self.bn0 = nn.BatchNorm2d(5)
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=3, padding=3)
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=3, padding=3)
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(480, 3)
        self._initialize_weights()

    def forward(self, x_seq):  # input: batch_size, T, 250, 400
        x = torch.mean(x_seq, dim=1).unsqueeze(dim=1)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.avgpool2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# VGG Models for Spike Streams
def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    T = kwargs.get('T')
    del(kwargs['T'])
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    model.features[0] = nn.Conv2d(in_channels=T, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    nn.init.kaiming_normal_(model.features[0].weight, mode='fan_out', nonlinearity='relu')
    if model.features[0].bias is not None:
        nn.init.constant_(model.features[0].bias, 0)
    return model

def spike_stream_vgg11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)

def spike_stream_vgg13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13_bn', 'A', True, pretrained, progress, **kwargs)

def spike_stream_vgg16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'A', True, pretrained, progress, **kwargs)

def spike_stream_vgg19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19_bn', 'A', True, pretrained, progress, **kwargs)