import os
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.autograd import Variable

from ..package_core.package_core.net_basics import *
from ..models.transformer_new import *

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


def TFP(spk, channel_step=1):
    num = spk.size(1) // 2
    rep_spk = torch.mean(spk, dim=1).unsqueeze(1)

    for i in range(1, num):
        if i*channel_step < num:
            rep_spk = torch.cat((rep_spk, torch.mean(spk[:, i*channel_step : -i*channel_step, :, :], 1).unsqueeze(1)), 1)
    
    return rep_spk

class ResidualBlock(nn.Module):
    def __init__(self, in_channles, num_channles, use_1x1conv=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channles, num_channles, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(
            num_channles, num_channles, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(
                in_channles, num_channles,kernel_size=1, stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channles)
        self.bn2=nn.BatchNorm2d(num_channles)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        y= F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y+=x
        return F.relu(y)

class DimReduceConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DimReduceConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(out_channels)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
        
def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )
    
class ImageEncoder(nn.Module):
    def __init__(self, in_chs, init_chs, num_resblock=1):
        super(ImageEncoder, self).__init__()
        self.conv0 = conv2d(
            in_planes=in_chs,
            out_planes=init_chs[0],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=7,
            stride=1
        )

        self.conv1 = conv2d(
            in_planes=init_chs[0],
            out_planes=init_chs[1],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks1 = Cascade_resnet_blocks(in_planes=init_chs[1], n_blocks=num_resblock)
        self.conv2 = conv2d(
            in_planes=init_chs[1],
            out_planes=init_chs[2],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks2 = Cascade_resnet_blocks(in_planes=init_chs[2], n_blocks=num_resblock)
        self.conv3 = conv2d(
            in_planes=init_chs[2],
            out_planes=init_chs[3],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks3 = Cascade_resnet_blocks(in_planes=init_chs[3], n_blocks=num_resblock)
        self.conv4 = conv2d(
            in_planes=init_chs[3],
            out_planes=init_chs[4],
            batch_norm=False,
            activation=nn.PReLU(),
            kernel_size=3,
            stride=2
        )
        self.resblocks4 = Cascade_resnet_blocks(in_planes=init_chs[4], n_blocks=num_resblock)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.resblocks1(self.conv1(x0))
        x2 = self.resblocks2(self.conv2(x1))
        x3 = self.resblocks3(self.conv3(x2))
        x4 = self.resblocks4(self.conv4(x3))

        return x4, x3, x2, x1

def predict_img(in_channels):
    return nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)

def predict_img_flow(in_channels):
    return nn.Conv2d(in_channels, 5, kernel_size=3, stride=1, padding=1, bias=True)#first 4: flow; last 1: img

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

class STIRDecorder_top_level(nn.Module):#top level
    def __init__(self, in_chs, hidd_chs):
        super(STIRDecorder_top_level, self).__init__()
        self.hidd_chs = hidd_chs

        self.convrelu = convrelu(in_chs*3, in_chs*3)

        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*3, n_blocks=1)#3
        
        self.predict_img_flow = predict_img_flow(in_chs*3)

    def forward(self, c_cat):
        x0 = c_cat
        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        img_flow_curr = self.predict_img_flow(x_hidd)
        flow_0, flow_1 = img_flow_curr[:,:2], img_flow_curr[:,2:4]
        img_pred = img_flow_curr[:,4:5]

        return img_pred, x_hidd, flow_0, flow_1

class STIRDecorder_bottom_level(nn.Module):#second and third levels
    def __init__(self, in_chs_last, in_chs, hidd_chs, N_group):
        super(STIRDecorder_bottom_level, self).__init__()
        self.hidd_chs = hidd_chs
        self.N_group = N_group

        if self.N_group > 1:
            self.predict_flow_group = nn.Conv2d(in_chs_last*3, 4*(self.N_group-1), kernel_size=3, stride=1, padding=1, bias=True)
            self.deconv_flow_group = deconv(4*(self.N_group-1), 4*(self.N_group-1), kernel_size=4, stride=2, padding=1)

        self.deconv_flow = deconv(4, 4, kernel_size=4, stride=2, padding=1)
        self.deconv_hidden = deconv(3*in_chs_last, self.hidd_chs, kernel_size=4, stride=2, padding=1)

        self.convrelu = DimReduceConv(in_chs*2*self.N_group + in_chs + 4*self.N_group + 1 + self.hidd_chs, in_chs*3)
        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*3, n_blocks=1)#3
        
        self.predict_img = predict_img(in_chs*3)

    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output
        
    def forward(self, img_last, hidden_last, flow_0_last, flow_1_last, upflow_last, c_0, c_1, c_2):
        
        warped_group = []
        if self.N_group > 1:
            flow_group_last = self.predict_flow_group(hidden_last) + torch.cat([upflow_last for _ in range(self.N_group-1)], dim=1) #flow residual
            upflow_group_last = self.deconv_flow_group(flow_group_last)
            for i in range(self.N_group-1):
                warped_group_0 = self.warp(c_0, upflow_group_last[:, 4*i   : 4*i+2])
                warped_group_2 = self.warp(c_2, upflow_group_last[:, 4*i+2 : 4*i+4])
                warped_group.append(warped_group_0)
                warped_group.append(warped_group_2)
        
        upflow = self.deconv_flow(torch.cat([flow_0_last, flow_1_last], dim=1))
        uphidden = self.deconv_hidden(hidden_last)
        upimg = F.interpolate(img_last, scale_factor=2.0, mode='bilinear')
        
        upflow_0, upflow_1 = upflow[:,0:2], upflow[:,2:4]
        
        warp_0 = self.warp(c_0, upflow_0)
        warp_2 = self.warp(c_2, upflow_1)
        
        x0 = torch.cat([c_1, warp_0, warp_2]+ warped_group, dim=1)
        if self.N_group > 1:
            x0 = torch.cat([upimg, x0, uphidden, upflow_0, upflow_1, upflow_group_last], dim=1)
        else:
            x0 = torch.cat([upimg, x0, uphidden, upflow_0, upflow_1], dim=1)
        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        img_pred = self.predict_img(x_hidd)

        return img_pred, x_hidd, upflow_0, upflow_1, upflow_0, upflow_1

class STIRDecorder(nn.Module):#second and third levels
    def __init__(self, in_chs_last, in_chs, hidd_chs):
        super(STIRDecorder, self).__init__()
        self.hidd_chs = hidd_chs

        self.deconv_flow = deconv(4, 4, kernel_size=4, stride=2, padding=1)
        self.deconv_hidden = deconv(3*in_chs_last, self.hidd_chs, kernel_size=4, stride=2, padding=1)

        self.convrelu = DimReduceConv(in_chs*3 + 4 + 1 + self.hidd_chs, in_chs*3)
        self.resblocks = Cascade_resnet_blocks(in_planes=in_chs*3, n_blocks=1)#3
        
        self.predict_img_flow = predict_img_flow(in_chs*3)
        
    def warp(self, img, flow):
        B, _, H, W = flow.shape
        xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1)
        yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
        grid = torch.cat([xx, yy], 1).to(img)
        flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
        grid_ = (grid + flow_).permute(0, 2, 3, 1)
        output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
        return output
        
    def forward(self, img_last, hidden_last, flow_0_last, flow_1_last, c_0, c_1, c_2):
        upflow = self.deconv_flow(torch.cat([flow_0_last, flow_1_last], dim=1))
        uphidden = self.deconv_hidden(hidden_last)
        upimg = F.interpolate(img_last, scale_factor=2.0, mode='bilinear')

        upflow_0, upflow_1 = upflow[:,0:2], upflow[:,2:4]
        
        warp_0 = self.warp(c_0, upflow_0)
        warp_2 = self.warp(c_2, upflow_1)

        x0 = torch.cat([c_1, warp_0, warp_2], dim=1)
        x0 = torch.cat([upimg, x0, uphidden, upflow_0, upflow_1], dim=1)
        x0 = self.convrelu(x0)
        x_hidd = self.resblocks(x0)

        img_flow_curr = self.predict_img_flow(x_hidd)
        flow_0, flow_1 = img_flow_curr[:,:2]+upflow_0, img_flow_curr[:,2:4]+upflow_1
        img_pred = img_flow_curr[:,4:5]

        return img_pred, x_hidd, flow_0, flow_1, upflow_0, upflow_1


##############################Our Model####################################
class STIR(BasicModel):
    def __init__(self, spike_dim = 61,hidd_chs=8, win_r=6, win_step=7):
        super().__init__()

        self.init_chs = [16, 24, 32, 64, 96]
        self.hidd_chs = hidd_chs
        self.spike_dim = spike_dim
        self.attn_num_splits = 1

        self.N_group = 3  
        if spike_dim == 61:
            self.resnet =  ResidualBlock(in_channles=21, num_channles=11, use_1x1conv=True)
            dim_tfp = 16 # 5 + num_channels
        elif spike_dim == 41:
            self.resnet =  ResidualBlock(in_channles=15, num_channles=11, use_1x1conv=True)
            dim_tfp = 15  # 4 + num_channels
        self.encoder = ImageEncoder(in_chs=dim_tfp, init_chs=self.init_chs)
        
        self.transformer = CrossTransformerBlock(dim=self.init_chs[-1], num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias')

        self.decorder_5nd = STIRDecorder_top_level(self.init_chs[-1], self.hidd_chs)
        self.decorder_4nd = STIRDecorder(self.init_chs[-1], self.init_chs[-2], self.hidd_chs)
        self.decorder_3rd = STIRDecorder(self.init_chs[-2], self.init_chs[-3], self.hidd_chs)
        self.decorder_2nd = STIRDecorder(self.init_chs[-3], self.init_chs[-4], self.hidd_chs)
        self.decorder_1st = STIRDecorder_bottom_level(self.init_chs[-4], dim_tfp, self.hidd_chs, self.N_group)
        self.win_r = win_r
        self.win_step = win_step

    def forward(self, x):
        b,_,h,w=x.size()
        if self.spike_dim == 61:
            block1 = x[:, 0  : 21, :, :]
            block2 = x[:, 20 : 41, :, :]
            block3 = x[:, 40 : 61, :, :]
        elif self.spike_dim == 41:
            block1 = x[:, 0  : 15, :, :]
            block2 = x[:, 13 : 28, :, :]
            block3 = x[:, 26 : 41, :, :]

        repre1 = TFP(block1, channel_step=2)#C: 5
        repre2 = TFP(block2, channel_step=2)
        repre3 = TFP(block3, channel_step=2)

        repre_resnet = self.resnet(torch.cat((block1, block2, block3), dim=0))  #[3B, 11, H, W]
        repre1_resnet, repre2_resnet, repre3_resnet = repre_resnet[:b],    repre_resnet[b:2*b],    repre_resnet[2*b:]

        repre1 = torch.cat((repre1, repre1_resnet), 1)#C: 16
        repre2 = torch.cat((repre2, repre2_resnet), 1)
        repre3 = torch.cat((repre3, repre3_resnet), 1)

        concat = torch.cat((repre1, repre2, repre3), dim=0)  
        feature_4, feature_3, feature_2, feature_1 = self.encoder(concat)
        c0_4, c0_3, c0_2, c0_1 = feature_4[:b],    feature_3[:b],    feature_2[:b],    feature_1[:b]
        c1_4, c1_3, c1_2, c1_1 = feature_4[b:2*b], feature_3[b:2*b], feature_2[b:2*b], feature_1[b:2*b]
        c2_4, c2_3, c2_2, c2_1 = feature_4[2*b:],  feature_3[2*b:],  feature_2[2*b:],  feature_1[2*b:]

        c_cat = self.transformer(c1_4, c0_4, c2_4)
        img_pred_4, x_hidd_4, flow_0_4, flow_1_4 = self.decorder_5nd(c_cat)
        img_pred_3, x_hidd_3, flow_0_3, flow_1_3, upflow_0_3, upflow_1_3 = self.decorder_4nd(img_pred_4, x_hidd_4, flow_0_4, flow_1_4, c0_3, c1_3, c2_3)
        img_pred_2, x_hidd_2, flow_0_2, flow_1_2, upflow_0_2, upflow_1_2 = self.decorder_3rd(img_pred_3, x_hidd_3, flow_0_3, flow_1_3, c0_2, c1_2, c2_2)
        img_pred_1, x_hidd_1, flow_0_1, flow_1_1, upflow_0_1, upflow_1_1 = self.decorder_2nd(img_pred_2, x_hidd_2, flow_0_2, flow_1_2, c0_1, c1_1, c2_1)
        img_pred_0, _,               _,        _, upflow_0_0, upflow_1_0 = self.decorder_1st(img_pred_1, x_hidd_1, flow_0_1, flow_1_1, torch.cat((upflow_0_1, upflow_1_1), dim=1), repre1, repre2, repre3)

        if self.training:
            return torch.clamp(img_pred_0, 0, 1),\
                 [torch.clamp(img_pred_0, 0, 1), upflow_0_0, upflow_1_0],\
                 [torch.clamp(img_pred_1, 0, 1), upflow_0_1, upflow_1_1],\
                 [torch.clamp(img_pred_2, 0, 1), upflow_0_2, upflow_1_2],\
                 [torch.clamp(img_pred_3, 0, 1), upflow_0_3, upflow_1_3],\
                 [torch.clamp(img_pred_4, 0, 1)],\
                 [img_pred_0, img_pred_0, img_pred_0]
        else:
            return img_pred_0


