import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Resize
import PIL
import os,sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import config as cfg
# import basicblock as B
from SwinTransformer import SwinTransformer

device = cfg.device
# device = 'cpu'

cfa = np.array(
    [[0.5, 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5, -0.5], [0.65, 0.2784, -0.2784, -0.65], [-0.2784, 0.65, -0.65, 0.2764]])

cfa = np.expand_dims(cfa, axis=2)
cfa = np.expand_dims(cfa, axis=3)
cfa = torch.tensor(cfa).float()  # .cuda()
cfa_inv = cfa.transpose(0, 1)

# dwt dec
h0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
h1 = np.array([-1 / math.sqrt(2), 1 / math.sqrt(2)])
h0 = np.array(h0[::-1]).ravel()
h1 = np.array(h1[::-1]).ravel()
h0 = torch.tensor(h0).float().reshape((1, 1, -1))
h1 = torch.tensor(h1).float().reshape((1, 1, -1))
h0_col = h0.reshape((1, 1, -1, 1))  # col lowpass
h1_col = h1.reshape((1, 1, -1, 1))  # col highpass
h0_row = h0.reshape((1, 1, 1, -1))  # row lowpass
h1_row = h1.reshape((1, 1, 1, -1))  # row highpass
ll_filt = torch.cat([h0_row, h1_row], dim=0)

# dwt rec
g0 = np.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
g1 = np.array([1 / math.sqrt(2), -1 / math.sqrt(2)])
g0 = np.array(g0).ravel()
g1 = np.array(g1).ravel()
g0 = torch.tensor(g0).float().reshape((1, 1, -1))
g1 = torch.tensor(g1).float().reshape((1, 1, -1))
g0_col = g0.reshape((1, 1, -1, 1))
g1_col = g1.reshape((1, 1, -1, 1))
g0_row = g0.reshape((1, 1, 1, -1))
g1_row = g1.reshape((1, 1, 1, -1))


class RB(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(RB, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        self.conv = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        c0 = self.conv(x)
        x = self.block(x)
        return x + c0

class NRB(nn.Module):
    def __init__(self, n, in_size, out_size, relu_slope):
        super(NRB, self).__init__()
        nets = []
        nets.append(RB(in_size, out_size, relu_slope))
        for i in range(n-1):
            nets.append(RB(out_size, out_size, relu_slope))
        self.body = nn.Sequential(*nets)

    def forward(self, x):
        return self.body(x)


class ColorTransfer(nn.Module):
    def __init__(self):
        super(ColorTransfer, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa)

    def forward(self, x):
        out = self.net1(x)
        return out


class ColorTransferInv(nn.Module):
    def __init__(self):
        super(ColorTransferInv, self).__init__()
        self.net1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(cfa_inv)

    def forward(self, x):
        out = self.net1(x)
        return out


class FreTransfer(nn.Module):
    def __init__(self):
        super(FreTransfer, self).__init__()
        self.net1 = nn.Conv2d(1, 2, kernel_size=(1, 2), stride=(1, 2), padding=0, bias=None)  # Cin = 1, Cout = 4, kernel_size = (1,2)
        self.net1.weight = torch.nn.Parameter(ll_filt)  # torch.Size([2, 1, 1, 2])

    def forward(self, x):
        B, C, H, W = x.shape
        ll = torch.ones([B, 1, int(H / 2), int(W / 2)], device=device)
        hl = torch.ones([B, 1, int(H / 2), int(W / 2)], device=device)
        lh = torch.ones([B, 1, int(H / 2), int(W / 2)], device=device)
        hh = torch.ones([B, 1, int(H / 2), int(W / 2)], device=device)

        for i in range(C):
            ll_ = self.net1(x[:, i:(i + 1) * 1, :, :])  # 1 * 2 * 128 * 64
            y = []
            for j in range(2):
                weight = self.net1.weight.transpose(2, 3)
                y_out = F.conv2d(ll_[:, j:(j + 1) * 1, :, :], weight, stride=(2, 1), padding=0, bias=None)
                y.append(y_out)
            y_ = torch.cat([y[0], y[1]], dim=1)
            ll[:, i:(i + 1), :, :] = y_[:, 0:1, :, :]
            hl[:, i:(i + 1), :, :] = y_[:, 1:2, :, :]
            lh[:, i:(i + 1), :, :] = y_[:, 2:3, :, :]
            hh[:, i:(i + 1), :, :] = y_[:, 3:4, :, :]

        out = torch.cat([ll, hl, lh, hh], dim=1)
        return out


class FreTransferInv(nn.Module):
    def __init__(self):
        super(FreTransferInv, self).__init__()
        self.net1 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net1.weight = torch.nn.Parameter(g0_col)  # torch.Size([1,1,2,1])
        self.net2 = nn.ConvTranspose2d(1, 1, kernel_size=(2, 1), stride=(2, 1), padding=0, bias=None)
        self.net2.weight = torch.nn.Parameter(g1_col)  # torch.Size([1,1,2,1])

    def forward(self, x):
        lls = x[:, 0:1, :, :]
        hls = x[:, 1:2, :, :]
        lhs = x[:, 2:3, :, :]
        hhs = x[:, 3:4, :, :]
        B, C, H, W = lls.shape
        out = torch.ones([B, C, int(H * 2), int(W * 2)], device=device)
        for i in range(C):
            ll = lls[:, i:i + 1, :, :]
            hl = hls[:, i:i + 1, :, :]
            lh = lhs[:, i:i + 1, :, :]
            hh = hhs[:, i:i + 1, :, :]

            lo = self.net1(ll) + self.net2(hl)  # 1 * 1 * 128 * 64
            hi = self.net1(lh) + self.net2(hh)  # 1 * 1 * 128 * 64
            weight_l = self.net1.weight.transpose(2, 3)
            weight_h = self.net2.weight.transpose(2, 3)
            l = F.conv_transpose2d(lo, weight_l, stride=(1, 2), padding=0, bias=None)
            h = F.conv_transpose2d(hi, weight_h, stride=(1, 2), padding=0, bias=None)
            out[:, i:i + 1, :, :] = l + h
        return out


class ResizeConv(nn.Module):
    def __init__(self):
        super(ResizeConv, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        resize = Resize([H*2, W*2], interpolation=PIL.Image.BICUBIC)
        x = resize(x)
        x1 = self.conv1(x)
        out = self.conv2(x1)
        return out


class Fusion_down(nn.Module):
    def __init__(self):
        super(Fusion_down, self).__init__()
        self.net1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Fusion_up(nn.Module):
    def __init__(self):
        super(Fusion_up, self).__init__()
        self.net1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class Denoise_down(nn.Module):

    def __init__(self):
        super(Denoise_down, self).__init__()
        # self.net1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        # self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        # self.net3 = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        # self.dncnn = DnCNN(in_nc=5, out_nc=4, nc=16, nb=17, act_mode='BR')
        self.nrb = NRB(4, 4, 16, relu_slope=0.2)
        self.st = SwinTransformer(pretrain_img_size=[128, 200], patch_size=1, in_chans=16, embed_dim=96, depths=[2], num_heads=[3], window_size=8)
        self.out = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # net1 = self.net1(x)
        # net2 = self.net2(net1)
        # out = self.net3(net2)
        # out = self.dncnn(x)
        x = self.nrb(x)
        xst = self.st(x)
        out = self.out(xst)
        return out


class Denoise_up(nn.Module):

    def __init__(self, scale=[64, 100]):
        super(Denoise_up, self).__init__()
        self.st = SwinTransformer(pretrain_img_size=scale, patch_size=1, in_chans=5, embed_dim=18, depths=[4], num_heads=[6], window_size=25)

    def forward(self, x):
        out = self.st(x)
        return out


class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.net1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.net2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.net3 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        net1 = F.relu(self.net1(x))
        net2 = F.relu(self.net2(net1))
        out = F.sigmoid(self.net3(net2))
        return out


class VideoDenoise(nn.Module):
    def __init__(self):
        super(VideoDenoise, self).__init__()

        self.fusion = Fusion_down()
        self.denoise = Denoise_down()

    def forward(self, ft1, ft0):

        ll0 = ft0[:, 0:1, :, :]
        ll1 = ft1[:, 0:1, :, :]

        # fusion
        fusion_in = torch.cat([ll0, ll1], dim=1)
        mask = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - mask)) + torch.mul(ft1, mask)
        # print(fusion_out.shape)

        # denoise
        denoise_in = fusion_out
        denoise_out = self.denoise(denoise_in)
        return fusion_out, denoise_out


class MultiVideoDenoise(nn.Module):
    def __init__(self, scale=[64, 100]):
        super(MultiVideoDenoise, self).__init__()
        self.fusion = Fusion_up()
        self.denoise = Denoise_up(scale=scale)

    def forward(self, ft1, ft0, mask_up, denoise_down):
        ll0 = ft0[:, 0:1, :, :]
        ll1 = ft1[:, 0:1, :, :]

        # fusion
        fusion_in = torch.cat([ll0, ll1, mask_up], dim=1)
        mask = self.fusion(fusion_in)
        fusion_out = torch.mul(ft0, (1 - mask)) + torch.mul(ft1, mask)

        # denoise
        denoise_in = torch.cat([fusion_out, denoise_down], dim=1)
        denoise_out = self.denoise(denoise_in)

        return mask, fusion_out, denoise_out[:, 0:4, :, :]


class FPN_Denoise(nn.Module):
    def __init__(self):
        super(FPN_Denoise, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 1, 1, 0)
        self.conv2 = nn.Conv2d(16, 16, 1, 1, 0)
        self.conv3 = nn.Conv2d(16, 16, 1, 1, 0)
        self.conv4 = nn.Conv2d(16, 1, 1, 1, 0)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.conv3(out))
        out = self.lrelu(self.conv4(out))
        return out


class FPN_Refine(nn.Module):
    def __init__(self):
        super(FPN_Refine, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 1, 1, 0)
        self.conv2 = nn.Conv2d(16, 16, 1, 1, 0)
        self.conv3 = nn.Conv2d(16, 16, 1, 1, 0)
        self.conv4 = nn.Conv2d(16, 1, 1, 1, 0)
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.lrelu(self.conv3(out))
        out = F.sigmoid(self.conv4(out))

        return out


class MainDenoise(nn.Module):
    def __init__(self, mode="train"):
        super(MainDenoise, self).__init__()
        self.fpn_denoise = FPN_Denoise()
        self.fpn_refine = FPN_Refine()
        self.ft = FreTransfer()
        self.fti = FreTransferInv()
        self.vd = VideoDenoise()
        self.refine = Refine()
        self.mode = mode

    def transform(self, x):
        # net1 = self.ct(x)
        out = self.ft(x)
        return out

    def transforminv(self, x):
        out = self.fti(x)
        # out = self.cti(x)
        return out

    def forward(self, x, y, q, nd, nl):
        q = q.expand(x.shape[0], 1, 250, 400)
        nd = nd.expand(x.shape[0], 1, 250, 400)
        nl = nl.expand(x.shape[0], 1, 250, 400)

        ft0 = x[:, 0:1, :, :]

        # trans to interval and enter NIM module
        D = 1 / (ft0 + torch.tensor([1e-5]).to(device)) - 1
        D_true = nl / (1/D - nd/q)
        img_true = 1 / (D_true + 1)
        img_fpn = torch.cat([ft0, nd/q], dim=1)
        img_fpn_denoise = self.fpn_denoise(img_fpn) / nl
        fpn_denoise = self.fpn_refine(img_fpn_denoise)

        if x.shape[1] == 1:
            if self.mode == "train":
                ft1 = fpn_denoise
            elif self.mode == "test":
                ft1 = fpn_denoise ** (1/2.2)
        else:
            ft1 = x[:, 1:2, :, :]

        # fpn_denoise = F.pad(fpn_denoise, [0, 0, 3, 3], mode="reflect")
        # ft1 = F.pad(ft1, [0, 0, 3, 3], mode="reflect")
        fgt = F.pad(y, [0, 0, 3, 3], mode="reflect")

        if self.mode == "test":
            fpn_denoise = fpn_denoise ** (1/2.2)
            fgt = fgt ** (1/2.2)

        ft0_d0 = self.transform(fpn_denoise)
        ft1_d0 = self.transform(ft1)
        fgt_d0 = self.transform(fgt)

        # fusion and denoise
        fusion_out, denoise_out = self.vd(ft0_d0, ft1_d0)
        ft_denoise_out_d0 = denoise_out

        # refine
        refine_in = torch.cat([fusion_out, denoise_out], dim=1)
        omega = self.refine(refine_in)
        refine_out = torch.mul(denoise_out, (1 - omega)) + torch.mul(fusion_out, omega)

        fusion_out = self.transforminv(fusion_out)
        refine_out = self.transforminv(refine_out)
        denoise_out = self.transforminv(denoise_out)

        return fpn_denoise, img_true, fusion_out, denoise_out, refine_out, ft_denoise_out_d0, fgt_d0
