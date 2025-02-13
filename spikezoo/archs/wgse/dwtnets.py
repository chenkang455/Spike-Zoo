import torch.nn as nn
from einops import rearrange
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from pytorch_wavelets import DWT1DForward, DWT1DInverse

from submodules import ResidualBlock


class TcnResidualLayer(nn.Module):
    def __init__(self, in_c, out_c, dilated=1, k=3, s=1, p=1, store_features=False):
        super().__init__()
        self.tcn0 = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=p, dilation=dilated),
            nn.ReLU(),
        )
        self.tcn1 = nn.Sequential(
            nn.Conv1d(out_c, out_c, kernel_size=k, stride=s, padding=p, dilation=dilated),
        )
        self.relu = nn.ReLU(inplace=False)
        self.store_features = store_features
        self.features = {}

    def forward(self, x):
        residual = x
        out = self.tcn0(x)
        if self.store_features:
            self.features['after_tcn0'] = out
        out = self.tcn1(out)
        out = out + residual
        out = self.relu(out)
        return out


class Dwt1dModule_Tcn(nn.Module):
    def __init__(
        self, 
        wvlname='db1', 
        J=3, 
        yl_size=14, 
        yh_size=[26, 18, 14], 
        ks = 3,
        store_features=False
    ):
        super().__init__()
        self.wvlname = wvlname
        self.J = J
        self.yl_num = yl_size
        self.yh_num = yh_size
        self.yh_blocks = nn.ModuleList()

        self.store_features = store_features
        self.features = {}

        for i in self.yh_num:
            self.yh_blocks.append(
                nn.Sequential(
                    TcnResidualLayer(1, 32, store_features=store_features, k=ks, p=ks//2),
                    nn.Conv1d(32, 1, kernel_size=ks, padding=ks//2, dilation=1),
                    nn.ReLU(),
                )
            )
        self.yl_block = nn.Sequential(
            TcnResidualLayer(1, 32, store_features=store_features, k=ks, p=ks//2),
            nn.Conv1d(32, 1, kernel_size=ks, padding=ks//2, dilation=1),
            nn.ReLU(),
        )
        self.dwt = DWT1DForward(wave=self.wvlname, J=self.J)
        self.idwt = DWT1DInverse(wave=self.wvlname)

    def forward(self, x):
        B, T, H, W = x.shape
        x_r = rearrange(x, 'b t h w -> b h w t')
        x_r = rearrange(x_r, 'b h w t -> (b h w) 1 t')

        yl, yh = self.dwt(x_r)
        yl_out = self.yl_block(yl)
        yh_out = []
        for i, yhi in enumerate(yh):
            yhi_out = self.yh_blocks[i](yhi)
            yh_out.append(yhi_out)

        out = self.idwt((yl_out, yh_out)) 
        out = rearrange(out, '(b h w) 1 t -> b h w t', b=B, h=H, w=W)
        out = rearrange(out, 'b h w t -> b t h w')

        return out



class Dwt1dResnetX_TCN(nn.Module):
    def __init__(
        self, 
        wvlname='db8', 
        J=5, 
        yl_size=15, 
        yh_size=[28, 21, 18, 16, 15], 
        num_residual_blocks=3, 
        norm=None, 
        inc=41, 
        ks=3,
        store_features=True
    ):
        super().__init__()

        self.wvl = Dwt1dModule_Tcn(wvlname, J, yl_size, yh_size, store_features=store_features, ks=ks)

        self.norm = norm
        self.num_residual_blocks = num_residual_blocks
        self.resblocks = nn.ModuleList()
        for _ in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(256, 256, norm=self.norm))


        self.conv = nn.Sequential(
            nn.Conv2d(inc if inc%2==0 else inc+1, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )


        self.tail = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
        )

        self.store_features = store_features
        self.features = {}

    def forward(self, x):
        y = self.wvl(x)

        y = self.conv(y)

        for resi, resblock in enumerate(self.resblocks):
            y = resblock(y)

        out = self.tail(y)

        return out
