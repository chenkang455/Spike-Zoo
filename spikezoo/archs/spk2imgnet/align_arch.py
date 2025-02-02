""" network architecture for Sakuya """
import torch
from DCNv2 import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d




class PCDAlign(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    """

    def __init__(self, nf=64, groups=8):
        super(PCDAlign, self).__init__()

        # fea1
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for dif
        self.L3_offset_conv2_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack_1 = DCN_sep(
            nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups
        )
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L2_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L2_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack_1 = DCN_sep(
            nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups
        )
        self.L2_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.L1_offset_conv2_1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for offset
        self.L1_offset_conv3_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack_1 = DCN_sep(
            nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups
        )
        self.L1_fea_conv_1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea

        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(
            nf * 2, nf, 3, 1, 1, bias=True
        )  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cas_dcnpack = DCN_sep(
            nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, fea1, fea2):
        """align other neighboring frames to the reference frame in the feature level
        fea1, fea2: [L1, L2, L3], each with [B,C,H,W] features
        fea1 : features of neighboring frame
        fea2 : features of reference (key) frame
        estimate offset bidirectionally
        """
        # param. of fea1
        # L3
        L3_offset = torch.cat([fea1[2], fea2[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1_1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2_1(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack_1(fea1[2], L3_offset))
        # L2
        L2_offset = torch.cat([fea1[1], fea2[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1_1(L2_offset))
        L3_offset = F.interpolate(
            L3_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_offset = self.lrelu(
            self.L2_offset_conv2_1(torch.cat([L2_offset, L3_offset * 2], dim=1))
        )
        L2_offset = self.lrelu(self.L2_offset_conv3_1(L2_offset))
        L2_fea = self.L2_dcnpack_1(fea1[1], L2_offset)
        L3_fea = F.interpolate(
            L3_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L2_fea = self.lrelu(self.L2_fea_conv_1(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([fea1[0], fea2[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1_1(L1_offset))
        L2_offset = F.interpolate(
            L2_offset, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_offset = self.lrelu(
            self.L1_offset_conv2_1(torch.cat([L1_offset, L2_offset * 2], dim=1))
        )
        L1_offset = self.lrelu(self.L1_offset_conv3_1(L1_offset))
        L1_fea = self.L1_dcnpack_1(fea1[0], L1_offset)
        L2_fea = F.interpolate(
            L2_fea, scale_factor=2, mode="bilinear", align_corners=False
        )
        L1_fea = self.L1_fea_conv_1(torch.cat([L1_fea, L2_fea], dim=1))

        # Cascading DCN
        offset = torch.cat([L1_fea, fea2[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))

        return L1_fea


class Easy_PCD(nn.Module):
    def __init__(self, nf=64, groups=8):
        super(Easy_PCD, self).__init__()

        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.pcd_align = PCDAlign(nf=nf, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, f1, f2):
        # input: extracted features
        # f1: feature of neighboring frame
        # f2: feature of the key (reference) frame
        # feature size: f1 = f2 = [B, C, H, W]
        # print(f1.size())
        L1_fea = torch.stack([f1, f2], dim=1)  # [B, 2, C, H, W]
        B, N, C, H, W = L1_fea.size()
        L1_fea = L1_fea.view(-1, C, H, W)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        fea1 = [
            L1_fea[:, 0, :, :, :].clone(),
            L2_fea[:, 0, :, :, :].clone(),
            L3_fea[:, 0, :, :, :].clone(),
        ]
        fea2 = [
            L1_fea[:, 1, :, :, :].clone(),
            L2_fea[:, 1, :, :, :].clone(),
            L3_fea[:, 1, :, :, :].clone(),
        ]
        aligned_fea = self.pcd_align(fea1, fea2)
        return aligned_fea
