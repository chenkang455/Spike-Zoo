import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from align_arch import *

class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels=features,
            out_channels=features,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return self.relu3(x + out)


# use Sigmoid
class CALayer2(nn.Module):
    def __init__(self, in_channels):
        super(CALayer2, self).__init__()
        self.ca_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.ca_block(x)
        return weight


# use CALayer
class FeatureExtractor(nn.Module):
    def __init__(
        self, in_channels, features, out_channels, channel_step, num_of_layers=16
    ):
        super(FeatureExtractor, self).__init__()
        # self.InferLayer = LightInferLayer(in_channels=in_channels)
        self.channel_step = channel_step
        self.conv0_0 = nn.Conv2d(
            in_channels=in_channels, out_channels=16, kernel_size=3, padding=1
        )
        self.conv0_1 = nn.Conv2d(
            in_channels=in_channels - 2 * channel_step,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv0_2 = nn.Conv2d(
            in_channels=in_channels - 4 * channel_step,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv0_3 = nn.Conv2d(
            in_channels=in_channels - 6 * channel_step,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.conv1_0 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_1 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.conv1_3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1
        )
        self.ca = CALayer2(in_channels=4)
        self.conv = nn.Conv2d(
            in_channels=4, out_channels=features, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()
        layers = []
        for _ in range(num_of_layers - 2):
            layers.append(BasicBlock(features=features))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out_0 = self.conv1_0(self.relu(self.conv0_0(x)))
        out_1 = self.conv1_1(
            self.relu(self.conv0_1(x[:, self.channel_step : -self.channel_step, :, :]))
        )
        out_2 = self.conv1_2(
            self.relu(
                self.conv0_2(x[:, 2 * self.channel_step : -2 * self.channel_step, :, :])
            )
        )
        out_3 = self.conv1_3(
            self.relu(
                self.conv0_3(x[:, 3 * self.channel_step : -3 * self.channel_step, :, :])
            )
        )
        out = torch.cat((out_0, out_1), 1)
        out = torch.cat((out, out_2), 1)
        out = torch.cat((out, out_3), 1)
        est = out
        weight = self.ca(out)
        out = weight * out
        out = self.conv(out)
        out = self.relu(out)
        tmp = out
        out = self.net(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        # out = self.conv3(out)
        return out + tmp, est


class FusionMaskV1(nn.Module):
    def __init__(self, features):
        super(FusionMaskV1, self).__init__()
        self.conv0 = nn.Conv2d(
            in_channels=2 * features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv1 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=features, out_channels=features, kernel_size=3, padding=1
        )
        self.prelu0 = nn.PReLU()
        self.prelu1 = nn.PReLU()
        self.sig = nn.Sigmoid()

    def forward(self, ref, key):
        fea = torch.cat((ref, key), 1)
        fea = self.conv2(self.prelu1(self.conv1(self.prelu0(self.conv0(fea)))))
        mask = self.sig(fea)
        return mask


# current best model
class SpikeNet(nn.Module):
    def __init__(self, in_channels = 13, features = 64, out_channels = 1, win_r = 6, win_step = 7):
        super(SpikeNet, self).__init__()
        self.extractor = FeatureExtractor(
            in_channels=in_channels,
            features=features,
            out_channels=features,
            channel_step=1,
            num_of_layers=12,
        )
        self.mask0 = FusionMaskV1(features=features)
        self.mask1 = FusionMaskV1(features=features)
        self.mask3 = FusionMaskV1(features=features)
        self.mask4 = FusionMaskV1(features=features)
        self.rec_conv0 = nn.Conv2d(
            in_channels=5 * features,
            out_channels=3 * features,
            kernel_size=3,
            padding=1,
        )
        self.rec_conv1 = nn.Conv2d(
            in_channels=3 * features, out_channels=features, kernel_size=3, padding=1
        )
        self.rec_conv2 = nn.Conv2d(
            in_channels=features, out_channels=1, kernel_size=3, padding=1
        )
        self.rec_relu = nn.ReLU()
        self.pcd_align = Easy_PCD(nf=features, groups=8)
        self.win_r = win_r
        self.win_step = win_step

    def forward(self, x):
        block0 = x[:, 0 : 2 * self.win_r + 1, :, :]
        block1 = x[:, self.win_step : self.win_step + 2 * self.win_r + 1, :, :]
        block2 = x[:, 2 * self.win_step : 2 * self.win_step + 2 * self.win_r + 1, :, :]
        block3 = x[:, 3 * self.win_step : 3 * self.win_step + 2 * self.win_r + 1, :, :]
        block4 = x[:, 4 * self.win_step : 4 * self.win_step + 2 * self.win_r + 1, :, :]
        block0_out, est0 = self.extractor(block0)
        block1_out, est1 = self.extractor(block1)
        block2_out, est2 = self.extractor(block2)
        block3_out, est3 = self.extractor(block3)
        block4_out, est4 = self.extractor(block4)
        aligned_block0_out = self.pcd_align(block0_out, block2_out)
        aligned_block1_out = self.pcd_align(block1_out, block2_out)
        aligned_block3_out = self.pcd_align(block3_out, block2_out)
        aligned_block4_out = self.pcd_align(block4_out, block2_out)
        mask0 = self.mask0(aligned_block0_out, block2_out)
        mask1 = self.mask1(aligned_block1_out, block2_out)
        mask3 = self.mask3(aligned_block3_out, block2_out)
        mask4 = self.mask4(aligned_block4_out, block2_out)
        out = torch.cat((aligned_block0_out * mask0, aligned_block1_out * mask1), 1)
        out = torch.cat((out, block2_out), 1)
        out = torch.cat((out, aligned_block3_out * mask3), 1)
        out = torch.cat((out, aligned_block4_out * mask4), 1)
        out = self.rec_relu(self.rec_conv0(out))
        out = self.rec_relu(self.rec_conv1(out))
        out = self.rec_conv2(out)
        return out


if __name__ == "__main__":
    print("out")
