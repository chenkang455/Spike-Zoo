import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        # self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1) #这个的repeat也是后加的
        # print(self.kernel.shape)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        # print('aaaa')
        # print(img.shape)
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        # return F.conv2d(img, self.kernel, groups=n_channels)
        return F.conv2d(img, self.kernel)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter.repeat(1,3,1,1)) # filter  #这里为什么需要repeat一下？原文的目的是什么？否则不能正常运行
        diff = current - filtered
        return diff

    def forward(self, x, y):
        y = y.repeat(1,3,1,1)
        x = x.repeat(1,3,1,1)
        # print('bbbbbb')
        # print(x.shape)
        # print(y.shape)
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class VGGLoss4(nn.Module):
    def __init__(self, path: str):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, 3, 1, 1),
            # nn.ReLU(inplace=True),
        )
        self.load_state_dict(torch.load(path))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, real_y, fake_y):
        real_y = real_y.repeat((1, 3, 1, 1))
        fake_y = fake_y.repeat((1, 3, 1, 1))
        with torch.no_grad():
            real_f = self.features(real_y)
        fake_f = self.features(fake_y)
        return F.mse_loss(real_f, fake_f)

    
