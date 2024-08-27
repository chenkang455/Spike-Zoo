import numpy as np
import torch

class DSFT:
    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device


    def spikes2images(self, spikes, max_search_half_window=20):
        '''
        将spikes整体转换为一段DSFT
        
        输入：
        spikes: T x H x W 的numpy张量, 类型: 整型与浮点皆可
        max_search_half_window: 对于要转换为图像的时刻点而言, 左右各参考的最大脉冲帧数量，超过这个数字就不搜了

        输出：
        ImageMatrix: T' x H x W 的numpy张量, 其中T' = T - (2 x max_search_half_window)
        类型: uint8, 取值范围: 0 ~ 255
        '''

        T = spikes.shape[0]
        T_im = T - 2*max_search_half_window

        if T_im < 0:
            raise ValueError('The length of spike stream {:d} is not enough for max_search half window length {:d}'.format(T, max_search_half_window))
        
        spikes = torch.from_numpy(spikes).to(self.device).float()
        ImageMatrix = torch.zeros([T_im, self.spike_h, self.spike_w]).to(self.device)

        pre_idx = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        cur_idx = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        
        for ii in range(T):
            if ii > 0:
                pre_idx[ii] = cur_idx[ii-1]
                cur_idx[ii] = cur_idx[ii-1]
            cur_spk = spikes[ii]
            cur_idx[ii][cur_spk==1] = ii

        diff = cur_idx - pre_idx


        interval = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        for ii in range(T-1, 0-1, -1):
            interval[ii][diff[ii]!=0] = diff[ii][diff[ii]!=0]
            if ii < T-1:
                interval[ii][diff[ii]==0] = interval[ii+1][diff[ii]==0]
        
        # boundary
        interval[interval==-1] = 255
        interval[pre_idx==-1] = 255
        
        # for uint8
        interval = torch.clip(interval, 0, 255)
        
        ImageMatrix = interval[max_search_half_window:-max_search_half_window].cpu().detach().numpy().astype(np.uint8)
        

        return ImageMatrix