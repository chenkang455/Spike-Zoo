import numpy as np
from skimage import metrics

class Metrics():

    def __init__(self):
        self.best_psnr = 0.
        self.best_ssim = 0.
        self.best_niqe = 0.

    def Update(self, psnr=0., ssim=0., niqe=0.):
        self.best_psnr = psnr
        self.best_ssim = ssim
        self.best_niqe = niqe

    def GetBestMetrics(self):

        return self.best_psnr, self.best_ssim, self.best_niqe

    def Cal_PSNR(self, preImgs, gtImgs): #shape:[B, H, W]

        B, _, _ = preImgs.shape
        total_psnr = 0.
        for i, (pre, gt) in enumerate(zip(preImgs, gtImgs)):
            print(i+1, metrics.peak_signal_noise_ratio(gt, pre))
            total_psnr += metrics.peak_signal_noise_ratio(gt, pre)

        avg_psnr = total_psnr / B

        return avg_psnr

    def Cal_SSIM(self, preImgs, gtImgs): #shape:[B, H, W]

        B, _, _ = preImgs.shape
        total_ssim = 0.
        for i, (pre, gt) in enumerate(zip(preImgs, gtImgs)):
            total_ssim += metrics.structural_similarity(pre, gt)

        avg_ssim = total_ssim / B

        return avg_ssim


if __name__ == "__main__":

    a = np.random.random((2,256,256))
    b = np.random.random((2,256,256))
    metrics = Metrics()

    print(metrics.Cal_NIQE(a))